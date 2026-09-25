import io
import os
import time
import json
import logging
import argparse
import cv2
import numpy as np
import joblib
import keras

import config
import db
import redis
from broker import MessageBroker, broker

# Suppress noisy TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("workers")

def get_protocol_number(proto_str: str) -> int:
    """Helper to convert protocol name to numerical value."""
    proto_map = {"tcp": 6, "udp": 17, "icmp": 1}
    return proto_map.get(str(proto_str).lower(), 6)

def calculate_severity(risk_score: float) -> str:
    """Maps continuous risk score to categorical severity."""
    if risk_score >= 0.90:
        return "CRITICAL"
    elif risk_score >= 0.70:
        return "HIGH"
    elif risk_score >= 0.40:
        return "MEDIUM"
    return "LOW"

class CNNWorker:
    """
    Dedicated worker for physical anomaly detection (SRS FR-2.1, FR-2.2).
    Preloads spatial_cnn.h5 once at startup and processes frames from video_stream.
    """
    def __init__(self, broker_instance: MessageBroker = None):
        self.broker = broker_instance or broker
        logger.info(f"Loading CNN model from {config.CNN_MODEL_PATH} (startup preload)...")
        self.model = keras.models.load_model(config.CNN_MODEL_PATH, compile=False)
        # Warm-up model so graph tracing happens strictly at startup (SRS FR-2.1)
        dummy_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = self.model(dummy_tensor, training=False)
        logger.info("CNN model loaded and warmed up successfully.")

    def preprocess_frame(self, frame_bytes: bytes) -> np.ndarray:
        """Converts raw frame bytes to 224x224 RGB float32 tensor using OpenCV."""
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = img.astype(np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def process_item(self, payload: bytes) -> dict:
        """Extracts frame metadata and executes CNN inference."""
        start_time = time.perf_counter()
        # Parse payload: [4 bytes meta length] + [meta JSON] + [raw frame bytes]
        meta_len = int.from_bytes(payload[:4], byteorder="big")
        meta_json = payload[4:4 + meta_len].decode("utf-8")
        metadata = json.loads(meta_json)
        frame_bytes = payload[4 + meta_len:]

        # Preprocess frame and run model inference
        tensor = self.preprocess_frame(frame_bytes)
        embedding = self.model(tensor, training=False).numpy().flatten()
        cnn_score = float(np.linalg.norm(embedding) / 10.0)  # Normalized spatial activity

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        source_id = metadata.get("source_id", "camera_default")

        result = {
            "source_id": source_id,
            "timestamp": metadata.get("timestamp", time.time()),
            "cnn_embedding": embedding.tolist(),
            "cnn_score": round(cnn_score, 4),
            "inference_time_ms": round(inference_time_ms, 2)
        }

        # Cache in broker temporal state for ANN fusion
        self.broker.set_temporal_state(
            f"state:video:{source_id}",
            result,
            ttl_seconds=int(config.TEMPORAL_WINDOW_SECONDS * 2)
        )
        return result

    def run(self, stop_event=None):
        logger.info("CNN Worker started. Listening on video_stream queue...")
        while stop_event is None or not stop_event.is_set():
            try:
                item = self.broker.dequeue(config.VIDEO_QUEUE, timeout=1)
                if item:
                    res = self.process_item(item)
                    logger.debug(f"CNN Worker processed frame for {res['source_id']} in {res['inference_time_ms']}ms")
            except redis.exceptions.TimeoutError as te:
                logger.warning(f"CNN Worker Redis timeout reading from queue: {te}. Safely continuing loop...")
            except Exception as e:
                logger.error(f"Error processing video frame: {e}", exc_info=True)

class KNNWorker:
    """
    Dedicated worker for cyber anomaly detection (SRS FR-2.1, FR-2.3).
    Preloads network_knn.pkl once at startup and computes baseline distance.
    """
    def __init__(self, broker_instance: MessageBroker = None):
        self.broker = broker_instance or broker
        logger.info(f"Loading KNN pipeline from {config.KNN_MODEL_PATH} (startup preload)...")
        self.pipeline = joblib.load(config.KNN_MODEL_PATH)
        logger.info("KNN pipeline loaded successfully.")

    def extract_features(self, log_dict: dict) -> np.ndarray:
        """Extracts [src_port, dst_port, packet_size, packet_rate, protocol_num]."""
        proto_num = get_protocol_number(log_dict.get("protocol", "TCP"))
        features = np.array([[
            float(log_dict.get("src_port", 0)),
            float(log_dict.get("dst_port", 0)),
            float(log_dict.get("packet_size", 0)),
            float(log_dict.get("packet_rate", 0.0)),
            float(proto_num)
        ]], dtype=np.float32)
        return features

    def process_item(self, payload: bytes) -> dict:
        """Extracts telemetry and calculates geometric distance to normal cluster."""
        start_time = time.perf_counter()
        log_dict = json.loads(payload.decode("utf-8"))
        features = self.extract_features(log_dict)

        # Scale features and compute nearest neighbor Euclidean distance
        scaler = self.pipeline.named_steps["scaler"]
        knn = self.pipeline.named_steps["knn"]
        scaled = scaler.transform(features)
        distances, _ = knn.kneighbors(scaled)
        knn_distance = float(np.mean(distances))

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        source_id = log_dict.get("source_id", "host_default")

        result = {
            "source_id": source_id,
            "timestamp": log_dict.get("timestamp", time.time()),
            "knn_distance": round(knn_distance, 4),
            "inference_time_ms": round(inference_time_ms, 2)
        }

        # Cache in broker temporal state for ANN fusion
        self.broker.set_temporal_state(
            f"state:network:{source_id}",
            result,
            ttl_seconds=int(config.TEMPORAL_WINDOW_SECONDS * 2)
        )
        return result

    def run(self, stop_event=None):
        logger.info("KNN Worker started. Listening on network_stream queue...")
        while stop_event is None or not stop_event.is_set():
            try:
                item = self.broker.dequeue(config.NETWORK_QUEUE, timeout=1)
                if item:
                    res = self.process_item(item)
                    logger.debug(f"KNN Worker processed telemetry for {res['source_id']} in {res['inference_time_ms']}ms")
            except redis.exceptions.TimeoutError as te:
                logger.warning(f"KNN Worker Redis timeout reading from queue: {te}. Safely continuing loop...")
            except Exception as e:
                logger.error(f"Error processing network log: {e}", exc_info=True)

class ANNFusionWorker:
    """
    Dedicated worker for multimodal risk scoring (SRS FR-2.1, FR-2.4, FR-3.1, FR-3.2).
    Preloads fusion_ann.h5 once at startup and evaluates correlated events within 2.0s.
    """
    def __init__(self, broker_instance: MessageBroker = None):
        self.broker = broker_instance or broker
        logger.info(f"Loading ANN Fusion model from {config.ANN_MODEL_PATH} (startup preload)...")
        self.model = keras.models.load_model(config.ANN_MODEL_PATH, compile=False)
        # Warm-up model so graph tracing happens strictly at startup (SRS FR-2.1)
        dummy_input = np.zeros((1, 17), dtype=np.float32)
        _ = self.model(dummy_input, training=False)
        logger.info("ANN Fusion model loaded and warmed up successfully.")

    def correlate_and_score(self, source_id: str, current_time: float) -> dict:
        """
        Consumes both CNN embedding and KNN distance within configurable temporal window (e.g. 2.0s).
        SRS FR-2.4: Produces unified risk score between 0.0 and 1.0.
        """
        start_time = time.perf_counter()

        video_state = self.broker.get_temporal_state(f"state:video:{source_id}")
        network_state = self.broker.get_temporal_state(f"state:network:{source_id}")

        # Check temporal window (default 2.0 seconds per SRS FR-2.4)
        has_video = False
        cnn_embedding = [0.0] * 16
        cnn_score = 0.0
        if video_state:
            delta = abs(current_time - video_state.get("timestamp", 0))
            if delta <= config.TEMPORAL_WINDOW_SECONDS:
                has_video = True
                cnn_embedding = video_state.get("cnn_embedding", cnn_embedding)
                cnn_score = video_state.get("cnn_score", 0.0)

        has_network = False
        knn_distance = 0.0
        if network_state:
            delta = abs(current_time - network_state.get("timestamp", 0))
            if delta <= config.TEMPORAL_WINDOW_SECONDS:
                has_network = True
                knn_distance = network_state.get("knn_distance", 0.0)

        # Concatenate 16-dim CNN features + 1-dim KNN distance -> 17 dims
        fused_input = np.array([cnn_embedding + [knn_distance]], dtype=np.float32)

        # Run ANN model inference
        risk_score_raw = self.model(fused_input, training=False).numpy().item()
        risk_score = float(np.clip(risk_score_raw, 0.0, 1.0))
        severity = calculate_severity(risk_score)

        e2e_latency_ms = (time.perf_counter() - start_time) * 1000.0

        alert_data = {
            "source_id": source_id,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current_time)),
            "risk_score": round(risk_score, 4),
            "severity": severity,
            "knn_distance": round(knn_distance, 4) if has_network else None,
            "cnn_score": round(cnn_score, 4) if has_video else None,
            "fusion_latency_ms": round(e2e_latency_ms, 2)
        }

        # SRS FR-3.2: Flag events scoring >= 0.7 as Anomalies and write them to database
        if risk_score >= config.ANOMALY_THRESHOLD:
            event_id = db.insert_event(
                source_id=source_id,
                risk_score=risk_score,
                severity=severity,
                knn_distance=knn_distance if has_network else None,
                cnn_score=cnn_score if has_video else None
            )
            alert_data["event_id"] = event_id
            logger.warning(f"ANOMALY DETECTED [ID={event_id}]: {source_id} Risk={risk_score:.4f} ({severity})")

            # SRS FR-3.1: Publish inference results back to Redis pub/sub channel
            self.broker.publish_alert(alert_data)

        return alert_data

    def evaluate_source(self, source_id: str):
        """Triggers fusion risk evaluation for a given source."""
        return self.correlate_and_score(source_id, time.time())

def run_all_workers():
    """Runs CNN, KNN, and Fusion background workers in concurrent threads."""
    import threading

    stop_event = threading.Event()
    cnn_worker = CNNWorker()
    knn_worker = KNNWorker()
    fusion_worker = ANNFusionWorker()

    # Wrap worker loop to trigger fusion correlation on each processed item
    def cnn_loop():
        logger.info("CNN Worker loop active.")
        while not stop_event.is_set():
            try:
                item = broker.dequeue(config.VIDEO_QUEUE, timeout=1)
                if item:
                    res = cnn_worker.process_item(item)
                    fusion_worker.evaluate_source(res["source_id"])
            except redis.exceptions.TimeoutError as te:
                logger.warning(f"CNN Worker loop Redis timeout reading from queue: {te}. Safely continuing loop...")
            except Exception as e:
                logger.error(f"CNN Loop error: {e}", exc_info=True)

    def knn_loop():
        logger.info("KNN Worker loop active.")
        while not stop_event.is_set():
            try:
                item = broker.dequeue(config.NETWORK_QUEUE, timeout=1)
                if item:
                    res = knn_worker.process_item(item)
                    fusion_worker.evaluate_source(res["source_id"])
            except redis.exceptions.TimeoutError as te:
                logger.warning(f"KNN Worker loop Redis timeout reading from queue: {te}. Safely continuing loop...")
            except Exception as e:
                logger.error(f"KNN Loop error: {e}", exc_info=True)

    t_cnn = threading.Thread(target=cnn_loop, daemon=True, name="CNNWorkerThread")
    t_knn = threading.Thread(target=knn_loop, daemon=True, name="KNNWorkerThread")

    t_cnn.start()
    t_knn.start()

    logger.info("All background workers started successfully. Press Ctrl+C to terminate.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down workers...")
        stop_event.set()
        t_cnn.join(timeout=2)
        t_knn.join(timeout=2)
        logger.info("Workers stopped cleanly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time ML/DL Inference Workers")
    parser.add_argument("--worker", choices=["all", "cnn", "knn", "fusion"], default="all", help="Target worker to run")
    args = parser.parse_args()

    if args.worker == "all":
        run_all_workers()
    elif args.worker == "cnn":
        worker = CNNWorker()
        worker.run()
    elif args.worker == "knn":
        worker = KNNWorker()
        worker.run()
    elif args.worker == "fusion":
        worker = ANNFusionWorker()
        logger.info("ANN Fusion worker active.")
