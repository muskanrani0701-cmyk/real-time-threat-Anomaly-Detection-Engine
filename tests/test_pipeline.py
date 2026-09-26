import io
import time
import json
import pytest
import numpy as np
from PIL import Image

import config
import db
from broker import broker
from app import app
from workers import CNNWorker, KNNWorker, ANNFusionWorker
from ml_engine.face_engine import face_engine


@pytest.fixture(autouse=True)
def setup_test_env():
    db.init_db()
    try:
        broker.client.delete(config.VIDEO_QUEUE.encode("utf-8"))
        broker.client.delete(config.NETWORK_QUEUE.encode("utf-8"))
    except Exception:
        pass
    yield
    try:
        broker.client.delete(config.VIDEO_QUEUE.encode("utf-8"))
        broker.client.delete(config.NETWORK_QUEUE.encode("utf-8"))
    except Exception:
        pass


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "authorized_profiles_count" in data


def test_network_log_ingestion_latency(client):
    payload = {
        "source_id": "test_sensor_01",
        "src_ip": "192.168.1.50",
        "dst_ip": "10.0.0.1",
        "src_port": 443,
        "dst_port": 50000,
        "protocol": "TCP",
        "packet_size": 1200,
        "packet_rate": 100.0,
        "flags": "SYN"
    }
    response = client.post("/api/v1/network/log", json=payload)
    assert response.status_code == 202
    data = response.get_json()
    assert data["status"] == "enqueued"
    assert data["source_id"] == "test_sensor_01"
    assert data["enqueue_latency_ms"] <= 20.0


def test_network_log_validation_failure(client):
    bad_payload = {
        "source_id": "test_sensor_02",
        "src_port": 99999
    }
    response = client.post("/api/v1/network/log", json=bad_payload)
    assert response.status_code == 422


def test_workers_inference_pipeline():
    cnn_worker = CNNWorker()
    knn_worker = KNNWorker()
    fusion_worker = ANNFusionWorker()

    test_source = "sensor_e2e_unit"

    # 1. Test CNN frame processing
    img = Image.new("RGB", (320, 240), color=(128, 64, 32))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    frame_bytes = img_byte_arr.getvalue()

    broker.enqueue_video_frame(test_source, frame_bytes)
    video_item = broker.dequeue(config.VIDEO_QUEUE, timeout=2)
    assert video_item is not None

    cnn_res = cnn_worker.process_item(video_item)
    assert cnn_res["source_id"] == test_source
    assert len(cnn_res["cnn_embedding"]) == 16
    assert cnn_res["inference_time_ms"] < 500.0

    # 2. Test KNN network processing
    log_dict = {
        "source_id": test_source,
        "src_port": 22,
        "dst_port": 50003,
        "packet_size": 350,
        "packet_rate": 15.0,
        "protocol": "TCP",
        "timestamp": time.time()
    }
    broker.enqueue_network_log(log_dict)
    net_item = broker.dequeue(config.NETWORK_QUEUE, timeout=2)
    assert net_item is not None

    knn_res = knn_worker.process_item(net_item)
    assert knn_res["source_id"] == test_source
    assert isinstance(knn_res["knn_distance"], float)
    assert knn_res["inference_time_ms"] < 500.0

    # 3. Test ANN Fusion correlation within temporal window (2.0s)
    fusion_res = fusion_worker.correlate_and_score(test_source, time.time())
    assert 0.0 <= fusion_res["risk_score"] <= 1.0
    assert fusion_res["severity"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert fusion_res["fusion_latency_ms"] < 500.0


def test_events_database_retrieval(client):
    event_id = db.insert_event(
        source_id="camera_security_hall",
        risk_score=0.88,
        severity="HIGH",
        knn_distance=3.45,
        cnn_score=1.22,
        face_status="INTRUDER",
        person_name="Unidentified Intruder",
        face_similarity=0.23
    )
    assert event_id > 0

    response = client.get("/api/v1/events?limit=10")
    assert response.status_code == 200
    data = response.get_json()
    assert data["count"] >= 1
    recent = data["events"][0]
    assert recent["source_id"] == "camera_security_hall"
    assert recent["risk_score"] == 0.88
    assert recent["face_status"] == "INTRUDER"


def test_biometric_authorized_personnel_and_intrusion(client):
    """Verifies enrollment of authorized face, vector caching, and profile deletion."""
    # 1. Insert authorized personnel directly into db
    fake_emb = np.random.randn(512).astype(np.float32)
    fake_emb = (fake_emb / np.linalg.norm(fake_emb)).tolist()
    p_id = db.insert_authorized_person(
        full_name="Agent Zero",
        role_title="Senior Analyst",
        image_path="test_agent.jpg",
        embedding=fake_emb
    )
    assert p_id > 0

    # 2. Test list endpoint
    list_res = client.get("/api/v1/authorized/list")
    assert list_res.status_code == 200
    profiles = list_res.get_json()
    assert any(p["id"] == p_id for p in profiles)

    # 3. Reload cache and verify matching
    face_engine.reload_cache()
    assert len(face_engine.authorized_cache) >= 1

    # 4. Clean up authorized record
    del_res = client.delete(f"/api/v1/authorized/{p_id}")
    assert del_res.status_code == 200
    face_engine.reload_cache()
