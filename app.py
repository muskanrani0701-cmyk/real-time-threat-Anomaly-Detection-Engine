import os
import time
import logging
from flask import Flask, request, jsonify
import flask
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from pydantic import ValidationError

import config
import db
from broker import broker
from schemas import NetworkLogPayload
import cv2 
from flask  import Response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api")

# Initialize SQLite database schema
db.init_db()

# Create Flask application
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "anomaly-detection-secret-key")
CORS(app)

# Initialize Flask-SocketIO (SRS FR-1.1)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

@app.route("/api/v1/health", methods=["GET"])
def health_check():
    """Health check endpoint reporting API and broker status."""
    return jsonify({
        "status": "healthy",
        "service": "Real-Time Threat & Anomaly Detection Engine",
        "timestamp": time.time(),
        "broker": "connected"
    }), 200

@app.route("/api/v1/network/log", methods=["POST"])
def ingest_network_log():
    """
    FR-1.2: POST endpoint to receive JSON-formatted network telemetry.
    FR-1.3: Strictly NOT running model inference; validates and instantly enqueues to Redis.
    NFR-1: Receive-to-enqueue latency target <= 20ms at p95.
    """
    start_time = time.perf_counter()
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400

    # Validate using Pydantic schema
    try:
        validated = NetworkLogPayload(**data)
    except ValidationError as err:
        return jsonify({"error": "Schema validation failed", "details": err.errors()}), 422

    # Enqueue to Redis network_stream
    enqueue_latency_ms = broker.enqueue_network_log(validated.model_dump())
    total_latency_ms = (time.perf_counter() - start_time) * 1000.0

    return jsonify({
        "status": "enqueued",
        "queue": config.NETWORK_QUEUE,
        "source_id": validated.source_id,
        "enqueue_latency_ms": round(enqueue_latency_ms, 3),
        "total_api_latency_ms": round(total_latency_ms, 3)
    }), 202

@app.route("/api/v1/events", methods=["GET"])
def get_events():
    """Retrieve recent anomaly events flagged and written to database (FR-3.2)."""
    limit = request.args.get("limit", default=50, type=int)
    events = db.get_recent_events(limit=limit)
    return jsonify({
        "count": len(events),
        "events": events
    }), 200

camera = cv2.VideoCapture(0) 

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# WebSocket Ingestion for Video Frames (SRS FR-1.1)
@socketio.on("connect", namespace="/ws/stream/video")
def handle_video_connect():
    logger.info("Video streaming client connected to /ws/stream/video")
    emit("connection_ack", {"status": "connected", "channel": "/ws/stream/video"})

@socketio.on("disconnect", namespace="/ws/stream/video")
def handle_video_disconnect():
    logger.info("Video streaming client disconnected from /ws/stream/video")

@socketio.on("frame", namespace="/ws/stream/video")
def handle_incoming_frame(data):
    """
    FR-1.1: Continuous ingestion of binary frame data via WebSocket.
    FR-1.3: Instantly enqueues to Redis video_stream without performing inference.
    Supports either pure binary payload or dict with binary 'data' and 'source_id'.
    """
    start_time = time.perf_counter()
    if isinstance(data, (bytes, bytearray)):
        frame_bytes = bytes(data)
        source_id = "camera_default"
        metadata = {"timestamp": time.time(), "source_id": source_id}
    elif isinstance(data, dict):
        frame_bytes = data.get("data")
        source_id = data.get("source_id", "camera_default")
        metadata = {
            "timestamp": data.get("timestamp", time.time()),
            "source_id": source_id,
            "frame_index": data.get("frame_index", 0)
        }
        if isinstance(frame_bytes, str):
            # In case client passed base64 or encoded string
            import base64
            try:
                frame_bytes = base64.b64decode(frame_bytes)
            except Exception:
                frame_bytes = frame_bytes.encode("utf-8")
    else:
        emit("frame_ack", {"status": "error", "message": "Unsupported payload format"})
        return

    if not frame_bytes:
        emit("frame_ack", {"status": "error", "message": "Empty frame data"})
        return

    enqueue_latency_ms = broker.enqueue_video_frame(source_id, frame_bytes, metadata)
    total_latency_ms = (time.perf_counter() - start_time) * 1000.0

    emit("frame_ack", {
        "status": "enqueued",
        "source_id": source_id,
        "enqueue_latency_ms": round(enqueue_latency_ms, 3),
        "total_latency_ms": round(total_latency_ms, 3)
    })

if __name__ == "__main__":
    logger.info(f"Starting API ingestion server on {config.HOST}:{config.PORT}")
    socketio.run(app, host=config.HOST, port=config.PORT, debug=config.DEBUG, allow_unsafe_werkzeug=True)
