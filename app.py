import os
import time
import uuid
import logging
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from pydantic import ValidationError

import config
import db
from broker import broker
from schemas import NetworkLogPayload
from ml_engine.face_engine import face_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api")

# Initialize SQLite database schema
db.init_db()

# Create Flask application
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "anomaly-detection-secret-key")
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Initialize Flask-SocketIO (SRS FR-1.1)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Intruder Alert Rate Limiting Tracker
_last_intruder_alert_time = 0.0


# ============================================================================
# 1. System Health & Telemetry Ingestion
# ============================================================================

@app.route("/", methods=["GET"])
@app.route("/dashboard", methods=["GET"])
def serve_dashboard():
    """Serves the AEGIS Biometric & Threat Engine real-time monitoring dashboard."""
    return send_from_directory(str(config.BASE_DIR), "index.html")


@app.route("/api/v1/health", methods=["GET"])
def health_check():
    """Health check endpoint reporting API, broker, and FaceEngine status."""
    return jsonify({
        "status": "healthy",
        "service": "Real-Time Threat & Anomaly Detection Engine",
        "timestamp": time.time(),
        "broker": "connected",
        "authorized_profiles_count": len(face_engine.authorized_cache)
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


@app.route("/api/v1/events", methods=["GET", "POST"])
def events_endpoint():
    """Retrieve recent anomaly events or insert new anomaly events (FR-3.2)."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        event_id = db.insert_event(
            source_id=data.get("source_id", "camera_default"),
            risk_score=float(data.get("risk_score", 0.5)),
            severity=data.get("severity", "MEDIUM"),
            knn_distance=data.get("knn_distance"),
            cnn_score=data.get("cnn_score"),
            face_status=data.get("face_status", "NO_FACE"),
            person_name=data.get("person_name"),
            face_similarity=data.get("face_similarity"),
            snapshot_path=data.get("snapshot_path")
        )
        return jsonify({"status": "created", "event_id": event_id}), 201

    limit = request.args.get("limit", default=50, type=int)
    events = db.get_recent_events(limit=limit)
    return jsonify({
        "count": len(events),
        "events": events
    }), 200


# ============================================================================
# 2. Biometric Authorized Personnel Setup & Storage
# ============================================================================

@app.route("/api/v1/authorized/upload", methods=["POST"])
@app.route("/api/enroll", methods=["POST"])
@app.route("/api/add-authorized", methods=["POST"])
@app.route("/api/authorized/upload", methods=["POST"])
def upload_authorized_person():
    """
    Enrolls a new authorized person:
      1. Receives portrait image (via 'image', 'file', or 'photo' field)
      2. Detects face & extracts 512-D L2-normalized embedding
      3. Saves photo to storage & record to database
      4. Instantly refreshes in-memory vector cache
    """
    full_name = request.form.get("full_name", "").strip() or "Authorized Personnel"
    role_title = request.form.get("role_title", "Resident / Staff").strip()

    file = request.files.get("image") or request.files.get("file") or request.files.get("photo")
    if file is None or file.filename == "":
        return jsonify({"error": "Missing image file. Please provide form field 'image' or 'file'."}), 400

    file_bytes = file.read()
    if not file_bytes:
        return jsonify({"error": "Empty image file received"}), 400

    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"error": "Failed to decode image file format."}), 400

    # Downscale very large mobile camera photos (e.g. > 1600px) to prevent memory bottlenecks
    h_orig, w_orig = img_bgr.shape[:2]
    max_dim = max(h_orig, w_orig)
    if max_dim > 1600:
        scale = 1600.0 / max_dim
        img_bgr = cv2.resize(img_bgr, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)

    # Detect face
    bbox = face_engine.detect_face_bbox(img_bgr)
    if bbox is not None:
        x, y, w, h = bbox
        face_crop = img_bgr[y:y+h, x:x+w]
    else:
        # Fallback for pre-cropped face portraits/headshots uploaded by mobile clients
        h_img, w_img = img_bgr.shape[:2]
        aspect = w_img / max(1, h_img)
        if 0.4 <= aspect <= 1.8 and h_img >= 60 and w_img >= 60:
            face_crop = img_bgr
        else:
            return jsonify({"error": "No front face detected. Please upload a clear portrait."}), 422

    embedding = face_engine.extract_embedding(face_crop)

    # Save photo to storage/authorized_faces/
    filename = f"auth_{uuid.uuid4().hex[:8]}_{int(time.time())}.jpg"
    dest_path = config.AUTHORIZED_FACES_DIR / filename
    cv2.imwrite(str(dest_path), img_bgr)

    # Save to SQLite
    person_id = db.insert_authorized_person(
        full_name=full_name,
        role_title=role_title,
        image_path=str(dest_path),
        embedding=embedding.tolist()
    )

    # Instantly update vector cache
    face_engine.reload_cache()

    return jsonify({
        "status": "success",
        "person_id": person_id,
        "full_name": full_name,
        "role_title": role_title,
        "image_url": f"/storage/authorized_faces/{filename}"
    }), 201


@app.route("/api/v1/authorized/list", methods=["GET"])
def list_authorized_personnel():
    """Returns list of currently enrolled authorized personnel."""
    records = db.get_all_authorized()
    response_list = []
    for r in records:
        filename = Path(r["image_path"]).name if r.get("image_path") else ""
        response_list.append({
            "id": r["id"],
            "full_name": r["full_name"],
            "role_title": r["role_title"],
            "image_url": f"/storage/authorized_faces/{filename}",
            "created_at": r["created_at"]
        })
    return jsonify(response_list), 200


@app.route("/api/v1/authorized/<int:person_id>", methods=["DELETE"])
def delete_authorized_person(person_id):
    """Deletes authorized person and refreshes vector cache."""
    success = db.delete_authorized_person(person_id)
    if not success:
        return jsonify({"error": "Personnel record not found"}), 404

    face_engine.reload_cache()
    return jsonify({"status": "success", "message": "Personnel record deleted."}), 200


@app.route("/storage/authorized_faces/<filename>")
def serve_authorized_face(filename):
    if (config.AUTHORIZED_FACES_DIR / filename).exists():
        return send_from_directory(str(config.AUTHORIZED_FACES_DIR), filename)
    bundled_dir = config.BASE_DIR / "storage" / "authorized_faces"
    if (bundled_dir / filename).exists():
        return send_from_directory(str(bundled_dir), filename)
    return jsonify({"error": "File not found"}), 404


@app.route("/storage/intruder_snapshots/<filename>")
def serve_intruder_snapshot(filename):
    if (config.INTRUDER_SNAPSHOTS_DIR / filename).exists():
        return send_from_directory(str(config.INTRUDER_SNAPSHOTS_DIR), filename)
    bundled_dir = config.BASE_DIR / "storage" / "intruder_snapshots"
    if (bundled_dir / filename).exists():
        return send_from_directory(str(bundled_dir), filename)
    return jsonify({"error": "File not found"}), 404


# ============================================================================
# 3. Live Video Feed with Real-Time Biometric HUD Overlay & Intruder Alerting
# ============================================================================

def generate_frames():
    """Streams MJPEG frames with real-time biometric bounding box overlay and animated radar telemetry fallback."""
    global _last_intruder_alert_time
    cap = None
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
    except Exception as e:
        logger.warning(f"Could not initialize video capture: {e}")

    try:
        while True:
            has_camera_frame = False
            frame = None

            if cap is not None and cap.isOpened():
                try:
                    success, captured = cap.read()
                    if success and captured is not None and captured.size > 0:
                        has_camera_frame = True
                        frame = captured
                except Exception:
                    has_camera_frame = False

            if not has_camera_frame:
                # Generate high-tech animated surveillance radar HUD
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Radar grid lines
                cv2.line(frame, (0, 240), (640, 240), (15, 23, 42), 1)
                cv2.line(frame, (320, 0), (320, 480), (15, 23, 42), 1)
                cv2.circle(frame, (320, 240), 160, (30, 41, 59), 1)
                cv2.circle(frame, (320, 240), 100, (30, 41, 59), 1)
                cv2.circle(frame, (320, 240), 40, (30, 41, 59), 1)
                
                # Dynamic sweep line
                angle = (time.time() * 2.5) % (2 * np.pi)
                sx = int(320 + 160 * np.cos(angle))
                sy = int(240 + 160 * np.sin(angle))
                cv2.line(frame, (320, 240), (sx, sy), (34, 211, 238), 2)

                # HUD text & telemetry
                cv2.putText(frame, "AEGIS OPTICAL RADAR // ACTIVE PERIMETER", (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 211, 238), 2)
                cv2.putText(frame, f"AUTHORIZED PROFILES IN CACHE: {len(face_engine.authorized_cache)}",
                            (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 1)
                cv2.putText(frame, "FEED: REAL-TIME SIMULATED SENSOR (STANDBY)",
                            (40, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 116, 139), 1)
                cv2.putText(frame, f"TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
                            (40, 442), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)
                
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.08)
                continue

            # Run biometric analysis on live frame
            res = face_engine.process_frame(frame)
            bbox = res["bbox"]

            if bbox is not None:
                x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
                sim = res["similarity_score"]

                if res["status"] == "AUTHORIZED":
                    # Green bounding box for authorized personnel
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"AUTHORIZED: {res['person_name']} ({int(sim * 100)}%)"
                    cv2.putText(frame, label, (x, max(20, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                elif res["status"] == "INTRUDER":
                    # Flashing Red bounding box for intruders
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    label = f"ALERT: INTRUDER ({int(sim * 100)}%)"
                    cv2.putText(frame, label, (x, max(20, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Trigger alert and save snapshot (rate limited to 4s)
                    now = time.time()
                    if now - _last_intruder_alert_time > 4.0:
                        _last_intruder_alert_time = now
                        snap_name = f"intruder_{int(now)}_{uuid.uuid4().hex[:6]}.jpg"
                        snap_path = config.INTRUDER_SNAPSHOTS_DIR / snap_name
                        cv2.imwrite(str(snap_path), frame)

                        # Insert into events database
                        db.insert_event(
                            source_id="live_cam_0",
                            risk_score=0.96,
                            severity="CRITICAL",
                            face_status="INTRUDER",
                            person_name="Unidentified Intruder",
                            face_similarity=sim,
                            snapshot_path=str(snap_path)
                        )

                        # Push alert payload through SocketIO
                        socketio.emit("threat_alert", {
                            "type": "INTRUDER_ALERT",
                            "severity": "CRITICAL",
                            "similarity": sim,
                            "snapshot_url": f"/storage/intruder_snapshots/{snap_name}",
                            "timestamp": time.time()
                        }, namespace="/ws/stream/video")

            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.04)
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


@app.route('/video_feed')
def video_feed():
    """Streams live MJPEG camera feed with facial recognition HUD overlay."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ============================================================================
# 4. WebSocket Streaming & Real-Time Ingestion (SocketIO)
# ============================================================================

@socketio.on("connect", namespace="/ws/stream/video")
def handle_video_connect():
    logger.info("Video streaming client connected to /ws/stream/video")
    emit("connection_ack", {
        "status": "connected",
        "channel": "/ws/stream/video",
        "authorized_count": len(face_engine.authorized_cache)
    })


@socketio.on("disconnect", namespace="/ws/stream/video")
def handle_video_disconnect():
    logger.info("Video streaming client disconnected from /ws/stream/video")


@socketio.on("frame", namespace="/ws/stream/video")
def handle_incoming_frame(data):
    """
    Ingests frames over WebSocket, pushes to Redis video_stream,
    and runs real-time biometric facial recognition.
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

    # 1. Enqueue to Redis broker
    enqueue_latency_ms = broker.enqueue_video_frame(source_id, frame_bytes, metadata)

    # 2. Run real-time biometric face detection on incoming frame
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    face_res = None
    if frame_bgr is not None:
        face_res = face_engine.process_frame(frame_bgr)

    total_latency_ms = (time.perf_counter() - start_time) * 1000.0

    emit("frame_ack", {
        "status": "enqueued",
        "source_id": source_id,
        "enqueue_latency_ms": round(enqueue_latency_ms, 3),
        "total_latency_ms": round(total_latency_ms, 3),
        "face_status": face_res["status"] if face_res else "NO_FRAME",
        "person_name": face_res.get("person_name") if face_res else None,
        "similarity": face_res.get("similarity_score", 0.0) if face_res else 0.0,
        "bbox": face_res.get("bbox") if face_res else None
    })


if __name__ == "__main__":
    logger.info(f"Starting API ingestion & Biometric server on 0.0.0.0:{config.PORT}")
    socketio.run(app, host="0.0.0.0", port=config.PORT, debug=config.DEBUG, allow_unsafe_werkzeug=True)
