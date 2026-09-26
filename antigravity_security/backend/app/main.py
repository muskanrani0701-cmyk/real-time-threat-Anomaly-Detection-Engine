"""
Project Antigravity: Main FastAPI Application
Handles User Authentication, Authorized Personnel Management,
and High-Throughput Real-Time WebSocket Video Frame Analysis.
"""

import os
import io
import time
import base64
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
    WebSocket,
    WebSocketDisconnect,
    Query
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from jose import jwt, JWTError

from .config import settings
from .database import engine, get_db, Base
from .models import User, AuthorizedPersonnel, IntruderAlert
from .schemas import (
    UserRegister,
    UserLogin,
    Token,
    AuthorizedPersonnelResponse,
    IntruderAlertResponse,
    DetectionEvent,
    FrameAnalysisRequest
)
from .auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
    get_current_user_optional
)
from .face_engine import face_engine

# Initialize database schema
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Real-time facial recognition and intrusion detection backend."
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static storage for uploaded face portraits & intruder snapshots
app.mount("/storage", StaticFiles(directory=str(settings.STORAGE_DIR)), name="storage")


@app.get("/")
@app.get("/dashboard")
def serve_dashboard():
    """Serves the AEGIS Biometric & Threat Engine real-time monitoring dashboard."""
    index_path = Path(__file__).resolve().parent.parent.parent.parent / "index.html"
    if not index_path.exists():
        index_path = Path("index.html").resolve()
    return FileResponse(str(index_path))


@app.get("/api/v1/events")
def get_events(limit: int = 50, db: Session = Depends(get_db)):
    """Telemetry and anomaly intrusion events for dashboard sync."""
    alerts = db.query(IntruderAlert).order_by(IntruderAlert.timestamp.desc()).limit(limit).all()
    events = []
    for a in alerts:
        events.append({
            "id": a.id,
            "captured_at": a.timestamp.isoformat(),
            "source_id": a.camera_id or "CAM_01_OPTICAL",
            "risk_score": 0.96,
            "severity": "CRITICAL",
            "face_status": "INTRUDER",
            "person_name": "Unidentified Intruder",
            "face_similarity": a.highest_similarity,
            "snapshot_path": a.snapshot_path
        })
    return {"count": len(events), "events": events}


def generate_fastapi_frames():
    """Streams MJPEG frames with real-time biometric analysis and simulated radar HUD."""
    cap = None
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
    except Exception:
        pass

    try:
        while True:
            has_camera = False
            frame = None
            if cap is not None and cap.isOpened():
                try:
                    ok, captured = cap.read()
                    if ok and captured is not None and captured.size > 0:
                        has_camera = True
                        frame = captured
                except Exception:
                    has_camera = False

            if not has_camera:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.line(frame, (0, 240), (640, 240), (15, 23, 42), 1)
                cv2.line(frame, (320, 0), (320, 480), (15, 23, 42), 1)
                cv2.circle(frame, (320, 240), 160, (30, 41, 59), 1)
                cv2.circle(frame, (320, 240), 100, (30, 41, 59), 1)
                angle = (time.time() * 2.5) % (2 * np.pi)
                sx = int(320 + 160 * np.cos(angle))
                sy = int(240 + 160 * np.sin(angle))
                cv2.line(frame, (320, 240), (sx, sy), (34, 211, 238), 2)
                cv2.putText(frame, "AEGIS LIVE PERIMETER MONITORING", (60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 211, 238), 2)
                cv2.putText(frame, f"UTC TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}",
                            (60, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)
                cv2.putText(frame, "STATUS: ARMED & ACTIVE",
                            (60, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 1)
                ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                time.sleep(0.08)
                continue

            ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.04)
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


@app.get("/video_feed")
def live_video_feed():
    """Streams live MJPEG camera feed."""
    return StreamingResponse(
        generate_fastapi_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================================
# 1. Onboarding & Authentication
# ============================================================================

@app.post("/api/v1/auth/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register_user(payload: UserRegister, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    hashed_pw = get_password_hash(payload.password)
    user = User(
        email=payload.email,
        hashed_password=hashed_pw,
        full_name=payload.full_name
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(data={"sub": user.email, "user_id": user.id})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_name": user.full_name,
        "user_email": user.email
    }


@app.post("/api/v1/auth/login", response_model=Token)
def login_user(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_access_token(data={"sub": user.email, "user_id": user.id})
    # Warm up in-memory vector cache for this user
    authorized_list = db.query(AuthorizedPersonnel).filter(AuthorizedPersonnel.user_id == user.id).all()
    face_engine.reload_user_embeddings(user.id, authorized_list)

    return {
        "access_token": token,
        "token_type": "bearer",
        "user_name": user.full_name,
        "user_email": user.email
    }


@app.get("/api/v1/auth/me")
def get_current_user_profile(user: User = Depends(get_current_user)):
    return {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "created_at": user.created_at
    }


# ============================================================================
# 2. Authorized Personnel Setup (Real-Time Embedding Extraction & Database Save)
# ============================================================================

@app.post("/api/v1/authorized/upload", response_model=AuthorizedPersonnelResponse)
@app.post("/api/enroll", response_model=AuthorizedPersonnelResponse)
@app.post("/api/add-authorized", response_model=AuthorizedPersonnelResponse)
@app.post("/api/authorized/upload", response_model=AuthorizedPersonnelResponse)
async def upload_authorized_person(
    full_name: Optional[str] = Form("Authorized Personnel"),
    role_title: Optional[str] = Form("Resident / Staff"),
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    photo: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Accepts photo of authorized person (via 'image', 'file', or 'photo' field),
    detects face, extracts 512-D embedding, saves to database, and instantly updates
    the in-memory vector cache. Supports large mobile images.
    """
    # Accept whichever file field was supplied
    upload_file = image or file or photo
    if upload_file is None:
        raise HTTPException(
            status_code=400,
            detail="Missing image file. Please provide multipart form field 'image' or 'file'."
        )

    contents = await upload_file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image payload received.")

    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image format.")

    # Downscale very large mobile camera photos (e.g. > 1600px) to prevent memory bottlenecks
    h_orig, w_orig = img_bgr.shape[:2]
    max_dim = max(h_orig, w_orig)
    if max_dim > 1600:
        scale = 1600.0 / max_dim
        img_bgr = cv2.resize(img_bgr, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)

    # 1. Detect face
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
            raise HTTPException(
                status_code=422,
                detail="No face detected in the uploaded image. Please provide a clear, front-facing portrait."
            )
    embedding_vec = face_engine.extract_embedding(face_crop)

    # 3. Save photo to persistent storage
    file_ext = Path(upload_file.filename or "portrait.jpg").suffix or ".jpg"
    filename = f"user_{current_user.id}_{uuid.uuid4().hex[:8]}{file_ext}"
    dest_path = settings.AUTHORIZED_FACES_DIR / filename
    cv2.imwrite(str(dest_path), img_bgr)

    # 4. Save to Database
    record = AuthorizedPersonnel(
        user_id=current_user.id,
        full_name=(full_name or "Authorized Personnel").strip(),
        role_title=(role_title or "Resident / Staff").strip(),
        image_path=str(dest_path)
    )
    record.embedding = embedding_vec.tolist()
    db.add(record)
    db.commit()
    db.refresh(record)

    # 5. Instantly refresh the user's vector cache
    all_user_records = db.query(AuthorizedPersonnel).filter(AuthorizedPersonnel.user_id == current_user.id).all()
    face_engine.reload_user_embeddings(current_user.id, all_user_records)

    return AuthorizedPersonnelResponse(
        id=record.id,
        full_name=record.full_name,
        role_title=record.role_title,
        image_url=f"/storage/authorized_faces/{filename}",
        created_at=record.created_at
    )


@app.get("/api/v1/authorized/list", response_model=List[AuthorizedPersonnelResponse])
@app.get("/api/authorized/list", response_model=List[AuthorizedPersonnelResponse])
def list_authorized_personnel(
    current_user: User = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    records = db.query(AuthorizedPersonnel).filter(AuthorizedPersonnel.user_id == current_user.id).all()
    return [
        AuthorizedPersonnelResponse(
            id=r.id,
            full_name=r.full_name,
            role_title=r.role_title,
            image_url=f"/storage/authorized_faces/{Path(r.image_path).name}",
            created_at=r.created_at
        )
        for r in records
    ]


@app.delete("/api/v1/authorized/{person_id}")
def delete_authorized_person(
    person_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    record = db.query(AuthorizedPersonnel).filter(
        AuthorizedPersonnel.id == person_id,
        AuthorizedPersonnel.user_id == current_user.id
    ).first()

    if not record:
        raise HTTPException(status_code=404, detail="Personnel not found.")

    # Remove photo if exists
    if os.path.exists(record.image_path):
        try:
            os.remove(record.image_path)
        except OSError:
            pass

    db.delete(record)
    db.commit()

    # Refresh vector cache
    remaining = db.query(AuthorizedPersonnel).filter(AuthorizedPersonnel.user_id == current_user.id).all()
    face_engine.reload_user_embeddings(current_user.id, remaining)

    return {"message": "Personnel removed successfully."}


# ============================================================================
# 3. Real-Time Detection & Live Video Feed Processing
# ============================================================================

def handle_frame_inference(frame_bgr: np.ndarray, user_id: int, db: Session) -> dict:
    """Core inference handler for single video frames."""
    # Ensure cache is populated
    if user_id not in face_engine.embedding_cache:
        records = db.query(AuthorizedPersonnel).filter(AuthorizedPersonnel.user_id == user_id).all()
        face_engine.reload_user_embeddings(user_id, records)

    result = face_engine.process_frame(frame_bgr, user_id)
    snapshot_url = None
    alert_triggered = False

    # If an intruder is detected, save snapshot & trigger alert
    if result["is_intruder"] and result.get("face_crop") is not None:
        alert_triggered = True
        snapshot_filename = f"intruder_{user_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
        snapshot_path = settings.INTRUDER_ALERTS_DIR / snapshot_filename
        
        # Draw bounding box and label onto snapshot for visual verification
        annotated = frame_bgr.copy()
        if result["bbox"]:
            x, y, w, h = result["bbox"]["x"], result["bbox"]["y"], result["bbox"]["w"], result["bbox"]["h"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(annotated, "ALERT: INTRUDER", (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imwrite(str(snapshot_path), annotated)
        snapshot_url = f"/storage/intruder_snapshots/{snapshot_filename}"

        # Persist alert record
        alert = IntruderAlert(
            user_id=user_id,
            snapshot_path=str(snapshot_path),
            highest_similarity=result["similarity_score"]
        )
        db.add(alert)
        db.commit()

    return {
        "status": result["status"],
        "person_name": result["person_name"],
        "similarity_score": result["similarity_score"],
        "bbox": result["bbox"],
        "alert_triggered": alert_triggered,
        "snapshot_url": snapshot_url,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/v1/stream/detect-frame", response_model=DetectionEvent)
def detect_single_frame(
    payload: FrameAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """HTTP REST endpoint for frame-by-frame analysis."""
    try:
        header, encoded = payload.frame_base64.split(",", 1) if "," in payload.frame_base64 else ("", payload.frame_base64)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("Corrupt frame")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

    res = handle_frame_inference(frame_bgr, current_user.id, db)
    return DetectionEvent(**res)


@app.websocket("/ws/live-stream")
async def websocket_live_stream(websocket: WebSocket, token: str = Query(...)):
    """
    Real-time high-FPS bidirectional WebSocket for live mobile camera feed.
    Streams base64 JPEG frames from client -> returns bounding boxes & alert notifications.
    """
    await websocket.accept()

    # Validate JWT token from query param
    db = SessionLocal()
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        user = db.query(User).filter(User.email == email).first()
        if not user:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
    except JWTError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Warm cache
    records = db.query(AuthorizedPersonnel).filter(AuthorizedPersonnel.user_id == user.id).all()
    face_engine.reload_user_embeddings(user.id, records)

    try:
        while True:
            # Receive frame data (base64 string or JSON)
            data = await websocket.receive_text()
            if not data:
                continue

            try:
                # Strip base64 data URL prefix if present
                encoded = data.split(",", 1)[1] if "," in data else data
                img_bytes = base64.b64decode(encoded)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame_bgr is not None:
                    inference_res = handle_frame_inference(frame_bgr, user.id, db)
                    await websocket.send_json(inference_res)
            except Exception as frame_err:
                await websocket.send_json({
                    "status": "ERROR",
                    "error": str(frame_err),
                    "timestamp": datetime.utcnow().isoformat()
                })
    except WebSocketDisconnect:
        pass
    finally:
        db.close()


# ============================================================================
# 4. Intruder Alerts Management
# ============================================================================

@app.get("/api/v1/alerts/list", response_model=List[IntruderAlertResponse])
def get_intruder_alerts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    alerts = (
        db.query(IntruderAlert)
        .filter(IntruderAlert.user_id == current_user.id)
        .order_by(IntruderAlert.timestamp.desc())
        .limit(50)
        .all()
    )
    return [
        IntruderAlertResponse(
            id=a.id,
            snapshot_url=f"/storage/intruder_snapshots/{Path(a.snapshot_path).name}",
            highest_similarity=a.highest_similarity,
            timestamp=a.timestamp,
            acknowledged=a.acknowledged,
            camera_id=a.camera_id
        )
        for a in alerts
    ]


@app.patch("/api/v1/alerts/{alert_id}/acknowledge")
def acknowledge_alert(
    alert_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    alert = db.query(IntruderAlert).filter(
        IntruderAlert.id == alert_id,
        IntruderAlert.user_id == current_user.id
    ).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found.")
    alert.acknowledged = True
    db.commit()
    return {"message": "Alert acknowledged."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

