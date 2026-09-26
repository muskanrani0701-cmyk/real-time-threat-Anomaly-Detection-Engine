"""
AEGIS Biometric Live Camera Worker
Runs real-time OpenCV desktop surveillance with 512-D FaceNet face recognition,
differentiating between authorized users and unauthorized intruders.
"""

import cv2
import requests
import time
import os
import uuid
from datetime import datetime, timezone
import numpy as np

import config
import db
from ml_engine.face_engine import face_engine

print("=" * 60)
print("AEGIS BIOMETRIC LIVE SURVEILLANCE WORKER")
print(f"Loaded Authorized Profiles: {len(face_engine.authorized_cache)}")
print("Press 'q' in the video window to stop.")
print("Press 'r' to refresh authorized personnel cache.")
print("=" * 60)

# Use CAP_DSHOW on Windows to bypass MSMF device lock bugs
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

last_alert_time = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            # Standby mode if camera isn't attached
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "STANDBY - WAITING FOR CAMERA DEVICE", (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.imshow("AEGIS Live Biometric Analysis", frame)
            key = cv2.waitKey(200) & 0xFF
            if key == ord('q'):
                break
            continue

        frame = cv2.resize(frame, (640, 480))

        # Process frame through 512-D FaceEngine
        result = face_engine.process_frame(frame)
        bbox = result["bbox"]
        status = result["status"]
        sim = result["similarity_score"]

        # Render Top HUD
        cv2.rectangle(frame, (0, 0), (640, 40), (15, 23, 42), -1)
        cv2.putText(frame, f"AEGIS BIOMETRIC RADAR // PROFILES: {len(face_engine.authorized_cache)}",
                    (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (34, 211, 238), 2)

        if bbox is not None:
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

            if status == "AUTHORIZED":
                # Green box for approved personnel
                cv2.rectangle(frame, (x, y), (x + w, y + h), (16, 185, 129), 2)
                label = f"AUTHORIZED: {result['person_name']} ({int(sim * 100)}%)"
                cv2.putText(frame, label, (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (16, 185, 129), 2)
            elif status == "INTRUDER":
                # Red box for intruders
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 239), 3)
                label = f"ALERT: INTRUDER ({int(sim * 100)}%)"
                cv2.putText(frame, label, (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 239), 2)

                now = time.time()
                if now - last_alert_time > 4.0:
                    last_alert_time = now
                    snap_name = f"intruder_{int(now)}_{uuid.uuid4().hex[:6]}.jpg"
                    snap_path = config.INTRUDER_SNAPSHOTS_DIR / snap_name
                    cv2.imwrite(str(snap_path), frame)

                    # Dispatch threat event to API
                    try:
                        requests.post("http://127.0.0.1:5000/api/v1/events", json={
                            "source_id": "laptop_cam_1",
                            "risk_score": 0.98,
                            "severity": "CRITICAL",
                            "face_status": "INTRUDER",
                            "person_name": "Unidentified Intruder",
                            "face_similarity": sim,
                            "snapshot_path": str(snap_path)
                        }, timeout=0.5)
                        print(f"[*] INTRUDER ALERT DISPATCHED: Saved snapshot {snap_name}")
                    except Exception as err:
                        pass

        cv2.imshow("AEGIS Live Biometric Analysis", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            face_engine.reload_cache()
            print(f"[*] Cache reloaded: {len(face_engine.authorized_cache)} profiles.")

finally:
    cap.release()
    cv2.destroyAllWindows()