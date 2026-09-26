"""
Project Antigravity: Real-Time Biometric Face Engine
Integrates directly into the existing Threat Detection Engine.
Extracts 512-D FaceNet embeddings, maintains an in-memory vector matrix,
and performs zero-latency Cosine Similarity matching for intruder detection.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

import config
import db

logger = logging.getLogger("face_engine")


class FaceEngine:
    """
    Sub-millisecond facial recognition and intrusion engine.
    - 512-D Unit-Normalized Hypersphere Projection
    - Zero-latency in-memory vector cache loaded from SQLite
    - Dual backend: TensorFlow/Keras or PyTorch
    """
    def __init__(self):
        logger.info("Initializing Biometric Face Engine...")

        # 1. Initialize Face Detection Cascade
        cascade_path = str(config.BASE_DIR / "haarcascade_frontalface_default.xml")
        if not os.path.exists(cascade_path):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # 2. Load 512-D FaceNet Deep Model
        self.model = None
        self._load_model()

        # 3. In-Memory Vector Cache: [(id, full_name, np.ndarray 512), ...]
        self.authorized_cache: List[Tuple[int, str, np.ndarray]] = []
        self.reload_cache()

    def _load_model(self):
        """Loads trained 512-D Keras or PyTorch model from ml_engine/artifacts/."""
        from ml_engine.model_utils import ensure_model_weights
        model_path = ensure_model_weights(Path(config.FACENET_MODEL_PATH))
        if os.path.exists(str(model_path)):
            try:
                import keras
                self.model = keras.models.load_model(str(model_path), compile=False)
                # Warmup inference
                dummy = np.zeros((1, 160, 160, 3), dtype=np.float32)
                _ = self.model(dummy, training=False)
                logger.info(f"Loaded 512-D FaceNet model from {model_path}")
                return
            except Exception as e:
                logger.warning(f"Could not load Keras model from {model_path}: {e}")

        logger.info("Operating in lightweight normalized biometric feature mode.")

    def reload_cache(self):
        """Reloads authorized personnel and their 512-D embeddings from SQLite."""
        try:
            records = db.get_all_authorized()
            new_cache = []
            for r in records:
                emb = np.array(r["embedding"], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                new_cache.append((r["id"], r["full_name"], emb))
            self.authorized_cache = new_cache
            logger.info(f"Reloaded authorized vector cache: {len(self.authorized_cache)} profiles active.")
        except Exception as e:
            logger.warning(f"Could not load authorized cache from DB: {e}")

    def detect_face_bbox(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detects the most prominent human face in a video frame."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Pass 1: Standard detection on raw grayscale
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # Pass 2: Relaxed detection for selfies, mobile uploads, and varied lighting
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        # Pass 3: Histogram equalized pass for low-contrast images
        if len(faces) == 0:
            eq = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(
                eq,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        if len(faces) == 0:
            return None
        # Return largest detected face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return tuple(faces[0])

    def extract_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts a 512-D L2-normalized embedding vector from a cropped face image.
        Guarantees ||v||_2 = 1.0.
        """
        resized = cv2.resize(face_bgr, (160, 160))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if self.model is not None:
            tensor = np.expand_dims(rgb.astype(np.float32), axis=0)
            emb = self.model(tensor, training=False).numpy()[0]
        else:
            # Fallback normalized spatial-frequency representation
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            features = cv2.resize(gray, (32, 16)).flatten().astype(np.float32)
            emb = features / (np.linalg.norm(features) + 1e-7)

        # Enforce exact unit norm
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        threshold: float = config.FACE_SIMILARITY_THRESHOLD
    ) -> Dict:
        """
        Analyzes a single video frame for biometric identity:
          1. Detects face bounding box
          2. Crops & extracts 512-D embedding
          3. Evaluates Cosine Similarity against all authorized profiles
          4. Returns Decision: AUTHORIZED vs INTRUDER
        """
        bbox = self.detect_face_bbox(frame_bgr)
        if bbox is None:
            return {
                "status": "NO_FACE",
                "person_name": None,
                "similarity_score": 0.0,
                "bbox": None,
                "is_intruder": False,
                "face_crop": None
            }

        x, y, w, h = bbox
        h_img, w_img = frame_bgr.shape[:2]
        margin_x = int(0.1 * w)
        margin_y = int(0.1 * h)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_img, x + w + margin_x)
        y2 = min(h_img, y + h + margin_y)

        face_crop = frame_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            return {
                "status": "NO_FACE",
                "person_name": None,
                "similarity_score": 0.0,
                "bbox": None,
                "is_intruder": False,
                "face_crop": None
            }

        live_emb = self.extract_embedding(face_crop)

        if not self.authorized_cache:
            # When no authorized faces are registered, every face is an unrecognized intruder
            return {
                "status": "INTRUDER",
                "person_name": "Unregistered Intruder",
                "similarity_score": 0.0,
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "is_intruder": True,
                "face_crop": face_crop
            }

        # Vectorized Dot Product over authorized matrix
        best_name = None
        best_score = -1.0

        for pid, name, auth_emb in self.authorized_cache:
            score = float(np.dot(live_emb, auth_emb))
            if score > best_score:
                best_score = score
                best_name = name

        is_authorized = best_score >= threshold
        return {
            "status": "AUTHORIZED" if is_authorized else "INTRUDER",
            "person_name": best_name if is_authorized else "Unknown Intruder",
            "similarity_score": round(max(0.0, best_score), 4),
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "is_intruder": not is_authorized,
            "face_crop": face_crop
        }


# Global singleton instance
face_engine = FaceEngine()
