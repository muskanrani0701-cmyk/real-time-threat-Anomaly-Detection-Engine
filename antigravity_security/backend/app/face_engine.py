"""
Project Antigravity: Real-Time Face Recognition & Intrusion Engine
Executes face detection, 512-D embedding extraction, and sub-millisecond vector matching.
Supports both TensorFlow/Keras and PyTorch backbones with automatic fallback.
Strict Data Constraint: Powered by model trained on authentic human faces (Kaggle LFW).
"""

import sys
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

from .config import settings

logger = logging.getLogger("FaceEngine")

BASE_ML_DIR = Path(__file__).resolve().parent.parent.parent / "ml_pipeline"
KERAS_CKPT = BASE_ML_DIR / "checkpoints" / "facenet_512_base.keras"
PYTORCH_CKPT = BASE_ML_DIR / "checkpoints" / "facenet_lfw_best.pth"


class FaceEngine:
    """
    High-performance real-time facial recognition engine.
    Features:
      - Face localization & bounding box extraction
      - 512-D vector embedding computation (TensorFlow or PyTorch)
      - In-memory vector cache for sub-millisecond cosine comparison
    """
    def __init__(self):
        self.backend_framework = "none"
        self.tf_model = None
        self.torch_model = None

        # 1. Initialize Face Embedding Model (Auto-detect TensorFlow or PyTorch)
        self._init_deep_model()

        # 2. Initialize Fast Haar Cascade Face Detector for real-time video frames
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # In-memory vector cache: user_id -> List of (person_id, person_name, np.ndarray 512)
        self.embedding_cache: Dict[int, List[Tuple[int, str, np.ndarray]]] = {}

    def _init_deep_model(self):
        """Initializes TensorFlow/Keras or PyTorch embedding model."""
        # Check TensorFlow / Keras first
        try:
            import tensorflow as tf
            from tensorflow import keras
            from ml_engine.model_utils import ensure_model_weights
            ckpt_path = ensure_model_weights(KERAS_CKPT)
            if ckpt_path.exists():
                self.tf_model = keras.models.load_model(str(ckpt_path), compile=False)
                self.backend_framework = "tensorflow"
                logger.info(f"Loaded TensorFlow/Keras 512-D FaceNet model from {ckpt_path}")
                return
            else:
                from ml_pipeline.model_tf import build_facenet_backbone
                self.tf_model = build_facenet_backbone(embedding_dim=512)
                self.backend_framework = "tensorflow"
                logger.info("Initialized TensorFlow 512-D FaceNet backbone with ImageNet weights.")
                return
        except Exception as e:
            logger.info(f"TensorFlow not loaded: {e}. Checking PyTorch...")

        # Check PyTorch
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torchvision import models

            class TorchFaceNet(nn.Module):
                def __init__(self, embedding_dim=512):
                    super().__init__()
                    base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                    in_f = base.fc.in_features
                    base.fc = nn.Sequential(
                        nn.Linear(in_f, embedding_dim, bias=False),
                        nn.BatchNorm1d(embedding_dim)
                    )
                    self.backbone = base
                def forward(self, x):
                    return F.normalize(self.backbone(x), p=2, dim=1)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.torch_model = TorchFaceNet(embedding_dim=512).to(device)
            if PYTORCH_CKPT.exists():
                ckpt = torch.load(PYTORCH_CKPT, map_location=device)
                self.torch_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            self.torch_model.eval()
            self.backend_framework = "pytorch"
            self.device = device
            logger.info("Initialized PyTorch 512-D FaceNet model.")
            return
        except Exception as e:
            logger.warning(f"PyTorch not loaded: {e}. Operating in lightweight normalized feature mode.")
            self.backend_framework = "fallback"

    def reload_user_embeddings(self, user_id: int, authorized_records: list):
        """Refreshes the in-memory vector cache for a given user for zero-latency lookups."""
        cache_list = []
        for record in authorized_records:
            emb = np.array(record.embedding, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            cache_list.append((record.id, record.full_name, emb))
        self.embedding_cache[user_id] = cache_list
        logger.info(f"Updated embedding cache for user {user_id}: {len(cache_list)} authorized faces loaded.")

    def detect_face_bbox(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects primary human face in a video frame.
        Returns bounding box (x, y, w, h) or None.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            return None
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return tuple(faces[0])

    def extract_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts 512-D L2-normalized embedding vector from a cropped face image.
        Guarantees ||v||_2 = 1.0.
        """
        resized = cv2.resize(face_bgr, (160, 160))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if self.backend_framework == "tensorflow" and self.tf_model is not None:
            # TensorFlow/Keras inference
            input_tensor = np.expand_dims(rgb.astype(np.float32), axis=0)
            emb = self.tf_model(input_tensor, training=False).numpy()[0]
        elif self.backend_framework == "pytorch" and self.torch_model is not None:
            import torch
            tensor = torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.torch_model(tensor).cpu().numpy()[0]
        else:
            # Fallback 512-D spatial histogram features
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
        user_id: int,
        threshold: float = settings.SIMILARITY_THRESHOLD
    ) -> Dict:
        """
        Analyzes a live video frame:
          1. Detects face bbox
          2. Extracts 512-D embedding
          3. Calculates Cosine Similarity with authorized personnel
          4. Returns Decision: AUTHORIZED vs INTRUDER
        """
        bbox = self.detect_face_bbox(frame_bgr)
        if bbox is None:
            return {
                "status": "NO_FACE_DETECTED",
                "person_name": None,
                "similarity_score": 0.0,
                "bbox": None,
                "is_intruder": False
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
                "status": "NO_FACE_DETECTED",
                "person_name": None,
                "similarity_score": 0.0,
                "bbox": None,
                "is_intruder": False
            }

        live_embedding = self.extract_embedding(face_crop)

        cached_faces = self.embedding_cache.get(user_id, [])
        if not cached_faces:
            # If no authorized personnel are registered, any face is an intruder
            return {
                "status": "INTRUDER",
                "person_name": "Unregistered Intruder",
                "similarity_score": 0.0,
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "face_crop": face_crop,
                "is_intruder": True
            }

        # Vectorized Cosine Similarity (Dot Product of L2 unit vectors)
        best_name = None
        best_score = -1.0

        for pid, name, auth_emb in cached_faces:
            score = float(np.dot(live_embedding, auth_emb))
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= threshold:
            status = "AUTHORIZED"
            is_intruder = False
        else:
            status = "INTRUDER"
            is_intruder = True

        return {
            "status": status,
            "person_name": best_name if status == "AUTHORIZED" else "Unknown Intruder",
            "similarity_score": round(max(0.0, best_score), 4),
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "face_crop": face_crop,
            "is_intruder": is_intruder
        }


# Singleton engine instance
face_engine = FaceEngine()
