import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Redis Message Broker
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", 15.0))
REDIS_CONNECT_TIMEOUT = float(os.getenv("REDIS_CONNECT_TIMEOUT", 5.0))
REDIS_HEALTH_CHECK_INTERVAL = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", 30))

# Queues & Channels (per SRS FR-1.1, FR-1.2, FR-3.1)
VIDEO_QUEUE = "video_stream"
NETWORK_QUEUE = "network_stream"
ALERT_CHANNEL = "threat_alerts"

# Database Configuration (per SRS 1 & 3.2)
DATABASE_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "events.db"))
DATABASE_URI = f"sqlite:///{DATABASE_PATH}"

# Model Artifact Paths (per SRS 3.1)
ARTIFACTS_DIR = BASE_DIR / "ml_engine" / "artifacts"
CNN_MODEL_PATH = str(ARTIFACTS_DIR / "spatial_cnn.h5")
KNN_MODEL_PATH = str(ARTIFACTS_DIR / "network_knn.pkl")
ANN_MODEL_PATH = str(ARTIFACTS_DIR / "fusion_ann.h5")

# Inference & Correlation Settings (per SRS FR-2.4, FR-3.2, NFR-1)
TEMPORAL_WINDOW_SECONDS = float(os.getenv("TEMPORAL_WINDOW_SECONDS", 2.0))
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", 0.7))
MAX_LATENCY_MS = 500.0

# API Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
