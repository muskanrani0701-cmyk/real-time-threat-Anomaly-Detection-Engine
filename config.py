import os
import shutil
import logging
from pathlib import Path

logger = logging.getLogger("config")

BASE_DIR = Path(__file__).resolve().parent

# Detection of Serverless Environments (Vercel, AWS Lambda)
IS_VERCEL = bool(
    os.getenv("VERCEL")
    or os.getenv("VERCEL_ENV")
    or os.getenv("AWS_LAMBDA_FUNCTION_NAME")
)

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
# On Vercel, the root deployment directory (/var/task) is strictly read-only.
# SQLite writes (including WAL/journaling) must use /tmp/events.db.
if IS_VERCEL:
    DEFAULT_DB_PATH = "/tmp/events.db"
    # Seed writable SQLite DB from repo bundled db if it exists
    try:
        bundled_db = BASE_DIR / "events.db"
        target_db = Path(DEFAULT_DB_PATH)
        if bundled_db.exists() and not target_db.exists():
            target_db.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundled_db, target_db)
    except Exception as e:
        logger.warning(f"Could not copy bundled DB to /tmp: {e}")
else:
    DEFAULT_DB_PATH = str(BASE_DIR / "events.db")

DATABASE_PATH = os.getenv("DATABASE_PATH", DEFAULT_DB_PATH)
DATABASE_URI = f"sqlite:///{DATABASE_PATH}"

# Model Artifact Paths (per SRS 3.1)
ARTIFACTS_DIR = BASE_DIR / "ml_engine" / "artifacts"
CNN_MODEL_PATH = str(ARTIFACTS_DIR / "spatial_cnn.h5")
KNN_MODEL_PATH = str(ARTIFACTS_DIR / "network_knn.pkl")
ANN_MODEL_PATH = str(ARTIFACTS_DIR / "fusion_ann.h5")
FACENET_MODEL_PATH = str(ARTIFACTS_DIR / "facenet_512_base.keras")

# Facial Recognition & Storage Settings
FACE_SIMILARITY_THRESHOLD = float(os.getenv("FACE_SIMILARITY_THRESHOLD", 0.68))

# Dynamic Storage Routing:
# On Vercel / AWS Lambda, route all write paths to /tmp/storage/ instead of read-only root
if IS_VERCEL:
    STORAGE_DIR = Path("/tmp/storage")
else:
    STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "storage")))

AUTHORIZED_FACES_DIR = STORAGE_DIR / "authorized_faces"
INTRUDER_SNAPSHOTS_DIR = STORAGE_DIR / "intruder_snapshots"

# Ensure storage directories exist safely without crashing on read-only filesystems
for dir_path in [AUTHORIZED_FACES_DIR, INTRUDER_SNAPSHOTS_DIR]:
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as err:
        logger.warning(f"Could not create storage directory at {dir_path} ({err}). Falling back to /tmp/storage.")
        STORAGE_DIR = Path("/tmp/storage")
        AUTHORIZED_FACES_DIR = STORAGE_DIR / "authorized_faces"
        INTRUDER_SNAPSHOTS_DIR = STORAGE_DIR / "intruder_snapshots"
        try:
            AUTHORIZED_FACES_DIR.mkdir(parents=True, exist_ok=True)
            INTRUDER_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        break

# Inference & Correlation Settings (per SRS FR-2.4, FR-3.2, NFR-1)
TEMPORAL_WINDOW_SECONDS = float(os.getenv("TEMPORAL_WINDOW_SECONDS", 2.0))
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", 0.7))
MAX_LATENCY_MS = 500.0

# API Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
