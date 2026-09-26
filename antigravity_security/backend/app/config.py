import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "Project Antigravity Security Backend"
    VERSION: str = "1.0.0"
    
    # Security & JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY", "antigravity_super_secret_jwt_key_2026_x99_sec")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/antigravity_security.db")
    
    # ML Face Recognition Thresholds
    # Cosine Similarity: 1.0 (exact identical face) to -1.0 (opposite).
    # Typical threshold on L2-normalized embeddings: 0.65 - 0.70
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.68"))
    
    # Storage Paths
    STORAGE_DIR: Path = BASE_DIR / "storage"
    AUTHORIZED_FACES_DIR: Path = STORAGE_DIR / "authorized_faces"
    INTRUDER_ALERTS_DIR: Path = STORAGE_DIR / "intruder_snapshots"
    
    # Model Weights
    MODEL_CHECKPOINT: Path = BASE_DIR.parent / "ml_pipeline" / "checkpoints" / "facenet_lfw_best.pth"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Ensure storage directories exist
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.AUTHORIZED_FACES_DIR.mkdir(parents=True, exist_ok=True)
settings.INTRUDER_ALERTS_DIR.mkdir(parents=True, exist_ok=True)
