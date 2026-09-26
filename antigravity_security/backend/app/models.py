from datetime import datetime
import json
from typing import List
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    authorized_faces = relationship("AuthorizedPersonnel", back_populates="owner", cascade="all, delete-orphan")
    intruder_alerts = relationship("IntruderAlert", back_populates="owner", cascade="all, delete-orphan")


class AuthorizedPersonnel(Base):
    __tablename__ = "authorized_personnel"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    full_name = Column(String(255), nullable=False)
    role_title = Column(String(100), default="Resident / Staff")
    image_path = Column(String(500), nullable=False)
    # Stored as JSON string of 512 floats (compatible with SQLite & PostgreSQL)
    embedding_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="authorized_faces")

    @property
    def embedding(self) -> List[float]:
        return json.loads(self.embedding_json)

    @embedding.setter
    def embedding(self, values: List[float]):
        self.embedding_json = json.dumps(values)


class IntruderAlert(Base):
    __tablename__ = "intruder_alerts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    snapshot_path = Column(String(500), nullable=False)
    highest_similarity = Column(Float, default=0.0)
    acknowledged = Column(Boolean, default=False)
    camera_id = Column(String(100), default="mobile_camera_front")
    timestamp = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="intruder_alerts")
