from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, EmailStr

# Auth Schemas
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_name: str
    user_email: str

class TokenData(BaseModel):
    email: Optional[str] = None

# Authorized Personnel Schemas
class AuthorizedPersonnelResponse(BaseModel):
    id: int
    full_name: str
    role_title: str
    image_url: str
    created_at: datetime

    class Config:
        from_attributes = True

# Intruder Alert Schemas
class IntruderAlertResponse(BaseModel):
    id: int
    snapshot_url: str
    highest_similarity: float
    timestamp: datetime
    acknowledged: bool
    camera_id: str

    class Config:
        from_attributes = True

# Real-time Video Detection Result
class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

class DetectionEvent(BaseModel):
    status: str  # "AUTHORIZED", "INTRUDER", "NO_FACE_DETECTED"
    person_name: Optional[str] = None
    similarity_score: float
    bbox: Optional[BoundingBox] = None
    alert_triggered: bool = False
    snapshot_url: Optional[str] = None
    timestamp: str

class FrameAnalysisRequest(BaseModel):
    frame_base64: str
    camera_id: Optional[str] = "mobile_native_cam"
