import time
from typing import Optional
from pydantic import BaseModel, Field

class NetworkLogPayload(BaseModel):
    """Network telemetry schema matching SRS Section 2.1 FR-1.2."""
    source_id: str = Field(..., description="Unique host/sensor identifier")
    src_ip: str = Field("192.168.1.100", description="Source IP address")
    dst_ip: str = Field("10.0.0.1", description="Destination IP address")
    src_port: int = Field(..., ge=0, le=65535, description="Source port")
    dst_port: int = Field(..., ge=0, le=65535, description="Destination port")
    protocol: str = Field("TCP", description="Protocol (e.g. TCP, UDP, ICMP)")
    packet_size: int = Field(..., ge=0, description="Packet size in bytes")
    packet_rate: float = Field(..., ge=0.0, description="Packets per second")
    flags: Optional[str] = Field(None, description="TCP flags (e.g. SYN, ACK, FIN)")
    timestamp: Optional[float] = Field(default_factory=time.time, description="Unix timestamp")

class VideoFrameHeader(BaseModel):
    """Metadata schema for incoming video frames matching SRS FR-1.1."""
    source_id: str = Field(..., description="Camera / stream identifier")
    frame_index: Optional[int] = Field(0, description="Sequential frame counter")
    timestamp: Optional[float] = Field(default_factory=time.time, description="Capture timestamp")
    width: Optional[int] = Field(None, description="Frame width")
    height: Optional[int] = Field(None, description="Frame height")
