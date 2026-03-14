from pydantic import BaseModel  # pyre-ignore
from typing import List, Optional, Dict
from datetime import datetime

class DetectionModel(BaseModel):
    label: str
    confidence: float
    box: List[int]
    is_waste: bool
    location: str
    message: Optional[str] = None
    bin_color: Optional[str] = None
    color_hex: Optional[str] = None
    tip: Optional[str] = None
    box_color: str
    box_color_hex: str
    raw_label: Optional[str] = None
    interaction_type: Optional[str] = None

class RealtimePredictRequest(BaseModel):
    frame_base64: str

class RealtimePredictResponse(BaseModel):
    detections: List[DetectionModel]
    summary: str
    scene_state: str
    saved: Optional[bool] = False
    saved_id: Optional[str] = None
    auto_saved: Optional[bool] = False

class ErrorResponse(BaseModel):
    detail: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_connected: bool

class ScanHistory(BaseModel):
    id: str
    timestamp: datetime
    predicted_class: str
    confidence: float
    image_thumbnail_url: Optional[str] = None
    interaction_type: str
    robot_message: Optional[str] = None

    class Config:
        from_attributes = True

class StatsResponse(BaseModel):
    total_scans: int
    model_accuracy: float
    average_confidence: float
    class_distribution: Dict[str, int]
    interaction_stats: Dict[str, int]
    waste_by_class: Dict[str, int]
    total_waste_scans: int
