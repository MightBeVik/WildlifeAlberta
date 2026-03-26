"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    category: str  # "animal", "person", "vehicle"


class Detection(BaseModel):
    image_id: str
    filename: str
    detections: List[BoundingBox]
    has_animal: bool
    processing_time_ms: float


class Classification(BaseModel):
    species: str
    confidence: float
    top_5: List[dict]  # [{"species": "...", "confidence": 0.xx}, ...]


class DetectionWithClassification(BaseModel):
    image_id: str
    filename: str
    image_url: str
    detections: List[BoundingBox]
    classifications: List[Classification]
    has_animal: bool
    processing_time_ms: float
    timestamp: datetime


class UploadResponse(BaseModel):
    image_id: str
    filename: str
    filepath: str
    size_bytes: int
    width: int
    height: int
    message: str


class BatchUploadResponse(BaseModel):
    uploaded: List[UploadResponse]
    failed: List[dict]
    total: int


class ImageInfo(BaseModel):
    image_id: str
    filename: str
    filepath: str
    size_bytes: int
    width: int
    height: int
    upload_time: str


class StatsResponse(BaseModel):
    total_images: int
    total_detections: int
    species_counts: dict
    recent_uploads: List[dict]
    model_info: dict


class HealthResponse(BaseModel):
    status: str
    detector_loaded: bool
    classifier_loaded: bool
    version: str
