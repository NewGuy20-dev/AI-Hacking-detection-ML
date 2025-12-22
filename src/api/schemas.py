"""Pydantic schemas for API requests and responses."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class PayloadRequest(BaseModel):
    payload: str = Field(..., max_length=10000, description="Text payload to analyze")
    include_explanation: bool = Field(False, description="Include feature explanation")


class URLRequest(BaseModel):
    url: str = Field(..., max_length=2000, description="URL to analyze")
    include_explanation: bool = Field(False, description="Include feature explanation")


class BatchRequest(BaseModel):
    payloads: Optional[List[str]] = Field(None, max_length=100)
    urls: Optional[List[str]] = Field(None, max_length=100)


class PredictResponse(BaseModel):
    is_attack: bool
    confidence: float = Field(..., ge=0, le=1)
    attack_type: Optional[str] = None
    severity: str
    explanation: Optional[Dict[str, Any]] = None
    processing_time_ms: float


class BatchResponse(BaseModel):
    results: List[PredictResponse]
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float


class ReadinessResponse(BaseModel):
    status: str
    models_loaded: Dict[str, List[str]]
    uptime_seconds: float
