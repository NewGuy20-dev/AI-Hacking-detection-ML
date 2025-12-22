"""Prediction API routes."""
from fastapi import APIRouter, HTTPException
import time

from src.api.schemas import PayloadRequest, URLRequest, BatchRequest, PredictResponse, BatchResponse
from src.api import server
from src.input_validator import ValidationError

router = APIRouter(prefix="/predict", tags=["Prediction"])


def _get_severity(confidence: float) -> str:
    if confidence > 0.95: return "CRITICAL"
    if confidence > 0.85: return "HIGH"
    if confidence > 0.7: return "MEDIUM"
    return "LOW"


def _classify_attack(text: str) -> str:
    text_lower = text.lower()
    if any(p in text_lower for p in ["'", "union", "select", "--"]): return "SQL_INJECTION"
    if any(p in text_lower for p in ["<script", "onerror", "javascript:"]): return "XSS"
    if any(p in text_lower for p in [";", "|", "`", "$("]): return "COMMAND_INJECTION"
    return "UNKNOWN"


@router.post("/payload", response_model=PredictResponse)
async def predict_payload(request: PayloadRequest):
    """Analyze payload for attacks."""
    start = time.perf_counter()
    predictor = server.get_predictor()
    
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        result = predictor.predict_batch({'payloads': [request.payload]})
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    confidence = float(result['confidence'][0])
    is_attack = bool(result['is_attack'][0])
    
    return PredictResponse(
        is_attack=is_attack,
        confidence=confidence,
        attack_type=_classify_attack(request.payload) if is_attack else None,
        severity=_get_severity(confidence),
        processing_time_ms=(time.perf_counter() - start) * 1000
    )


@router.post("/url", response_model=PredictResponse)
async def predict_url(request: URLRequest):
    """Analyze URL for maliciousness."""
    start = time.perf_counter()
    predictor = server.get_predictor()
    
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        result = predictor.predict_batch({'urls': [request.url]})
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    confidence = float(result['confidence'][0])
    
    return PredictResponse(
        is_attack=bool(result['is_attack'][0]),
        confidence=confidence,
        attack_type="MALICIOUS_URL" if confidence > 0.5 else None,
        severity=_get_severity(confidence),
        processing_time_ms=(time.perf_counter() - start) * 1000
    )


@router.post("/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """Batch prediction for multiple inputs."""
    start = time.perf_counter()
    predictor = server.get_predictor()
    
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    data = {}
    if request.payloads:
        data['payloads'] = request.payloads[:100]
    if request.urls:
        data['urls'] = request.urls[:100]
    
    if not data:
        raise HTTPException(status_code=422, detail="No inputs provided")
    
    try:
        result = predictor.predict_batch(data)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    results = [
        PredictResponse(
            is_attack=bool(result['is_attack'][i]),
            confidence=float(result['confidence'][i]),
            severity=_get_severity(float(result['confidence'][i])),
            processing_time_ms=0
        )
        for i in range(len(result['is_attack']))
    ]
    
    return BatchResponse(
        results=results,
        total_processing_time_ms=(time.perf_counter() - start) * 1000
    )
