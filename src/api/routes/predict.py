"""Prediction API routes."""
from fastapi import APIRouter, HTTPException
import time

from src.api.schemas import PayloadRequest, URLRequest, BatchRequest, PredictResponse, BatchResponse
from src.api import server
from src.input_validator import ValidationError
from src.benign_filter import get_filter

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Confidence threshold for attack classification
ATTACK_THRESHOLD = 0.75  # Lowered from 0.85 to catch more attacks


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
    
    # Check benign pre-filter first
    benign_filter = get_filter()
    is_benign, benign_confidence, reason = benign_filter.is_benign(request.payload)
    
    if is_benign:
        return PredictResponse(
            is_attack=False,
            confidence=1.0 - benign_confidence,  # Low attack confidence
            attack_type=None,
            severity="LOW",
            processing_time_ms=(time.perf_counter() - start) * 1000
        )
    
    predictor = server.get_predictor()
    
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        result = predictor.predict_batch({'payloads': [request.payload]})
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    raw_confidence = float(result['confidence'][0])
    
    # Scale confidence based on input length
    scale = benign_filter.get_confidence_scale(request.payload)
    confidence = raw_confidence * scale
    
    # Apply threshold
    is_attack = confidence >= ATTACK_THRESHOLD
    
    return PredictResponse(
        is_attack=is_attack,
        confidence=confidence,
        attack_type=_classify_attack(request.payload) if is_attack else None,
        severity=_get_severity(confidence) if is_attack else "LOW",
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
    is_attack = confidence >= 0.80  # URL threshold
    
    return PredictResponse(
        is_attack=is_attack,
        confidence=confidence,
        attack_type="MALICIOUS_URL" if is_attack else None,
        severity=_get_severity(confidence) if is_attack else "LOW",
        processing_time_ms=(time.perf_counter() - start) * 1000
    )


@router.post("/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """Batch prediction for multiple inputs."""
    start = time.perf_counter()
    predictor = server.get_predictor()
    benign_filter = get_filter()
    
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    data = {}
    benign_indices = []
    
    # Pre-filter payloads
    if request.payloads:
        filtered_payloads = []
        for i, p in enumerate(request.payloads[:100]):
            is_benign, _, _ = benign_filter.is_benign(p)
            if is_benign:
                benign_indices.append(i)
            else:
                filtered_payloads.append(p)
        if filtered_payloads:
            data['payloads'] = filtered_payloads
    
    if request.urls:
        data['urls'] = request.urls[:100]
    
    results = []
    
    # Add benign results
    for i in benign_indices:
        results.append(PredictResponse(
            is_attack=False,
            confidence=0.05,
            severity="LOW",
            processing_time_ms=0
        ))
    
    # Process remaining through ML
    if data:
        try:
            result = predictor.predict_batch(data)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))
        
        for i in range(len(result['is_attack'])):
            confidence = float(result['confidence'][i])
            is_attack = confidence >= ATTACK_THRESHOLD
            results.append(PredictResponse(
                is_attack=is_attack,
                confidence=confidence,
                severity=_get_severity(confidence) if is_attack else "LOW",
                processing_time_ms=0
            ))
    
    return BatchResponse(
        results=results,
        total_processing_time_ms=(time.perf_counter() - start) * 1000
    )
