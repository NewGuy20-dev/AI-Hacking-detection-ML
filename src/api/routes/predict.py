"""Prediction API routes."""
from fastapi import APIRouter, HTTPException
import time
import os
from pathlib import Path

from src.api.schemas import PayloadRequest, URLRequest, BatchRequest, PredictResponse, BatchResponse
from src.api import server
from src.input_validator import ValidationError
from src.benign_filter import get_filter
from src.stress_test.v14.shadow_logger import ShadowLogger
from src.stress_test.v14.models import ModelWrapper
import logging

router = APIRouter(prefix="/predict", tags=["Prediction"])

SHADOW_ENABLED = os.getenv("SHADOW_EVAL_ENABLED", "0") == "1"
SHADOW_STORE_RAW = os.getenv("SHADOW_EVAL_STORE_RAW", "0") == "1"
SHADOW_LOG_PATH = os.getenv("SHADOW_EVAL_LOG_PATH", "evaluation/shadow_logs.jsonl")
shadow_logger = ShadowLogger(Path(SHADOW_LOG_PATH), store_raw=SHADOW_STORE_RAW) if SHADOW_ENABLED else None
logger = logging.getLogger(__name__)


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
    
    raw_confidence = float(result['confidence'][0])
    
    confidence = raw_confidence
    thresholds = ModelWrapper._load_thresholds()
    attack_threshold = float(thresholds.get('attack', thresholds.get('payload', 0.5)))
    is_attack = confidence >= attack_threshold
    
    latency_ms = (time.perf_counter() - start) * 1000

    if shadow_logger:
        try:
            shadow_logger.log(
                model=predictor.active_model if hasattr(predictor, 'active_model') else 'payload',
                route='payload',
                input_data=request.payload,
                prediction=int(is_attack),
                confidence=confidence,
                latency_ms=latency_ms,
                version=getattr(predictor, 'version', ''),
                error=None,
            )
        except Exception as e:
            logger.warning("shadow_logger.log failed for route='payload': %s", e)

    return PredictResponse(
        is_attack=is_attack,
        confidence=confidence,
        attack_type=_classify_attack(request.payload) if is_attack else None,
        severity=_get_severity(confidence) if is_attack else "LOW",
        processing_time_ms=latency_ms
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
    thresholds = ModelWrapper._load_thresholds()
    url_threshold = float(thresholds.get('url', 0.5))
    is_attack = confidence >= url_threshold
    
    latency_ms = (time.perf_counter() - start) * 1000

    if shadow_logger:
        try:
            shadow_logger.log(
                model=predictor.active_model if hasattr(predictor, 'active_model') else 'url',
                route='url',
                input_data=request.url,
                prediction=int(is_attack),
                confidence=confidence,
                latency_ms=latency_ms,
                version=getattr(predictor, 'version', ''),
                error=None,
            )
        except Exception as e:
            logger.warning("shadow_logger.log failed for route='url': %s", e)

    return PredictResponse(
        is_attack=is_attack,
        confidence=confidence,
        attack_type="MALICIOUS_URL" if is_attack else None,
        severity=_get_severity(confidence) if is_attack else "LOW",
        processing_time_ms=latency_ms
    )


@router.post("/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """Batch prediction for multiple inputs."""
    start = time.perf_counter()
    predictor = server.get_predictor()
    benign_filter = get_filter()
    
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    payloads = (request.payloads or [])[:100]
    urls = (request.urls or [])[:100]
    if not payloads and not urls:
        raise HTTPException(status_code=422, detail="At least one of payloads or urls is required")

    total = len(payloads) + len(urls)
    results = [None] * total

    # Prefilter payloads while preserving original payload positions.
    ml_payloads = []
    ml_payload_indices = []
    for i, payload in enumerate(payloads):
        is_benign, benign_confidence, _ = benign_filter.is_benign(payload)
        if is_benign:
            results[i] = PredictResponse(
                is_attack=False,
                confidence=1.0 - benign_confidence,
                attack_type=None,
                severity="LOW",
                processing_time_ms=0
            )
        else:
            ml_payloads.append(payload)
            ml_payload_indices.append(i)

    if ml_payloads:
        try:
            payload_result = predictor.predict_batch({'payloads': ml_payloads})
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))

        payload_confidences = payload_result.get('confidence', [])
        safe_count = min(len(ml_payloads), len(payload_confidences))
        for i in range(safe_count):
            payload = ml_payloads[i]
            idx = ml_payload_indices[i]
            confidence = float(payload_confidences[i])
            thresholds = ModelWrapper._load_thresholds()
            attack_threshold = float(thresholds.get('attack', thresholds.get('payload', 0.5)))
            is_attack = confidence >= attack_threshold
            results[idx] = PredictResponse(
                is_attack=is_attack,
                confidence=confidence,
                attack_type=_classify_attack(payload) if is_attack else None,
                severity=_get_severity(confidence) if is_attack else "LOW",
                processing_time_ms=0
            )

    if urls:
        try:
            url_result = predictor.predict_batch({'urls': urls})
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))

        base_idx = len(payloads)
        for i, raw_confidence in enumerate(url_result['confidence']):
            confidence = float(raw_confidence)
            thresholds = ModelWrapper._load_thresholds()
            url_threshold = float(thresholds.get('url', 0.5))
            is_attack = confidence >= url_threshold
            results[base_idx + i] = PredictResponse(
                is_attack=is_attack,
                confidence=confidence,
                attack_type="MALICIOUS_URL" if is_attack else None,
                severity=_get_severity(confidence) if is_attack else "LOW",
                processing_time_ms=0
            )

    # Defensive fallback: never return missing slots.
    for i, item in enumerate(results):
        if item is None:
            results[i] = PredictResponse(
                is_attack=False,
                confidence=0.5,
                attack_type=None,
                severity="LOW",
                processing_time_ms=0
            )
    
    total_latency_ms = (time.perf_counter() - start) * 1000

    if shadow_logger:
        try:
            # log aggregate info only to avoid large payloads
            shadow_logger.log(
                model=predictor.active_model if hasattr(predictor, 'active_model') else 'batch',
                route='batch',
                input_data='batch',
                prediction=0,
                confidence=0.0,
                latency_ms=total_latency_ms,
                version=getattr(predictor, 'version', ''),
                error=None,
            )
        except Exception as e:
            logger.warning("shadow_logger.log failed for route='batch': %s", e)

    return BatchResponse(
        results=results,
        total_processing_time_ms=total_latency_ms
    )
