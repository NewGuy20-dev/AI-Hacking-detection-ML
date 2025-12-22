"""Health check routes."""
from fastapi import APIRouter

from src.api.schemas import HealthResponse, ReadinessResponse
from src.api import server

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness check."""
    return HealthResponse(status="healthy", uptime_seconds=server.get_uptime())


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check():
    """Readiness check - models loaded."""
    predictor = server.get_predictor()
    
    models_loaded = {
        "pytorch": list(predictor.pytorch_models.keys()) if predictor else [],
        "sklearn": list(predictor.sklearn_models.keys()) if predictor else []
    }
    
    is_ready = len(models_loaded["pytorch"]) > 0 or len(models_loaded["sklearn"]) > 0
    
    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        models_loaded=models_loaded,
        uptime_seconds=server.get_uptime()
    )
