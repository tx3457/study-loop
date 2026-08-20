"""Operational health endpoints."""

from fastapi import APIRouter, Response, status

from models.health import StorageReadinessResponse
from services.provider_health import provider_health_checker
from services.storage_readiness import storage_readiness_checker


router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def liveness():
    """Process liveness only; never contacts a model provider."""
    return {"name": "StudyLoop", "status": "ok"}


@router.get(
    "/ready",
    response_model=StorageReadinessResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": StorageReadinessResponse,
            "description": "A required storage component is unavailable.",
        }
    },
)
async def storage_readiness(response: Response):
    """Check required storage without contacting a model provider."""
    result = await storage_readiness_checker.check()
    if result["status"] != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result


@router.get("/providers")
async def provider_health(response: Response):
    """Check provider model catalogs without generation or embedding calls."""
    result = await provider_health_checker.check()
    if result["status"] != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result
