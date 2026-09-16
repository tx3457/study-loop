"""Operational health endpoints."""

from fastapi import APIRouter, Depends, Response, status

from models.health import StorageReadinessResponse
from services.auth import auth_enabled, require_user_id
from services.provider_health import provider_health_checker
from services.storage_readiness import storage_readiness_checker


router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def liveness():
    """Process liveness only; never contacts a model provider.

    也报告访问令牌闸门是否启用：匿名模式下每个调用方都被当作同一个默认用户，
    部署方需要一个不必先猜出令牌就能看到这件事的地方。
    """
    return {
        "name": "StudyLoop",
        "status": "ok",
        "auth": "token" if auth_enabled() else "anonymous",
    }


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


# 探针（/live、/ready）必须对容器编排开放，但这个端点会真的向 provider 发请求：
# 开放它等于给未认证调用方一个烧额度、顺带探测配置状态的入口。
@router.get("/providers", dependencies=[Depends(require_user_id)])
async def provider_health(response: Response):
    """Check provider model catalogs without generation or embedding calls."""
    result = await provider_health_checker.check()
    if result["status"] != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result
