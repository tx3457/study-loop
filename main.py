import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Path,Query,Depends,HTTPException,Request
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import APIError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field
from routers.chat import router as chat_router
from routers.documents import router as document_router
from routers.quiz import router as quiz_router
from routers.learning_path import router as learning_path_router
from routers.session import router as session_router
from routers.wrong_questions import router as wrong_questions_router
from routers.user import router as user_router
from routers.orchestrator import router as orchestrator_router
from routers.stream import router as stream_router
from routers.eval import router as eval_router
from routers.autonomous import router as autonomous_router
from routers.adaptive import router as adaptive_router
from routers.audit import router as audit_router
from routers.tutor import router as tutor_router
from routers.health import router as health_router
from services.memory_persist import load_snapshot
from services.idempotency import IdempotencyConflictError
from services.provider_config import (
    ProviderDeadlineExceeded,
    ProviderConfigurationError,
    close_managed_provider_clients,
)
from services.retry import RetryExhausted
from services.tool_registry import SideEffectAmbiguousError


async def _load_memory_snapshot():
    """本机记忆快照兜底：InMemoryStore 重启后从 JSON 回灌（生产 DATABASE_URL→PostgresStore 时 no-op）。"""
    try:
        n = load_snapshot()
        if n:
            logging.getLogger(__name__).info(f"[startup] 跨会话记忆：从本机快照恢复 {n} 条")
    except Exception as e:
        logging.getLogger(__name__).warning(f"[startup] load_snapshot 失败（忽略）: {e}")


async def _connect_mcp_live_servers():
    """接入真实 MCP live server（灰度 MCP_LIVE_ENABLED）：联网搜索/抓取工具注册进 ToolRegistry。"""
    try:
        from services.mcp_servers import connect_and_register_all
        names = await connect_and_register_all()
        if names:
            logging.getLogger(__name__).info(f"[startup] MCP live 工具接入: {names}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"[startup] MCP live 接入失败（降级，无联网）: {e}")


async def _cleanup_mcp_live_servers():
    """释放 MCP live server 子进程。"""
    try:
        from services.mcp_servers import cleanup_all
        await cleanup_all()
    except Exception:
        pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Own startup resources and release them in reverse dependency order."""
    try:
        await _load_memory_snapshot()
        await _connect_mcp_live_servers()
        yield
    finally:
        try:
            await _cleanup_mcp_live_servers()
        finally:
            await close_managed_provider_clients()


app = FastAPI(
    title="StudyLoop API",
    description="Document-grounded adaptive tutoring and bounded tool-use agents.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev
        "http://127.0.0.1:5173",   # Vite dev（loopback 地址）
        "http://localhost:4001",   # Docker 前端（直接访问后端时）
        "http://127.0.0.1:4001",   # Docker 前端（loopback 地址）
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(document_router)
app.include_router(quiz_router)
app.include_router(learning_path_router)
app.include_router(session_router)
app.include_router(wrong_questions_router)
app.include_router(user_router)
app.include_router(orchestrator_router)
app.include_router(stream_router)
app.include_router(eval_router)
app.include_router(autonomous_router)
app.include_router(adaptive_router)
app.include_router(audit_router)
app.include_router(tutor_router)   # Supervisor-based MAS guided 辅导（灰度，默认 503）
app.include_router(health_router)


@app.get("/")
async def root():
    return {"name": "StudyLoop", "status": "ok"}



@app.exception_handler(ValueError)
async def deal(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"error": "参数错误", "detail": str(exc)})


@app.exception_handler(SideEffectAmbiguousError)
async def side_effect_ambiguous_handler(
    request: Request, exc: SideEffectAmbiguousError
):
    logging.getLogger(__name__).error(
        "tool result ambiguous; automatic retry forbidden: %s",
        exc.tool_name,
    )
    return JSONResponse(
        status_code=409,
        content={
            "detail": "工具执行结果不确定，请勿自动重试；请刷新学习状态后重新开始",
            "code": "side_effect_ambiguous",
            "reason": "ambiguous",
        },
    )


@app.exception_handler(IdempotencyConflictError)
async def idempotency_conflict_handler(
    request: Request, exc: IdempotencyConflictError
):
    details = {
        "payload_mismatch": "该 Idempotency-Key 已用于不同请求，请生成新 key",
        "in_progress": "相同请求正在处理中，请勿并发重复提交",
        "ambiguous": "此前请求可能已执行写操作，请刷新学习状态后重新开始",
    }
    return JSONResponse(
        status_code=409,
        content={
            "detail": details.get(exc.reason, "请求无法安全重复执行"),
            "code": "idempotency_conflict",
            "reason": exc.reason,
        },
    )


@app.exception_handler(RetryExhausted)
async def retry_exhausted_handler(request: Request, exc: RetryExhausted):
    logging.getLogger(__name__).warning("provider retries exhausted: %s", exc)
    return _provider_failure_response(exc)


def _exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _provider_failure_response(exc: BaseException) -> JSONResponse:
    chain = tuple(_exception_chain(exc))
    if any(isinstance(item, RateLimitError) for item in chain):
        return JSONResponse(
            status_code=429,
            content={
                "error": "模型服务请求过于频繁",
                "detail": "模型服务当前限流，请稍后重试",
                "code": "provider_rate_limited",
            },
        )
    if any(
        isinstance(item, (ProviderDeadlineExceeded, APITimeoutError))
        for item in chain
    ):
        return JSONResponse(
            status_code=504,
            content={
                "error": "模型服务请求超时",
                "detail": "模型服务未在时间预算内响应",
                "code": "provider_timeout",
            },
        )
    return JSONResponse(
        status_code=503,
        content={
            "error": "服务暂时不可用",
            "detail": "模型服务请求失败",
            "code": "provider_unavailable",
        },
    )


@app.exception_handler(ProviderDeadlineExceeded)
async def provider_deadline_handler(
    request: Request, exc: ProviderDeadlineExceeded
):
    logging.getLogger(__name__).warning(
        "provider request exceeded %.1fs deadline",
        exc.deadline_seconds,
    )
    return _provider_failure_response(exc)


@app.exception_handler(ProviderConfigurationError)
async def provider_configuration_handler(
    request: Request, exc: ProviderConfigurationError
):
    logging.getLogger(__name__).warning(
        "provider configuration invalid for %s: %s",
        exc.capability,
        ",".join(exc.issues),
    )
    return JSONResponse(
        status_code=503,
        content={
            "error": "服务暂时不可用",
            "detail": "模型服务尚未正确配置",
            "code": "provider_not_configured",
        },
    )


@app.exception_handler(APIError)
async def provider_error_handler(request: Request, exc: APIError):
    logging.getLogger(__name__).warning(
        "provider request failed: %s", type(exc).__name__
    )
    return _provider_failure_response(exc)
