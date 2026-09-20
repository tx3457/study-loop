import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import APIError, APITimeoutError, RateLimitError
from agents.supervisor import supervisor_enabled
from routers.chat import router as chat_router
from routers.documents import router as document_router
from routers.knowledge import router as knowledge_router
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
from routers.health import router as health_router
from services.auth import require_user_id, verify_configuration
from services.memory_persist import load_snapshot, persist_snapshot
from services.idempotency import (
    IdempotencyConflictError,
    InvalidIdempotencyKeyError,
)
from services.provider_config import (
    ProviderDeadlineExceeded,
    ProviderConfigurationError,
    close_managed_provider_clients,
)
from services.retry import RetryExhausted
from services.tool_registry import SideEffectAmbiguousError
from services.quiz_sessions import QuizSessionApiError
from services.request_context import (
    RequestContextMiddleware,
    SafeErrorMiddleware,
    current_request_id,
    public_error_payload,
)
from services.origin_guard import (
    ALLOWED_BROWSER_ORIGINS,
    OriginGuardMiddleware,
)


logger = logging.getLogger(__name__)


async def _load_memory_snapshot():
    """本机记忆快照兜底：InMemoryStore 重启后从 JSON 回灌（生产 DATABASE_URL→PostgresStore 时 no-op）。"""
    try:
        n = load_snapshot()
        if n:
            logger.info("[startup] 跨会话记忆：从本机快照恢复 %s 条", n)
    except Exception as exc:
        logger.warning(
            "[startup] load_snapshot failed; continuing error_type=%s",
            type(exc).__name__,
        )


async def _save_memory_snapshot():
    """正常退出前刷新本机记忆快照；PostgreSQL 后端自动 no-op。"""
    try:
        await persist_snapshot()
    except Exception as exc:
        logger.warning(
            "[shutdown] persist_snapshot failed; continuing error_type=%s",
            type(exc).__name__,
        )


async def _connect_mcp_live_servers():
    """接入真实 MCP live server（灰度 MCP_LIVE_ENABLED）：联网搜索/抓取工具注册进 ToolRegistry。"""
    try:
        from services.mcp_servers import connect_and_register_all
        names = await connect_and_register_all()
        if names:
            logger.info("[startup] MCP live tools connected: count=%d", len(names))
    except Exception as exc:
        logger.warning(
            "[startup] MCP live unavailable; continuing error_type=%s",
            type(exc).__name__,
        )


async def _cleanup_mcp_live_servers():
    """释放 MCP live server 子进程。"""
    try:
        from services.mcp_servers import cleanup_all
        await cleanup_all()
    except Exception:
        pass


async def _shutdown_vectorstore_io():
    """Give owned embedded-Chroma work a bounded opportunity to finish."""
    try:
        from services.vectorstore import shutdown_vectorstore_io

        await shutdown_vectorstore_io()
    except Exception as exc:
        logger.warning(
            "[shutdown] vectorstore I/O cleanup failed; continuing error_type=%s",
            type(exc).__name__,
        )


async def _start_vectorstore_io():
    """Open a fresh embedded-Chroma lifecycle before accepting requests."""
    from services.vectorstore import start_vectorstore_io

    start_vectorstore_io()


def _include_experimental_routers(application: FastAPI) -> bool:
    """Register opt-in Lab APIs only when enabled at process startup."""
    if not supervisor_enabled():
        return False
    from routers.tutor import router as tutor_router

    application.include_router(tutor_router, dependencies=_AUTHENTICATED)
    return True


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Own startup resources and release them in reverse dependency order."""
    try:
        # 令牌配置错误在这里就让进程起不来，而不是每个请求各报一次 500。
        verify_configuration()
        await _start_vectorstore_io()
        await _load_memory_snapshot()
        await _connect_mcp_live_servers()
        yield
    finally:
        try:
            await _cleanup_mcp_live_servers()
        finally:
            try:
                await _save_memory_snapshot()
            finally:
                try:
                    await _shutdown_vectorstore_io()
                finally:
                    await close_managed_provider_clients()


app = FastAPI(
    title="StudyLoop API",
    description="Document-grounded adaptive tutoring and bounded tool-use agents.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(SafeErrorMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_BROWSER_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)
app.add_middleware(OriginGuardMiddleware)
app.add_middleware(RequestContextMiddleware)

# 闸门挂在 include_router 上而不是逐个端点：默认关闭，将来新增的端点
# 自动落在闸门内侧，不会因为有人忘了加 Depends 而漏出去。
# 豁免的只有 "/"、/health/live 和 /health/ready，由 tests/test_auth_subject.py 钉死。
_AUTHENTICATED = [Depends(require_user_id)]

app.include_router(chat_router, dependencies=_AUTHENTICATED)
app.include_router(document_router, dependencies=_AUTHENTICATED)
app.include_router(knowledge_router, dependencies=_AUTHENTICATED)
app.include_router(quiz_router, dependencies=_AUTHENTICATED)
app.include_router(learning_path_router, dependencies=_AUTHENTICATED)
app.include_router(session_router, dependencies=_AUTHENTICATED)
app.include_router(wrong_questions_router, dependencies=_AUTHENTICATED)
app.include_router(user_router, dependencies=_AUTHENTICATED)
app.include_router(orchestrator_router, dependencies=_AUTHENTICATED)
app.include_router(stream_router, dependencies=_AUTHENTICATED)
app.include_router(eval_router, dependencies=_AUTHENTICATED)
app.include_router(autonomous_router, dependencies=_AUTHENTICATED)
app.include_router(adaptive_router, dependencies=_AUTHENTICATED)
app.include_router(audit_router, dependencies=_AUTHENTICATED)
_include_experimental_routers(app)
app.include_router(health_router)


@app.get("/")
async def root():
    return {"name": "StudyLoop", "status": "ok"}



@app.exception_handler(InvalidIdempotencyKeyError)
async def invalid_idempotency_key_handler(
    request: Request, exc: InvalidIdempotencyKeyError
):
    return JSONResponse(
        status_code=400,
        content={
            "error": "参数错误",
            "detail": str(exc),
            "code": "invalid_idempotency_key",
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Unhandled ValueError means an internal contract broke, not a bad request.

    Client-side validation already fails earlier: FastAPI raises
    RequestValidationError (422) for malformed bodies, and routers raise explicit
    HTTPException for domain errors (see routers/session.py and
    routers/wrong_questions.py). Anything still reaching this handler is an
    internal invariant failure — for example the retrieval alignment assertion in
    services/tools.py or a pydantic ValidationError on an outbound model, both of
    which are ValueError subclasses. Reporting those as 400 hid real 500s from
    error budgets and alerting, so this handler reports 500 and stays opaque both
    on the wire and in the log.
    """
    # 只记类型和 request_id，不带 exc_info：ValueError 的消息可能含 provider
    # 连接串等凭据（见 tests/test_request_error_boundaries.py 的脱敏用例），
    # traceback 会把它一起写进日志。定位靠 request_id 关联。
    logger.error(
        "unhandled ValueError escaped to the HTTP boundary request_id=%s error_type=%s",
        current_request_id(),
        type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content=public_error_payload(
            error="服务内部错误",
            detail="请求处理失败，请稍后重试",
            code="internal_error",
            include_request_id=True,
        ),
    )


@app.exception_handler(QuizSessionApiError)
async def quiz_session_error_handler(request: Request, exc: QuizSessionApiError):
    content = {"detail": exc.detail, "code": exc.code}
    if exc.reason is not None:
        content["reason"] = exc.reason
    return JSONResponse(status_code=exc.status_code, content=content)


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
    logger.warning(
        "provider retries exhausted request_id=%s error_type=%s",
        current_request_id(),
        type(exc).__name__,
    )
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
