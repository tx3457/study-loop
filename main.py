import logging

from fastapi import FastAPI, Path,Query,Depends,HTTPException,Request
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import APIError
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
from services.memory_persist import load_snapshot
from services.retry import RetryExhausted


app = FastAPI(
    title="StudyLoop API",
    description="Document-grounded adaptive tutoring and bounded tool-use agents.",
    version="0.1.0",
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
app.include_router(tutor_router)   # Phase 2: supervisor-based MAS guided 辅导（灰度，默认 503）


@app.on_event("startup")
async def _load_memory_snapshot():
    """本机记忆快照兜底：InMemoryStore 重启后从 JSON 回灌（生产 DATABASE_URL→PostgresStore 时 no-op）。"""
    try:
        n = load_snapshot()
        if n:
            logging.getLogger(__name__).info(f"[startup] 跨会话记忆：从本机快照恢复 {n} 条")
    except Exception as e:
        logging.getLogger(__name__).warning(f"[startup] load_snapshot 失败（忽略）: {e}")


@app.on_event("startup")
async def _connect_mcp_live_servers():
    """接入真实 MCP live server（灰度 MCP_LIVE_ENABLED）：联网搜索/抓取工具注册进 ToolRegistry。"""
    try:
        from services.mcp_servers import connect_and_register_all
        names = await connect_and_register_all()
        if names:
            logging.getLogger(__name__).info(f"[startup] MCP live 工具接入: {names}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"[startup] MCP live 接入失败（降级，无联网）: {e}")


@app.on_event("shutdown")
async def _cleanup_mcp_live_servers():
    """释放 MCP live server 子进程。"""
    try:
        from services.mcp_servers import cleanup_all
        await cleanup_all()
    except Exception:
        pass


@app.get("/")
async def root():
    return {"name": "StudyLoop", "status": "ok"}



@app.exception_handler(ValueError)
async def deal(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"error": "参数错误", "detail": str(exc)})

@app.exception_handler(RetryExhausted)
async def retry_exhausted_handler(request: Request, exc: RetryExhausted):
    logging.getLogger(__name__).warning("provider retries exhausted: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"error": "服务暂时不可用", "detail": "模型服务请求失败"},
    )


@app.exception_handler(APIError)
async def provider_error_handler(request: Request, exc: APIError):
    logging.getLogger(__name__).warning(
        "provider request failed: %s", type(exc).__name__
    )
    return JSONResponse(
        status_code=503,
        content={"error": "服务暂时不可用", "detail": "模型服务请求失败"},
    )
