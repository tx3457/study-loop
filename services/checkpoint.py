"""
LangGraph Durable Checkpointer(Phase 9 P0-4)

设计来源:直接 wrap langgraph 原生 AsyncSqliteSaver
(reference/langgraph/libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/aio.py:509-559),
不自己造轮子。

为什么需要 checkpointer?
  没有它:LangGraph 是无状态执行,进程挂掉 / 服务重启后,
          多轮交互(Tutor↔Critic 反思循环)的 state 全部丢失,用户必须从头开始。
  有了它:每个节点执行后自动 INSERT 一行 (thread_id, checkpoint_id, parent_id, blob),
          崩溃恢复时用同 thread_id 调 ainvoke,自动从最近 checkpoint 续跑。

使用模式(异步上下文,确保 sqlite 连接正确释放):
    from services.checkpoint import open_sqlite_checkpointer
    from agents.orchestrator import compile_with_checkpointer

    async with open_sqlite_checkpointer("./.checkpoints/orch.db") as cp:
        orch = compile_with_checkpointer(cp)
        config = {"configurable": {"thread_id": "user_abc_session_1"}}
        result = await orch.ainvoke({"action": "quiz", ...}, config=config)
        # 进程崩溃后,用同 thread_id 再次 ainvoke → 自动恢复

env 开关:
  CHECKPOINT_ENABLED=false → 上层应跳过 checkpointer 路径,用原 orchestrator 单例
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger(__name__)


def checkpoint_enabled() -> bool:
    """env 开关,ablation 实验或测试时可关闭"""
    return os.getenv("CHECKPOINT_ENABLED", "true").lower() in ("1", "true", "yes")


def default_checkpoint_path() -> str:
    """默认 sqlite 路径(项目根 .checkpoints/orchestrator.db)"""
    return os.getenv(
        "CHECKPOINT_DB_PATH",
        str(Path(__file__).parent.parent / ".checkpoints" / "orchestrator.db"),
    )


@asynccontextmanager
async def open_sqlite_checkpointer(db_path: str | None = None) -> AsyncIterator:
    """异步上下文:打开 AsyncSqliteSaver,退出时自动释放连接。

    用法:
        async with open_sqlite_checkpointer() as cp:
            orch = compile_with_checkpointer(cp)
            await orch.ainvoke(state, config={"configurable": {"thread_id": "t1"}})
    """
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError as e:
        raise RuntimeError(
            f"langgraph-checkpoint-sqlite not installed: {e}. "
            "pip install langgraph-checkpoint-sqlite aiosqlite"
        ) from e

    path = db_path or default_checkpoint_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[checkpoint] opening AsyncSqliteSaver at {path}")

    async with AsyncSqliteSaver.from_conn_string(path) as cp:
        yield cp
