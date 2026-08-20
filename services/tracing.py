"""
双路 Tracing：LangSmith + Langfuse

── 为什么双路 ──────────────────────────────────────────────────────────────
LangSmith：LangChain 原生栈，可按需启用，UI 深度适配 LangGraph。
Langfuse：开源可自托管，适合生产部署（合规 / 数据主权 / 自定义评估）。

统一入口 `@traceable` 让业务代码零修改，两个后端**同时**启用：
  - LangSmith 配置存在    → 走 langsmith.traceable span
  - Langfuse 配置存在      → 走 langfuse.observe span
  - 两者都配置            → 同一函数被两个装饰器包裹，trace 分别上报两个后端
  - 两者都未配置 / 未安装 → 零开销 no-op，原函数直接返回

── 自动 Tracing vs 自定义 Span ──────────────────────────────────────────────
LangGraph 内置自动追踪：设置环境变量后，orchestrator.ainvoke() / .astream()
自动生成完整的 Graph 执行追踪树（每个节点一个 span）。自定义 span 在关键
service 函数上附加语义元数据（strategy / k / CE enabled）。

Langfuse 针对 LangGraph 的自动追踪走 CallbackHandler（不是装饰器）：
  router 层调用 graph.ainvoke(input, config={"callbacks": [get_langfuse_callback()]})
  即可在 Langfuse UI 看到完整的 Graph 执行树。

── 配置（.env）─────────────────────────────────────────────────────────────
  # LangSmith（可选）
  LANGSMITH_TRACING=true
  LANGSMITH_API_KEY=lsv2_pt_xxx
  LANGSMITH_PROJECT=study-loop

  # Langfuse（可选）
  LANGFUSE_PUBLIC_KEY=pk-lf-xxx
  LANGFUSE_SECRET_KEY=sk-lf-xxx
  LANGFUSE_BASE_URL=https://cloud.langfuse.com      # 自托管可改

"""
import os
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

try:
    from langsmith import traceable as _ls_traceable
    _LANGSMITH_AVAILABLE = True
except ImportError:
    _LANGSMITH_AVAILABLE = False
    logger.debug("[tracing] langsmith not installed")

try:
    from langfuse import observe as _lf_observe
    from langfuse.langchain import CallbackHandler as _LangfuseCallbackHandler
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False
    _LangfuseCallbackHandler = None  # type: ignore
    logger.debug("[tracing] langfuse not installed")


# LangSmith run_type → Langfuse as_type 映射。
# Langfuse v4 as_type 合法值：generation / embedding / span / agent / tool /
# chain / retriever / evaluator / guardrail。未映射或 None → 默认 span。
_RUN_TYPE_TO_LANGFUSE_AS_TYPE: dict[str, str] = {
    "llm":       "generation",
    "embedding": "embedding",
    "retriever": "retriever",
    "tool":      "tool",
    "chain":     "chain",
}


def _env_flag(name: str, *, default: bool = False) -> bool:
    """Parse an opt-in environment flag without treating ``"false"`` as true."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes"}


def _langsmith_flag() -> bool:
    """Mirror the precedence and exact-``true`` semantics of langsmith 0.7.22."""
    for name in (
        "LANGSMITH_TRACING_V2",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING",
    ):
        raw = os.getenv(name)
        if raw is not None and raw.strip():
            return raw == "true"
    return False


def _langsmith_enabled() -> bool:
    return _LANGSMITH_AVAILABLE and _langsmith_flag()


def _langfuse_enabled() -> bool:
    return (
        _LANGFUSE_AVAILABLE
        and _env_flag("LANGFUSE_TRACING_ENABLED", default=True)
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY", "").strip())
        and bool(os.getenv("LANGFUSE_SECRET_KEY", "").strip())
    )


def traceable(
    *,
    name: str | None = None,
    run_type: str = "chain",
    metadata: dict | None = None,
) -> Callable:
    """统一 @traceable 装饰器工厂，同时写 LangSmith + Langfuse。

    Args:
        name:      span 名称（默认：函数名）
        run_type:  LangSmith span 类型，影响 UI 图标和过滤分类：
                   "retriever" | "chain" | "llm" | "tool" | "embedding"
                   （Langfuse 忽略此字段）
        metadata:  附加键值对，两者都会记录

    Example:
        @traceable(
            name="hybrid_retrieval",
            run_type="retriever",
            metadata={"strategy": "bm25+vector+rrf", "k": 60},
        )
        async def hybrid_query_document(document_id: str, query: str):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        wrapped = fn

        # ── 内层：Langfuse @observe ──────────────────────────────────────
        # Langfuse v4 @observe 签名：name / as_type / capture_input / capture_output。
        # run_type 字符串映射到 Langfuse 的 as_type（不匹配的 fallback 到 'span'）。
        # metadata 不走 @observe（v4 不接受此 kwarg），仅由 LangSmith 承载。
        # 需要动态 metadata 时，业务代码里调 langfuse.update_current_observation(...)。
        if _langfuse_enabled():
            lf_kwargs: dict[str, Any] = {}
            if name:
                lf_kwargs["name"] = name
            as_type = _RUN_TYPE_TO_LANGFUSE_AS_TYPE.get(run_type)
            if as_type:
                lf_kwargs["as_type"] = as_type
            wrapped = _lf_observe(**lf_kwargs)(wrapped)

        # ── 外层：LangSmith @traceable（保持现有语义）─────────────────────
        if _langsmith_enabled():
            ls_kwargs: dict[str, Any] = {"run_type": run_type}
            if name:
                ls_kwargs["name"] = name
            if metadata:
                ls_kwargs["metadata"] = metadata
            wrapped = _ls_traceable(**ls_kwargs)(wrapped)

        return wrapped

    return decorator


def get_langfuse_callback():
    """返回 Langfuse LangChain CallbackHandler 实例，供 graph.ainvoke 使用。

    未配置 / 未安装时返回 None，router 层需判空。

    Example:
        from services.tracing import get_langfuse_callback
        callbacks = [cb for cb in [get_langfuse_callback()] if cb]
        await graph.ainvoke(input, config={"callbacks": callbacks, ...})
    """
    if not _langfuse_enabled():
        return None
    return _LangfuseCallbackHandler()


def get_agent_callbacks() -> list:
    """返回当前所有启用的 Agent 级 callback 列表。router 调用此函数简化逻辑。"""
    callbacks = []
    lf_cb = get_langfuse_callback()
    if lf_cb is not None:
        callbacks.append(lf_cb)
    return callbacks


def build_trace_config(
    *,
    thread_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    """统一构造 graph.ainvoke / graph.astream 的 config 字典。

    负责三件事：
      1. 注入 callbacks（Langfuse handler，LangSmith 走环境变量不需要 callback）
      2. 设置 configurable.thread_id（LangGraph Checkpointer 要求）
      3. 将 user_id / session_id / tags 按 Langfuse 约定写入 metadata
         key（langfuse_user_id / langfuse_session_id / langfuse_tags），
         CallbackHandler 会自动提取为 trace 顶层属性，UI 可按用户 / 会话聚合。

    Langfuse 未启用时这些 metadata 仅是普通键值对，不会引发错误。

    Args:
        thread_id:       LangGraph Checkpointer 的会话键（可选）
        user_id:         用户标识，用于 Langfuse Users 面板聚合
        session_id:      会话标识，用于 Langfuse Sessions 面板聚合（多轮对话同一 session）
        tags:            附加标签，便于按 action / env 过滤 trace
        extra_metadata:  额外自定义 metadata（会与 langfuse_* 合并）

    Example:
        config = build_trace_config(
            thread_id=req.document_id,
            user_id=req.user_id,
            session_id=req.session_id or req.document_id,
            tags=[f"action:{req.action}"],
        )
        await graph.ainvoke(input, config=config)
    """
    metadata: dict = {}
    if user_id:
        metadata["langfuse_user_id"] = user_id
    if session_id:
        metadata["langfuse_session_id"] = session_id
    if tags:
        metadata["langfuse_tags"] = tags
    if extra_metadata:
        metadata.update(extra_metadata)

    config: dict = {"callbacks": get_agent_callbacks()}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    if metadata:
        config["metadata"] = metadata
    return config
