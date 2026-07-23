"""
Orchestrator：多 Agent 编排入口

完整 Resilience 流程（容错维度的 defense-in-depth；非完整 Agent Harness）：
  input_guard → route → agents... → (critic_adapter)? → output_guard → END

路由规则（根据 action 字段）：
  "quiz"  → adapt_reader → quiz_agent → critic_adapter → (重出 or output_guard)
  "grade" → grader_agent → adapt_writer → output_guard → END
  "plan"  → planner_agent → output_guard → END

Critic 质量门：
  quiz_agent 完成后插入独立 critic_agent subgraph 节点（critic_adapter）。
  critic 输出 CritiqueReport → 若 overall<0.7 或含 high severity 建议
  且 revision_count<2 → 回 quiz_agent 重出；否则 → output_guard。

  通过环境变量 CRITIC_ENABLED 控制（默认 true，ablation 时设 false）。

Guardrail 位置：
  input_guard  — 在所有 Agent 之前：fail-fast，校验失败不浪费 LLM token
  output_guard — 在所有 Agent 之后：兜底，防止格式异常数据流入 API 响应
"""
import json
import logging
import os

from langgraph.graph import StateGraph, START, END

from agents.state import OrchestratorState
from agents.adapt_agent import adapt_reader, adapt_writer
from agents.quiz_agent import quiz_agent
from agents.grader_agent import grader_agent
from agents.planner_agent import planner_agent
from agents.critic_agent import critic_agent
from agents.reviser_agent import reviser_agent
from agents.guardrails import input_guard, output_guard
from services.tools import dispatch_tool

logger = logging.getLogger(__name__)


def _critic_enabled() -> bool:
    """通过环境变量 CRITIC_ENABLED 控制 critic 节点是否启用（ablation 用）"""
    return os.getenv("CRITIC_ENABLED", "true").lower() in ("1", "true", "yes")


def _reviser_enabled() -> bool:
    """REVISER_ENABLED=true → critic 不通过走 reviser 精修
    false → critic 不通过走 quiz_agent 整轮重跑"""
    return os.getenv("REVISER_ENABLED", "true").lower() in ("1", "true", "yes")


def _format_reflected_message(critique: dict) -> str:
    """把 CritiqueReport 格式化为给下一轮 quiz_agent 的反思文本。

    将三维度低分原因 + suggestions 拼成结构化反思文本，让 LLM 知道上一轮
    为什么被拒，而不只是机械重跑。
    """
    if not critique:
        return ""

    lines = ["【上一轮出题被 Critic 拒绝,请根据以下反馈重新生成】", ""]

    overall = critique.get("overall_score", 0.0)
    lines.append(f"整体评分:{overall:.2f} / 1.0（阈值 0.7）")
    lines.append("")

    # 列出各维度低分原因(score < 0.7 的)
    low_dims = []
    for dim_key, dim_label in (("difficulty", "难度"), ("relevance", "相关性"), ("coverage", "覆盖度")):
        d = critique.get(dim_key, {}) or {}
        score = d.get("score", 1.0)
        reasoning = d.get("reasoning", "")
        if score < 0.7 and reasoning:
            low_dims.append(f"- [{dim_label}] 评分 {score:.2f}:{reasoning}")
    if low_dims:
        lines.append("低分维度:")
        lines.extend(low_dims)
        lines.append("")

    # 具体改进建议(取最严重的 5 条)
    suggestions = critique.get("suggestions", []) or []
    if suggestions:
        # 按 severity 排序:high > medium > low
        severity_rank = {"high": 0, "medium": 1, "low": 2}
        sorted_sugs = sorted(
            (s for s in suggestions if isinstance(s, dict)),
            key=lambda s: severity_rank.get(s.get("severity", "low"), 3),
        )
        lines.append("具体改进建议:")
        for i, s in enumerate(sorted_sugs[:5], 1):
            target = s.get("target", "general")
            severity = s.get("severity", "medium")
            action = s.get("action", "")
            lines.append(f"  {i}. [{severity}·{target}] {action}")
        lines.append("")

    lines.append("请在新一轮生成时严格遵守以上反馈,避免重复同样的问题。")
    return "\n".join(lines)


def _route(state: OrchestratorState) -> str:
    """根据 action 字段决定进入哪条 Agent 流水线。"""
    action = state.get("action", "quiz")
    if action == "plan":
        return "planner_agent"
    elif action == "grade":
        return "grader_agent"
    else:
        return "adapt_reader"


async def _critic_adapter(state: OrchestratorState) -> dict:
    """Adapter：包装 critic_agent subgraph 调用，把 OrchestratorState 转 CriticState。

    设计：critic 自主拉一次最新 chunks 作为评估证据（不依赖 quiz_agent 内部 chunks 透传）。
    revision_count 在此处 +1（语义：critic 评估次数 = tutor 出题轮次）。
    """
    if not _critic_enabled():
        logger.info("[critic_adapter] CRITIC_ENABLED=false, skip critic evaluation")
        return {}

    # 拉一次最新 chunks 给 critic 作为证据（独立于 quiz_agent 内部检索）
    chunks: list[str] = []
    description = state.get("description", "")
    document_id = state.get("document_id", "")
    if description and document_id:
        try:
            result_json = await dispatch_tool(
                "search_document",
                {"document_id": document_id, "query": description},
            )
            chunks = json.loads(result_json).get("chunks", [])
        except Exception as e:
            logger.warning(f"[critic_adapter] pre-fetch chunks failed: {e}")

    critic_input = {
        "quiz": state.get("quiz", {}),
        "chunks": chunks,
        "user_id": state.get("user_id", ""),
        "document_id": document_id,
        "difficulty_score": state.get("difficulty_score", 0.5),
        "weak_points": state.get("weak_points", []),
        # 把上游 sufficiency_check 的诊断传给 critic，让它对 relevance 维度更严格
        "insufficient_evidence": state.get("insufficient_evidence", False),
    }

    try:
        result = await critic_agent.ainvoke(critic_input)
        critique = result.get("critique", {})
    except Exception as e:
        logger.exception(f"[critic_adapter] critic_agent invoke failed: {e}")
        critique = {}

    history = list(state.get("critique_history", []))
    if critique:
        history.append(critique)

    # Reflection 回灌:把 critique 转成给下一轮 quiz_agent 看的反思文本
    # 即使本轮 critic 通过(_should_revise → output_guard),也写入字段,
    # 让 reflected_message 与 critique_history 一一对应,便于后续审计
    reflected = _format_reflected_message(critique)

    return {
        "critique_history": history,
        "revision_count": state.get("revision_count", 0) + 1,
        "reflected_message": reflected,
    }


def _should_revise(state: OrchestratorState) -> str:
    """条件路由：critic 评分低 + 未到重出上限 → 回 quiz_agent；否则 → output_guard"""
    if not _critic_enabled():
        return "output_guard"

    history = state.get("critique_history", [])
    if not history:
        return "output_guard"

    if state.get("revision_count", 0) >= 2:
        return "output_guard"

    latest = history[-1]
    overall = latest.get("overall_score", 1.0)
    has_high = any(
        s.get("severity") == "high"
        for s in latest.get("suggestions", [])
        if isinstance(s, dict)
    )
    needs_revise = (overall < 0.7) or has_high

    if needs_revise:
        next_node = "reviser" if _reviser_enabled() else "quiz_agent"
        logger.info(
            f"[orchestrator] critic triggers revision: overall={overall:.2f} "
            f"has_high={has_high} count={state.get('revision_count', 0)} "
            f"→ {next_node}"
        )
        return next_node
    return "output_guard"


_builder = StateGraph(OrchestratorState)

# ── 注册节点 ────────────────────────────────────────────────────────────────
_builder.add_node("input_guard",     input_guard)      # Resilience：输入校验
_builder.add_node("adapt_reader",    adapt_reader)     # 读画像 → difficulty_score / weak_points
_builder.add_node("quiz_agent",      quiz_agent)        # Hybrid检索 + CE出题 + 审核 → quiz
_builder.add_node("critic_adapter",  _critic_adapter)   # 包装 critic_agent subgraph
_builder.add_node("reviser",         reviser_agent)    # reviewer↔reviser 循环子图
_builder.add_node("grader_agent",    grader_agent)     # AI批改 → grading_report
_builder.add_node("adapt_writer",    adapt_writer)     # 写回画像（EMA更新掌握度）
_builder.add_node("planner_agent",   planner_agent)    # 学习路径生成 → learning_path
_builder.add_node("output_guard",    output_guard)     # Resilience：输出校验

# ── Resilience 入口：START → input_guard → 路由 ──────────────────────────────
_builder.add_edge(START, "input_guard")
_builder.add_conditional_edges("input_guard", _route, {
    "adapt_reader":  "adapt_reader",
    "grader_agent":  "grader_agent",
    "planner_agent": "planner_agent",
})

# ── Quiz 流 + reviewer↔reviser 循环子图 ─────────────────────────────────────
#   adapt_reader → quiz_agent → critic_adapter
#                                    │
#                ┌───────────────────┼─────────────────────┐
#                │ pass               │ revise(REVISER_ENABLED)│ full regenerate
#                ↓                    ↓                     ↓
#           output_guard          reviser              quiz_agent
#                                    │                     │
#                                    └──────→ critic_adapter ←┘  (循环重审)
_builder.add_edge("adapt_reader",   "quiz_agent")
_builder.add_edge("quiz_agent",     "critic_adapter")
_builder.add_conditional_edges("critic_adapter", _should_revise, {
    "quiz_agent":    "quiz_agent",     # critic 低分 → 回 quiz_agent 整轮重跑
    "reviser":       "reviser",        # critic 低分 → reviser 精修（只改题）
    "output_guard":  "output_guard",   # critic 通过 → 收尾
})
_builder.add_edge("reviser",        "critic_adapter")  # reviser → critic 重审,形成循环

# ── Grade 流：grader_agent → adapt_writer → output_guard ──────────────────
_builder.add_edge("grader_agent", "adapt_writer")
_builder.add_edge("adapt_writer", "output_guard")

# ── Plan 流：planner_agent → output_guard ─────────────────────────────────
_builder.add_edge("planner_agent", "output_guard")

# ── Resilience 出口：output_guard → END ──────────────────────────────────
_builder.add_edge("output_guard", END)

orchestrator = _builder.compile()


# ── Durable Checkpointer 工厂 ───────────────────────────────────────────────
# 同步 orchestrator 单例向后兼容(无 checkpointer);新代码可用 factory 接 sqlite
# checkpointer 实现进程崩溃恢复。生命周期由调用方管理(async with)。
def compile_with_checkpointer(checkpointer):
    """用同一个 StateGraph builder 编译一份带 checkpointer 的 orchestrator。

    Args:
        checkpointer: LangGraph BaseCheckpointSaver 实例(如 AsyncSqliteSaver)

    Returns:
        CompiledGraph,调用 ainvoke 时必须传 config={"configurable": {"thread_id": ...}}
    """
    return _builder.compile(checkpointer=checkpointer)
