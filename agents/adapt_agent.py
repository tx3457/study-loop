"""
AdaptAgent：精准 bank 读取 + decision_log

两个 subgraph：
  adapt_reader  → 出题前：精准读 mastery / weak_points / preferences 三个 bank
                  + 写一条 decision_log 记录"为什么选这个难度"
  adapt_writer  → 批改后：write_episodic_memory + update_semantic_memory（兼容 API 内部分发到语义 bank）

decision_log 将 Adapt 的"推理过程"持久化成可审计事件。
critic / 评估系统可以查 decision_log 看"为什么是这个难度而不是别的"。
"""
import logging

from langgraph.graph import END, START, StateGraph

from agents.state import OrchestratorState
from models.grader import GradingReport
from services.memory import (
    append_decision,
    commit_learning_memory,
    get_mastery,
    get_preferences,
    get_weak_points,
)
from services.session import sessions
from services.tracing import traceable

_DIFFICULTY_MAP = {"easy": 0.2, "medium": 0.5, "hard": 0.8}
logger = logging.getLogger(__name__)


# ── adapt_reader：读 3 个 bank，决定下次出题参数 ─────────────────────────────
@traceable(name="adapt_agent.read_profile", run_type="chain")
async def _read_profile(state: OrchestratorState) -> dict:
    """精准读 mastery / weak_points / preferences；写一条 decision_log。

    优先级：
      1. mastery（per-doc）有数据 → difficulty_score = mastery + 0.15（最近发展区）
      2. preferences.preferred_difficulty 有声明 → 用偏好（_DIFFICULTY_MAP 映射）
      3. 都没有 → state.difficulty fallback（前端传入或 'medium' 默认）
    """
    user_id = state["user_id"]
    document_id = state["document_id"]

    # 精准 3 bank 读，不再拉一坨 profile
    # weak_points 按当前文档隔离，避免别领域薄弱点污染本文档出题
    mastery_val = await get_mastery(user_id, document_id)
    weak_points = await get_weak_points(user_id, document_id)
    prefs = await get_preferences(user_id)

    # 决策推理（可写入 decision_log）
    if mastery_val is not None:
        difficulty_score = round(min(mastery_val + 0.15, 1.0), 2)
        rationale = f"mastery={mastery_val:.2f}, ZPD policy +0.15 → {difficulty_score:.2f}"
        source = "mastery"
    elif prefs.get("preferred_difficulty") in _DIFFICULTY_MAP:
        difficulty_score = _DIFFICULTY_MAP[prefs["preferred_difficulty"]]
        rationale = f"no mastery, fallback to preferences.preferred_difficulty='{prefs['preferred_difficulty']}'"
        source = "preferences"
    else:
        fallback = state.get("difficulty", "medium")
        difficulty_score = _DIFFICULTY_MAP.get(fallback, 0.5)
        rationale = f"cold start, fallback to state.difficulty='{fallback}' → {difficulty_score}"
        source = "cold_start"

    # 持久化决策（可供 critic / 评估系统审计）
    await append_decision(user_id, {
        "agent": "adapt_reader",
        "document_id": document_id,
        "difficulty_score": difficulty_score,
        "weak_points_count": len(weak_points),
        "source": source,
        "rationale": rationale,
    })

    return {"difficulty_score": difficulty_score, "weak_points": weak_points}


_rb = StateGraph(OrchestratorState)
_rb.add_node("read_profile", _read_profile)
_rb.add_edge(START, "read_profile")
_rb.add_edge("read_profile", END)
adapt_reader = _rb.compile()


# ── adapt_writer：批改后写回语义 bank ──────────────────────────────────────
@traceable(name="adapt_agent.write_profile", run_type="chain")
async def _write_profile(state: OrchestratorState) -> dict:
    """批改后：session_briefs + error_log + mastery + weak_points 全部更新。"""
    report = GradingReport.model_validate(state["grading_report"])
    session = sessions.get(report.session_id)
    if session is None:
        raise ValueError(f"Session {report.session_id} not found for profile write")

    user_id = session.user_id
    document_id = session.document_id
    if (
        state.get("user_id") not in {None, user_id}
        or state.get("document_id") not in {None, document_id}
    ):
        logger.warning(
            "[adapt_writer] ignored mismatched state ownership for session %s",
            report.session_id,
        )

    async def record_decision() -> None:
        # 一条 decision_log 标记写入了哪些 bank（便于审计）
        await append_decision(user_id, {
            "decision_id": f"adapt_writer:{report.session_id}",
            "agent": "adapt_writer",
            "document_id": document_id,
            "session_id": report.session_id,
            "score": report.score,
            "errors_logged": sum(1 for g in report.grades if not g.is_correct),
            "rationale": (
                f"session {report.session_id} 完成，写入 "
                "session_briefs/error_log/mastery/weak_points"
            ),
        })

    await commit_learning_memory(
        user_id,
        report,
        document_id,
        questions=session.questions,
        after_write=record_decision,
        on_core_written=lambda: setattr(session, "profile_written", True),
    )
    return {}


_wb = StateGraph(OrchestratorState)
_wb.add_node("write_profile", _write_profile)
_wb.add_edge(START, "write_profile")
_wb.add_edge("write_profile", END)
adapt_writer = _wb.compile()
