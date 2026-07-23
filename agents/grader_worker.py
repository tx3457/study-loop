"""
Grader Worker：supervisor-based MAS 的批改 worker 包装

不改 grader_agent.py / adapt_agent.py 源码，用 adapter 复用三件事：
  ① grader_agent：调 services.grader.grade_session 批改 → grading_report
  ② adapt_writer：批改后画像 EMA 写回（write_episodic_memory + update_semantic_memory）
  ③ 轨迹回填：把本轮压成一条 AdaptiveTurn dict 追加进 history + 写 last_report

与 routers/adaptive._grade_and_update 同构（填答案 → 批改 → 更新画像 → 回填轨迹），
但这里跑在 tutor_graph 的 supervisor⇄worker 循环里，且 grading 由上游
wait_for_answers interrupt 恢复的 answers 驱动（answers 已通过 session 写入）。

回填的 AdaptiveTurn dict 字段（与 models.adaptive.AdaptiveTurn 对齐，便于前端画曲线）：
  turn / action / topic / difficulty_score / score / mastery_after / knowledge_gaps
"""
import logging

from agents.adapt_agent import adapt_writer
from agents.grader_agent import grader_agent
from agents.state import TutorState
from models.grader import GradingReport
from services.memory import consolidate_session_extras, get_mastery
from services.srs import update_after_session

logger = logging.getLogger(__name__)


async def grader_worker(state: TutorState) -> dict:
    """批改 worker：grader_agent → adapt_writer 画像写回 → 轨迹回填 last_report/history。

    设计：grader_agent 读 state["session_id"] 批改（answers 已在上游写入该 session），
    产出 grading_report。随后调 adapt_writer 做画像 EMA 写回（复用其 decision_log 审计），
    最后把本轮压成 AdaptiveTurn dict 追加进 history，并写 last_report 供 supervisor 决策。

    重置质量门标记（critic_passed/quiz_served/critique_history/revision_count），
    使 supervisor 决定下一轮 quiz 时重新走 critic→reviser↔critic→await_answers 全门。
    """
    # ① 批改（grader_agent 复用，写 grading_report）
    grade_result = await grader_agent.ainvoke(state)
    report_dict = grade_result.get("grading_report")
    if not report_dict:
        logger.warning("[grader_worker] grader_agent 未产出 grading_report")
        return {}

    # ② 画像 EMA 写回（adapt_writer 复用：write_episodic_memory + update_semantic_memory）
    #    adapt_writer 从 state["grading_report"] 读，故先把 report 注入临时 state 透传。
    writer_state = {**state, "grading_report": report_dict}
    try:
        await adapt_writer.ainvoke(writer_state)
    except Exception as e:
        logger.warning(f"[grader_worker] adapt_writer 画像写回失败（不中断闭环）: {e}")

    # ③ 轨迹回填：本轮压成一条 AdaptiveTurn dict 追加进 history
    report = GradingReport.model_validate(report_dict)
    gaps = [g.knowledge_gap for g in report.grades if not g.is_correct and g.knowledge_gap]
    try:
        mastery_after = await get_mastery(state.get("user_id", ""), state.get("document_id", ""))
    except Exception:
        mastery_after = None

    history = list(state.get("history", []))
    history.append({
        "turn": state.get("turn", len(history) + 1),
        "agent": "grader",
        "action": state.get("last_action", "continue"),
        "topic": state.get("description", "") or state.get("goal", ""),
        "difficulty_score": state.get("difficulty_score", 0.5),
        "score": report.score,
        "mastery_after": mastery_after,
        "knowledge_gaps": gaps,
    })

    # ④ 增量 consolidation（填 preferences 只读不写 + 画像卡 + 本机快照）——零 LLM，fail-soft
    #    传含本轮的 history（用于偏好趋势/盲点复现推断）+ 本轮题型（QuestionGrade 不带题型）。
    try:
        await consolidate_session_extras(
            state.get("user_id", ""), report, state.get("document_id", ""),
            history=history, question_type=state.get("type"),
        )
    except Exception as e:
        logger.warning(f"[grader_worker] consolidate_session_extras 失败（不中断闭环）: {e}")

    # ⑤ SRS 间隔重复复习调度（零 LLM，fail-soft）：
    #    本轮 supervisor 针对复习的点 = state["weak_points"]（开场可能是到期复习项）；
    #    未再错 → 间隔拉长、又错 → 重置；本轮新暴露的错点 → 加入调度（明天到期）。
    try:
        await update_after_session(
            state.get("user_id", ""), state.get("document_id"),
            reviewed_points=list(state.get("weak_points", []) or []),
            wrong_gaps=gaps,
        )
    except Exception as e:
        logger.warning(f"[grader_worker] SRS 调度更新失败（不中断闭环）: {e}")

    return {
        "grading_report": report_dict,
        "last_report": report_dict,
        "history": history,
        # 本轮 quiz 已批改完成：标记 graded，并重置质量门，让下一轮新 quiz 重新过门
        "quiz_served_graded": True,
        "critic_passed": False,
        "quiz_served": False,
        "critique_history": [],
        "revision_count": 0,
        "answers": [],
    }
