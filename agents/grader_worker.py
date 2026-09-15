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
import asyncio
import logging
from weakref import WeakValueDictionary

from agents.adapt_agent import adapt_writer
from agents.grader_agent import grader_agent
from agents.state import TutorState
from models.grader import GradingReport
from services.memory import consolidate_session_extras, get_mastery
from services.session import sessions
from services.srs import update_after_session
from services.tutor_sessions import restore_completed_tutor_session

logger = logging.getLogger(__name__)
_postprocess_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()


async def grader_worker(state: TutorState) -> dict:
    """批改 worker：grader_agent → adapt_writer 画像写回 → 轨迹回填 last_report/history。

    设计：grader_agent 读 state["session_id"] 批改（answers 已在上游写入该 session），
    产出 grading_report。随后调 adapt_writer 做画像 EMA 写回（复用其 decision_log 审计），
    最后把本轮压成 AdaptiveTurn dict 追加进 history，并写 last_report 供 supervisor 决策。

    重置质量门标记（critic_passed/quiz_served/critique_history/revision_count），
    使 supervisor 决定下一轮 quiz 时重新走 critic→reviser↔critic→await_answers 全门。
    """
    # ① 批改（grader_agent 复用，写 grading_report）。如果进程在答案 checkpoint
    # 落盘后、grader 开始前崩溃，按持久 state 重建进程内 completed session。
    state_session_id = state.get("session_id")
    if (
        state_session_id
        and state.get("quiz") is not None
        and "answers" in state
    ):
        restore_completed_tutor_session(state)

    grade_result = await grader_agent.ainvoke(state)
    report_dict = grade_result.get("grading_report")
    if not report_dict:
        logger.warning("[grader_worker] grader_agent 未产出 grading_report")
        return {}

    report = GradingReport.model_validate(report_dict)
    if state_session_id and report.session_id != state_session_id:
        raise RuntimeError("Tutor grader returned a report for a different session")
    session = sessions.get(report.session_id)
    if session is None:
        raise ValueError(f"Session {report.session_id} not found for tutor grading")

    user_id = session.user_id
    document_id = session.document_id
    ownership_matches = (
        state.get("user_id") in {None, user_id}
        and state.get("document_id") in {None, document_id}
    )
    if not ownership_matches:
        logger.warning(
            "[grader_worker] ignored mismatched state ownership for session %s",
            report.session_id,
        )

    question_types = {question.type for question in session.questions}
    question_type = next(iter(question_types)) if len(question_types) == 1 else None
    writer_state = {
        **state,
        "user_id": user_id,
        "document_id": document_id,
        "type": question_type,
        "grading_report": report_dict,
    }
    postprocess_lock = _postprocess_locks.setdefault(report.session_id, asyncio.Lock())

    async with postprocess_lock:
        # ② 画像 EMA 写回（adapt_writer 复用：write_episodic_memory + update_semantic_memory）
        if not session.profile_written:
            try:
                await adapt_writer.ainvoke(writer_state)
            except Exception as e:
                logger.warning(
                    "[grader_worker] adapt_writer 画像写回失败（不中断闭环）: "
                    "error_type=%s",
                    type(e).__name__,
                )

        # ③ 轨迹回填：本轮压成一条 AdaptiveTurn dict 追加进 history
        gaps = [g.knowledge_gap for g in report.grades if not g.is_correct and g.knowledge_gap]
        try:
            mastery_after = await get_mastery(user_id, document_id)
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

        # ④ 增量 consolidation（填 preferences + 画像卡 + 本机快照），按会话幂等。
        if not session.extras_written:
            try:
                await consolidate_session_extras(
                    user_id,
                    report,
                    document_id,
                    history=history,
                    question_type=question_type,
                    session_id=report.session_id,
                )
                session.extras_written = True
            except Exception as e:
                logger.warning(
                    "[grader_worker] consolidate_session_extras 失败（不中断闭环）: "
                    "error_type=%s",
                    type(e).__name__,
                )

        # ⑤ SRS 调度按会话幂等；身份错配时不采用调用方携带的旧复习点。
        if not session.review_schedule_written:
            try:
                await update_after_session(
                    user_id,
                    document_id,
                    reviewed_points=(
                        list(state.get("weak_points", []) or [])
                        if ownership_matches
                        else []
                    ),
                    wrong_gaps=gaps,
                    session_id=report.session_id,
                )
                session.review_schedule_written = True
            except Exception as e:
                logger.warning(
                    "[grader_worker] SRS 调度更新失败（不中断闭环）: error_type=%s",
                    type(e).__name__,
                )

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
        # 一轮"讲→练→批"已经走完，解除 tutor_node 置的讲解禁令。
        # 不复位的话 allow_teach 一旦被置 False 就再没人改回来，
        # 整个会话只能讲一次——语义要的是"上一步刚讲过"，不是"讲过一次就永久禁讲"。
        "allow_teach": True,
    }
