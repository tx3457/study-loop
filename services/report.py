import asyncio
from pathlib import Path

from dotenv import load_dotenv

from models.grader import GradingReport
from models.report import LearningReport, _ReportCore
from models.session import QuizSession
from services.session import sessions
from services.grader import grade_session
from services.llm import llm_parse, structured_client as client, structured_model as model

load_dotenv(Path(__file__).parent.parent / ".env")

_report_locks: dict[str, asyncio.Lock] = {}

REPORT_SYSTEM_PROMPT = """
你是学习评估专家。根据学生的答题记录，生成一份结构化学习评估报告。

要求：
- 从题目内容中识别所有涉及的知识点/主题（通常 3-6 个）
- 按知识点统计 question_count 和 correct_count，计算 mastery_pct（0-100）
- strengths：mastery_pct ≥ 70% 的知识点名称列表
- weaknesses：mastery_pct < 70% 的知识点名称列表
- recommendations：针对薄弱点给出 3-5 条具体可行的学习建议
- summary：100字以内的总体评语，指出本次学习的主要收获和改进方向
"""


async def _generate_report_uncached(
    session: QuizSession,
    grading: GradingReport,
) -> LearningReport:
    if session.status != "completed":
        raise ValueError("Session not completed yet — submit all answers first")

    # 仅使用上游已确定的权威批改结果构建报告。
    records = []
    for g in grading.grades:
        status = "✓" if g.is_correct else "✗"
        record = (
            f"{g.index + 1}. [{status}] 题目：{g.question} | "
            f"学生答案：{g.user_answer} | 参考答案：{g.correct_answer}"
        )
        if g.knowledge_gap:
            record += f" | 知识盲点：{g.knowledge_gap}"
        records.append(record)
    records_text = "\n".join(records)

    response = await llm_parse(
        messages=[
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": f"""
学生答题记录（共 {grading.total} 题，答对 {grading.correct} 题）：

{records_text}

请生成学习评估报告。
"""},
        ],
        response_format=_ReportCore,
        client=client,
        model=model,
    )
    core = response.choices[0].message.parsed

    return LearningReport(
        session_id=session.session_id,
        document_id=session.document_id,
        overall_score=grading.score,
        topic_mastery=core.topic_mastery,
        strengths=core.strengths,
        weaknesses=core.weaknesses,
        recommendations=core.recommendations,
        summary=core.summary,
    )


async def generate_report_for_quiz(
    session: QuizSession,
    grading: GradingReport,
) -> LearningReport:
    """Generate or return the canonical report for an explicitly owned session."""
    if session.status != "completed":
        raise ValueError("Session not completed yet — submit all answers first")
    if session.grading_report is None:
        raise ValueError("Session does not contain a canonical grading report")
    if grading.model_dump() != session.grading_report.model_dump():
        raise ValueError("Grading report does not match the session cache")
    if session.learning_report is not None:
        return session.learning_report.model_copy(deep=True)

    report = await _generate_report_uncached(session, grading)
    session.learning_report = report.model_copy(deep=True)
    return report


async def generate_report(
    session_id: str,
    grading: GradingReport | None = None,
) -> LearningReport:
    """Return the one canonical report for an immutable completed session."""
    session = sessions.get(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")
    if session.status != "completed":
        raise ValueError("Session not completed yet — submit all answers first")

    lock = _report_locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        session = sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        if session.status != "completed":
            raise ValueError("Session not completed yet — submit all answers first")
        canonical_grading = await grade_session(session_id)
        if grading is not None and grading.model_dump() != canonical_grading.model_dump():
            raise ValueError("Grading report does not match the session cache")
        if session.learning_report is not None:
            return session.learning_report.model_copy(deep=True)

        return await generate_report_for_quiz(session, canonical_grading)
