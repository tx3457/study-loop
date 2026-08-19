import logging
import re
import uuid
from collections.abc import Awaitable, Callable

from models.grader import GradingReport, QuestionGrade
from services.memory import (
    get_user_profile,
    update_semantic_memory,
    write_episodic_memory,
)
from models.quiz import Question
from models.session import (
    AnswerResult,
    QuestionView,
    QuizSession,
    SessionResult,
    SessionStartRequest,
)
from services.rag import generate_question
from services.vectorstore import ensure_document_available as _ensure_document_available

logger = logging.getLogger(__name__)

# 内存会话存储（服务重启后清空）
sessions: dict[str, QuizSession] = {}

# 文字难度 → 连续值（无用户画像时的兜底）
_DIFFICULTY_SCORE = {"easy": 0.2, "medium": 0.5, "hard": 0.8}

_OPTION_PREFIX = re.compile(r"^\s*([A-Z])(?:\s*[.．、:：)）]\s*|\s+)(.+?)\s*$", re.I)


class InvalidQuizResponseError(RuntimeError):
    """The provider returned no usable questions."""


class SessionNotFoundError(ValueError):
    """The requested quiz session does not exist."""


class SessionConflictError(ValueError):
    """The requested operation conflicts with the current session state."""


def _normalized_text(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _option_label(value: str, options: list[str]) -> str | None:
    """Resolve a label from ``C``, ``C. text``, or an exact option body."""
    normalized = _normalized_text(value)
    valid_labels = {chr(ord("a") + index) for index in range(len(options))}
    if normalized in valid_labels:
        return normalized

    prefixed = _OPTION_PREFIX.match(value)
    if prefixed and prefixed.group(1).casefold() in valid_labels:
        return prefixed.group(1).casefold()

    for index, option in enumerate(options):
        fallback_label = chr(ord("a") + index)
        option_match = _OPTION_PREFIX.match(option)
        option_label = (
            option_match.group(1).casefold() if option_match else fallback_label
        )
        if normalized == _normalized_text(option):
            return option_label
        if option_match and normalized == _normalized_text(option_match.group(2)):
            return option_label
    return None


def answers_match(question: Question, user_answer: str) -> bool:
    """Compare answers consistently across feedback, memory, and final results."""
    if _normalized_text(user_answer) == _normalized_text(question.answer):
        return True
    if question.type == "short_answer" or not question.options:
        return False
    expected_label = _option_label(question.answer, question.options)
    submitted_label = _option_label(user_answer, question.options)
    return expected_label is not None and expected_label == submitted_label


def validate_provider_questions(questions: list[Question], question_type: str) -> None:
    if not questions:
        raise InvalidQuizResponseError("模型未返回题目")
    for question in questions:
        # The requested mode is authoritative; providers may omit this optional
        # field and otherwise inherit Question's "choice" default.
        question.type = question_type
        if not question.question.strip() or not question.answer.strip():
            raise InvalidQuizResponseError("模型返回了空题目或空答案")
        if question_type in {"choice", "true_false"}:
            if not question.options:
                raise InvalidQuizResponseError("模型返回的客观题缺少选项")
            if _option_label(question.answer, question.options) is None:
                raise InvalidQuizResponseError("模型返回的答案不属于任何选项")


async def start_session(req: SessionStartRequest) -> tuple[str, list[QuestionView]]:
    await _ensure_document_available(req.document_id)

    # 自适应：读取用户画像，计算本次出题参数
    profile = await get_user_profile(req.user_id)
    if profile and req.document_id in profile.get("topic_mastery", {}):
        # 有该文档的历史成绩：难度 = 掌握度 + 0.15（略高于当前水平，触发学习区间）
        mastery = profile["topic_mastery"][req.document_id]
        difficulty_score = round(min(mastery + 0.15, 1.0), 2)
        weak_points = profile.get("weak_points", [])
    else:
        # 新用户或新文档：将文字难度转换为连续值，无薄弱知识点
        difficulty_score = _DIFFICULTY_SCORE.get(req.difficulty, 0.5)
        weak_points = []

    quiz_response = await generate_question(
        req.document_id,
        req.description,
        req.count,
        req.difficulty,
        req.type,
        difficulty_score=difficulty_score,
        weak_points=weak_points,
    )
    questions = getattr(quiz_response, "questions", None)
    if not isinstance(questions, list):
        raise InvalidQuizResponseError("模型题目响应结构无效")
    validate_provider_questions(questions, req.type)

    session_id = str(uuid.uuid4())
    session = QuizSession(
        session_id=session_id,
        document_id=req.document_id,
        user_id=req.user_id,
        questions=questions,
        user_answers=[],
        status="active",
    )
    sessions[session_id] = session

    questions_view = [
        QuestionView(index=i, question=q.question, options=q.options, type=q.type)
        for i, q in enumerate(session.questions)
    ]
    return session_id, questions_view


async def submit_answer(
    session_id: str,
    answer: str,
    *,
    question_index: int | None = None,
    before_commit: Callable[[], Awaitable[None]] | None = None,
) -> AnswerResult:
    session = sessions.get(session_id)
    if not session:
        raise SessionNotFoundError(f"Session {session_id} not found")
    if session.status != "active":
        raise SessionConflictError("Session is not accepting answers")

    current_index = len(session.user_answers)
    if current_index >= len(session.questions):
        raise SessionConflictError("All questions already answered")
    if question_index is not None and question_index != current_index:
        raise SessionConflictError("Question index no longer matches session state")

    question = session.questions[current_index]
    if before_commit is not None:
        await before_commit()
    session.user_answers.append(answer)

    correct = answers_match(question, answer)

    is_last = len(session.user_answers) == len(session.questions)
    if is_last:
        session.status = "completed"
        # 画像写回统一使用 memory bank API；失败时不影响答题结果返回。
        if not any(question.type == "short_answer" for question in session.questions):
            try:
                await _write_back_profile(session, session_id)
                session.profile_written = True
            except Exception as e:
                logger.warning(f"[session] 画像写回失败（不影响答题结果）: {e}")

    return AnswerResult(
        correct=correct,
        correct_answer=question.answer,
        explanation=question.explanation,
        is_last=is_last,
        next_index=current_index + 1 if not is_last else None,
    )


async def _write_back_profile(session: QuizSession, session_id: str) -> None:
    """会话完成后把成绩写回画像（session_briefs / error_log / mastery EMA / weak_points）。

    与 adapt_writer 走同一套 write_episodic_memory + update_semantic_memory，
    保证两条轨道（/session/* 与 /agent/stream grade）的记忆口径一致。
    本轨道为字符串比较批改，无 LLM 提炼的 knowledge_gap，错题以题干截断兜底。
    """
    grades = []
    correct_count = 0
    for i, (q, ua) in enumerate(zip(session.questions, session.user_answers)):
        is_correct = answers_match(q, ua)
        correct_count += is_correct
        grades.append(
            QuestionGrade(
                index=i,
                question=q.question,
                user_answer=ua,
                correct_answer=q.answer,
                is_correct=is_correct,
                ai_feedback=None if is_correct else q.explanation,
                knowledge_gap=None if is_correct else q.question[:40],
            )
        )
    total = len(session.questions)
    report = GradingReport(
        session_id=session_id,
        total=total,
        correct=correct_count,
        score=round(correct_count / total, 2) if total else 0.0,
        grades=grades,
    )
    await write_episodic_memory(
        session.user_id,
        report,
        session.document_id,
        questions=session.questions,
    )
    await update_semantic_memory(session.user_id, report, session.document_id)


async def get_result(session_id: str) -> SessionResult:
    session = sessions.get(session_id)
    if not session:
        raise SessionNotFoundError(f"Session {session_id} not found")
    if session.status != "completed":
        raise SessionConflictError("Session not completed")

    details = []
    correct_count = 0
    for i, (q, user_ans) in enumerate(zip(session.questions, session.user_answers)):
        is_correct = answers_match(q, user_ans)
        if is_correct:
            correct_count += 1
        details.append(
            {
                "index": i,
                "question": q.question,
                "user_answer": user_ans,
                "correct_answer": q.answer,
                "correct": is_correct,
                "explanation": q.explanation,
            }
        )

    total = len(session.questions)
    return SessionResult(
        session_id=session_id,
        document_id=session.document_id,
        total=total,
        correct=correct_count,
        score=round(correct_count / total, 2) if total > 0 else 0.0,
        details=details,
    )
