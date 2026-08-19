import hashlib
import logging
import uuid

from pydantic import ValidationError

from models.quiz import Question
from models.session import QuestionView, QuizSession
from models.wrong_questions import WrongEntry, WrongQuestionBank
from services.memory import list_errors

logger = logging.getLogger(__name__)


def _get_sessions():
    """延迟导入，避免 services.session 与错题服务循环导入。"""
    from services.session import sessions

    return sessions


def _legacy_entry_id(item: dict) -> str:
    """为升级前没有 error_id 的记录生成稳定标识。"""
    identity = "\x1f".join(
        str(item.get(field, ""))
        for field in ("session_id", "question_index", "question", "correct_answer")
    )
    return f"legacy:{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]}"


def _to_wrong_entry(item: dict, document_id: str) -> WrongEntry:
    options = item.get("options")
    question_type = item.get("question_type") or (
        "choice" if options else "short_answer"
    )
    return WrongEntry(
        entry_id=item.get("error_id") or _legacy_entry_id(item),
        document_id=document_id,
        question=item.get("question", ""),
        options=options,
        question_type=question_type,
        correct_answer=item.get("correct_answer", ""),
        explanation=item.get("explanation") or item.get("knowledge_gap") or "",
        user_answer=item.get("user_answer", ""),
        knowledge_gap=item.get("knowledge_gap"),
        session_id=item.get("session_id", "unknown"),
    )


async def get_wrong_questions(
    document_id: str,
    user_id: str = "default_user",
) -> WrongQuestionBank:
    """从持久学习记忆读取错题；无效的历史记录会被跳过而不拖垮整页。"""
    items = await list_errors(user_id, document_id=document_id, limit=200)
    entries: list[WrongEntry] = []
    for item in items:
        try:
            entry = _to_wrong_entry(item, document_id)
        except ValidationError:
            logger.warning("忽略无法解析的错题记录: %s", item.get("error_id"))
            continue
        if not entry.question or not entry.correct_answer:
            logger.warning("忽略缺少题干或答案的错题记录: %s", entry.entry_id)
            continue
        entries.append(entry)

    return WrongQuestionBank(
        document_id=document_id,
        total=len(entries),
        entries=entries,
    )


async def start_repractice(
    document_id: str,
    user_id: str = "default_user",
) -> tuple[str, list[QuestionView]]:
    """Legacy in-memory wrapper used by direct service callers."""
    sessions = _get_sessions()
    session = await prepare_repractice_session(document_id, user_id=user_id)
    sessions[session.session_id] = session
    return session.session_id, _question_views(session)


def _question_views(session: QuizSession) -> list[QuestionView]:
    return [
        QuestionView(
            index=index,
            question=question.question,
            options=question.options,
            type=question.type,
        )
        for index, question in enumerate(session.questions)
    ]


async def prepare_repractice_session(
    document_id: str,
    user_id: str = "default_user",
) -> QuizSession:
    """Build a wrong-question QuizSession without choosing its storage backend."""
    bank = await get_wrong_questions(document_id, user_id=user_id)
    if not bank.entries:
        raise ValueError(f"No wrong questions for document '{document_id}'")

    questions = [
        Question(
            question=entry.question,
            options=entry.options,
            answer=entry.correct_answer,
            explanation=entry.explanation,
            source=f"wrong-question:{entry.entry_id}",
            type=entry.question_type,
        )
        for entry in bank.entries
    ]

    return QuizSession(
        session_id=str(uuid.uuid4()),
        document_id=document_id,
        user_id=user_id,
        questions=questions,
        user_answers=[],
        status="active",
    )
