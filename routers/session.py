import asyncio
import logging

from chromadb.errors import ChromaError, NotFoundError
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ValidationError

from models.session import (
    SessionStartRequest,
    AnswerRequest,
    AnswerResult,
    QuizSession,
    SessionResult,
    QuestionView,
)
from models.grader import GradingReport
from models.report import LearningReport
from services.session import (
    InvalidQuizResponseError,
    SessionConflictError,
    SessionNotFoundError,
    get_result,
    start_session,
    submit_answer,
)
from services.grader import grade_session
from services.report import generate_report
from services.memory import (
    commit_learning_memory,
)
from services.idempotency import (
    abort_idempotency_claim,
    normalize_idempotency_key,
    request_idempotency,
)
from services.session import sessions
from services.tool_registry import SideEffectAmbiguousError

router = APIRouter(prefix="/session")
logger = logging.getLogger(__name__)
_answer_locks: dict[str, asyncio.Lock] = {}


class StartSessionResponse(BaseModel):
    session_id: str
    total: int
    questions: list[QuestionView]


@router.post("/start", response_model=StartSessionResponse)
async def start(req: SessionStartRequest):
    try:
        session_id, questions = await start_session(req)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail="文档不存在") from exc
    except ChromaError as exc:
        logger.exception("session start document lookup failed")
        raise HTTPException(status_code=503, detail="文档存储暂时不可用") from exc
    except (InvalidQuizResponseError, ValidationError) as exc:
        logger.warning(
            "quiz provider returned invalid structured output: %s", type(exc).__name__
        )
        raise HTTPException(status_code=503, detail="模型返回的题目格式无效") from exc
    return StartSessionResponse(
        session_id=session_id, total=len(questions), questions=questions
    )


@router.post("/{session_id}/answer", response_model=AnswerResult)
async def answer(
    session_id: str,
    req: AnswerRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    key = normalize_idempotency_key(idempotency_key)
    claimed = False
    if key:
        decision = await request_idempotency.begin(
            key,
            "session.answer",
            {"session_id": session_id, **req.model_dump(mode="json")},
        )
        if decision.replayed:
            return AnswerResult.model_validate(decision.response)
        claimed = True

    effect_started = False

    async def mark_effect() -> None:
        nonlocal effect_started
        if effect_started:
            return
        if key:
            await request_idempotency.mark_effect_started(key, "session_answer")
        effect_started = True

    try:
        if session_id not in sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found",
            )

        lock = _answer_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            response = await submit_answer(
                session_id,
                req.answer,
                question_index=req.question_index,
                before_commit=mark_effect,
            )
            if key:
                await request_idempotency.complete(
                    key, response.model_dump(mode="json")
                )
            return response
    except BaseException as exc:
        durable_effect = False
        if claimed and key:
            try:
                durable_effect = await abort_idempotency_claim(
                    request_idempotency,
                    key,
                )
            except BaseException:
                logger.exception("session answer receipt cleanup failed")

        ambiguous = effect_started or durable_effect
        if ambiguous:
            session = sessions.get(session_id)
            if session is not None:
                session.status = "ambiguous"

        if isinstance(exc, asyncio.CancelledError):
            raise
        if ambiguous and not isinstance(exc, SideEffectAmbiguousError):
            raise SideEffectAmbiguousError("session_answer") from exc
        if isinstance(exc, SessionNotFoundError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if isinstance(exc, SessionConflictError):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise


@router.get("/{session_id}/result", response_model=SessionResult)
async def result(session_id: str):
    try:
        return await get_result(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SessionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _require_completed_session(session_id: str):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    if session.status != "completed":
        raise HTTPException(status_code=409, detail="Session not completed")
    return session


async def _ensure_grading_and_memory(
    session_id: str,
    session: QuizSession,
) -> GradingReport:
    """Create grading once, then make its learner-memory effect retryable."""
    report = await grade_session(session_id)
    if not session.profile_written:
        await commit_learning_memory(
            session.user_id,
            report,
            session.document_id,
            questions=session.questions,
            on_core_written=lambda: setattr(session, "profile_written", True),
        )
    return report


@router.post("/{session_id}/grade", response_model=GradingReport)
async def grade(session_id: str):
    _require_completed_session(session_id)
    lock = _answer_locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        session = _require_completed_session(session_id)
        try:
            return await _ensure_grading_and_memory(session_id, session)
        except ValidationError as exc:
            logger.warning("grader provider returned invalid structured output")
            raise HTTPException(
                status_code=503,
                detail="模型返回的批改格式无效",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{session_id}/report", response_model=LearningReport)
async def report(session_id: str):
    _require_completed_session(session_id)
    lock = _answer_locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        session = _require_completed_session(session_id)
        try:
            grading = await _ensure_grading_and_memory(session_id, session)
            return await generate_report(session_id, grading=grading)
        except ValidationError as exc:
            logger.warning("report provider returned invalid structured output")
            raise HTTPException(
                status_code=503,
                detail="模型返回的报告格式无效",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
