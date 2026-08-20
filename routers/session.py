import asyncio
import hashlib
import json
import logging

from chromadb.errors import ChromaError, NotFoundError
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ValidationError

from models.grader import GradingReport
from models.report import LearningReport
from models.session import (
    AnswerRequest,
    AnswerResult,
    QuestionView,
    QuizSessionAggregate,
    SessionResult,
    SessionSnapshot,
    SessionStartRequest,
)
from services.grader import grade_quiz_session
from services.idempotency import (
    IdempotencyConflictError,
    normalize_idempotency_key,
)
from services.memory import commit_learning_memory
from services.quiz_sessions import (
    QuizSessionAlreadyExistsError,
    QuizSessionApiError,
    QuizSessionCapacityError,
    QuizSessionCorruptError,
    QuizSessionPayloadTooLargeError,
    QuizSessionStartConflictError,
    StoredQuizSession,
    quiz_sessions,
)
from services.report import generate_report_for_quiz
from services.session import (
    InvalidQuizResponseError,
    SessionConflictError,
    answer_result_for_index,
    apply_answer_to_session,
    build_result,
    prepare_session,
    question_views,
    write_objective_profile,
)


router = APIRouter(prefix="/session")
logger = logging.getLogger(__name__)


class StartSessionResponse(BaseModel):
    session_id: str
    total: int
    questions: list[QuestionView]
    revision: int
    expires_at: float


class GradingResponse(GradingReport):
    revision: int
    expires_at: float


class LearningReportResponse(LearningReport):
    revision: int
    expires_at: float


def _not_found() -> QuizSessionApiError:
    return QuizSessionApiError(
        404,
        "quiz_session_not_found",
        "答题会话不存在",
        reason="missing",
    )


def _expired() -> QuizSessionApiError:
    return QuizSessionApiError(
        410,
        "quiz_session_expired",
        "答题会话已过期，请重新开始",
        reason="expired",
    )


def _busy() -> QuizSessionApiError:
    return QuizSessionApiError(
        409,
        "quiz_session_busy",
        "答题会话正在处理上一项操作，请稍后重试",
        reason="in_progress",
    )


def _stale() -> QuizSessionApiError:
    return QuizSessionApiError(
        409,
        "quiz_session_stale",
        "答题进度已变化，请刷新后继续",
        reason="stale",
    )


def _require_live(record: StoredQuizSession | None) -> StoredQuizSession:
    if record is None:
        raise _not_found()
    if record.expired:
        raise _expired()
    return record


async def _inspect_live(session_id: str) -> StoredQuizSession:
    try:
        return _require_live(await quiz_sessions.inspect(session_id))
    except QuizSessionApiError:
        raise
    except QuizSessionCorruptError as exc:
        logger.error("durable quiz session payload is invalid: %s", type(exc).__name__)
        raise QuizSessionApiError(
            503,
            "quiz_session_corrupt",
            "答题会话数据无效，请重新开始",
        ) from exc
    except Exception as exc:
        logger.error("durable quiz session read failed: %s", type(exc).__name__)
        raise QuizSessionApiError(
            503,
            "quiz_session_store_unavailable",
            "答题会话存储暂时不可用",
        ) from exc


async def _claim_live(session_id: str, operation: str):
    try:
        claim = await quiz_sessions.claim(session_id, operation)
    except QuizSessionCorruptError as exc:
        logger.error("durable quiz session claim found invalid payload")
        raise QuizSessionApiError(
            503,
            "quiz_session_corrupt",
            "答题会话数据无效，请重新开始",
        ) from exc
    except Exception as exc:
        logger.error("durable quiz session claim failed: %s", type(exc).__name__)
        raise QuizSessionApiError(
            503,
            "quiz_session_store_unavailable",
            "答题会话存储暂时不可用",
        ) from exc

    if claim.claimed:
        return claim
    if claim.reason == "missing":
        raise _not_found()
    if claim.reason == "expired":
        raise _expired()
    raise _busy()


def _start_response(record: StoredQuizSession) -> StartSessionResponse:
    session = record.aggregate.session
    return StartSessionResponse(
        session_id=session.session_id,
        total=len(session.questions),
        questions=question_views(session),
        revision=record.revision,
        expires_at=record.expires_at,
    )


def _answer_response(record: StoredQuizSession, index: int) -> AnswerResult:
    response = answer_result_for_index(record.aggregate.session, index)
    response.revision = record.revision
    response.expires_at = record.expires_at
    return response


def _matching_committed_answer(
    record: StoredQuizSession,
    req: AnswerRequest,
) -> AnswerResult | None:
    answers = record.aggregate.session.user_answers
    if req.question_index >= len(answers) or answers[req.question_index] != req.answer:
        return None
    return _answer_response(record, req.question_index)


def _answer_request_hashes(
    key: str,
    session_id: str,
    req: AnswerRequest,
) -> tuple[str, str]:
    key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    canonical = json.dumps(
        {
            "operation": "session.answer.v3",
            "session_id": session_id,
            **req.model_dump(mode="json"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return key_hash, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _replay_bound_answer(
    record: StoredQuizSession,
    key_hash: str,
    request_hash: str,
    req: AnswerRequest,
) -> AnswerResult | None:
    stored_request_hash = record.aggregate.answer_request_hashes.get(key_hash)
    if stored_request_hash is None:
        return None
    if stored_request_hash != request_hash:
        raise IdempotencyConflictError("payload_mismatch")
    committed = _matching_committed_answer(record, req)
    if committed is None:
        raise QuizSessionApiError(
            503,
            "quiz_session_corrupt",
            "答题会话的重试记录无效，请重新开始",
        )
    return committed


def _snapshot(record: StoredQuizSession) -> SessionSnapshot:
    aggregate = record.aggregate
    session = aggregate.session
    last_answer_result = (
        aggregate.last_answer_result.model_copy(deep=True)
        if aggregate.last_answer_result is not None
        else None
    )
    if last_answer_result is not None:
        last_answer_result.revision = record.revision
        last_answer_result.expires_at = record.expires_at

    result = None
    if session.status == "completed":
        result = build_result(session)
        result.revision = record.revision
        result.expires_at = record.expires_at

    return SessionSnapshot(
        origin=aggregate.origin,
        session_id=session.session_id,
        document_id=session.document_id,
        revision=record.revision,
        status=session.status,
        total=len(session.questions),
        answered_count=len(session.user_answers),
        questions=question_views(session),
        last_answer_index=aggregate.last_answer_index,
        last_user_answer=(
            session.user_answers[aggregate.last_answer_index]
            if aggregate.last_answer_index is not None
            else None
        ),
        last_answer_result=last_answer_result,
        result=result,
        grading_report=(
            session.grading_report.model_copy(deep=True)
            if session.grading_report is not None
            else None
        ),
        learning_report=(
            session.learning_report.model_copy(deep=True)
            if session.learning_report is not None
            else None
        ),
        expires_at=record.expires_at,
        busy=record.busy,
    )


async def _release_claim(session_id: str, token: str | None) -> None:
    if not token:
        return
    try:
        await quiz_sessions.release(session_id, token)
    except Exception:
        logger.error("durable quiz session claim release failed")


def _objective_memory_pending(record: StoredQuizSession) -> bool:
    session = record.aggregate.session
    return (
        session.status == "completed"
        and not session.profile_written
        and all(question.type != "short_answer" for question in session.questions)
    )


async def _persist_objective_memory(
    session_id: str,
    record: StoredQuizSession,
) -> StoredQuizSession:
    """Best-effort repair of the durable objective-memory outbox marker."""
    if not _objective_memory_pending(record):
        return record

    claim = None
    try:
        claim = await quiz_sessions.claim(session_id, "objective_memory")
        if not claim.claimed or claim.record is None or claim.token is None:
            return record
        aggregate = claim.record.aggregate.model_copy(deep=True)
        session = aggregate.session
        if not _objective_memory_pending(claim.record):
            await quiz_sessions.release(session_id, claim.token)
            refreshed = await quiz_sessions.inspect(session_id)
            return refreshed or record

        if session.grading_report is not None:
            await commit_learning_memory(
                session.user_id,
                session.grading_report,
                session.document_id,
                questions=session.questions,
                on_core_written=lambda: setattr(session, "profile_written", True),
            )
        else:
            await write_objective_profile(session, session_id)
        if not session.profile_written:
            raise RuntimeError("objective memory writer did not publish its marker")
        completed = await quiz_sessions.complete(session_id, claim.token, aggregate)
        if completed is None:
            logger.warning("objective memory marker lost its quiz session claim")
            await _release_claim(session_id, claim.token)
            return record
        return completed
    except BaseException as exc:
        if claim is not None and claim.claimed:
            await _release_claim(session_id, claim.token)
        if isinstance(exc, asyncio.CancelledError):
            raise
        logger.warning(
            "objective quiz memory write failed without blocking answer: %s",
            type(exc).__name__,
        )
        return record


async def _checkpoint_claim(
    session_id: str,
    token: str,
    aggregate: QuizSessionAggregate,
) -> StoredQuizSession:
    try:
        record = await quiz_sessions.checkpoint(session_id, token, aggregate)
    except QuizSessionPayloadTooLargeError as exc:
        raise QuizSessionApiError(
            413,
            "quiz_session_too_large",
            "答题会话内容过大，无法继续保存",
        ) from exc
    except QuizSessionCorruptError as exc:
        raise QuizSessionApiError(
            503,
            "quiz_session_corrupt",
            "答题会话数据无效，请重新开始",
        ) from exc
    except Exception as exc:
        logger.error("durable quiz checkpoint failed: %s", type(exc).__name__)
        raise QuizSessionApiError(
            503,
            "quiz_session_store_unavailable",
            "答题会话存储暂时不可用",
        ) from exc
    if record is None:
        raise _stale()
    return record


async def _complete_claim(
    session_id: str,
    token: str,
    aggregate: QuizSessionAggregate,
) -> StoredQuizSession:
    try:
        record = await quiz_sessions.complete(session_id, token, aggregate)
    except QuizSessionPayloadTooLargeError as exc:
        raise QuizSessionApiError(
            413,
            "quiz_session_too_large",
            "答题会话内容过大，无法继续保存",
        ) from exc
    except QuizSessionCorruptError as exc:
        raise QuizSessionApiError(
            503,
            "quiz_session_corrupt",
            "答题会话数据无效，请重新开始",
        ) from exc
    except Exception as exc:
        logger.error("durable quiz commit failed: %s", type(exc).__name__)
        raise QuizSessionApiError(
            503,
            "quiz_session_store_unavailable",
            "答题会话存储暂时不可用",
        ) from exc
    if record is None:
        raise _stale()
    return record


async def _grade_and_write_memory(
    session_id: str,
    token: str,
    aggregate: QuizSessionAggregate,
) -> GradingReport:
    session = aggregate.session

    async def checkpoint() -> None:
        await _checkpoint_claim(session_id, token, aggregate)

    grading = await grade_quiz_session(session, checkpoint=checkpoint)
    if not session.profile_written:
        await commit_learning_memory(
            session.user_id,
            grading,
            session.document_id,
            questions=session.questions,
            on_core_written=lambda: setattr(session, "profile_written", True),
        )
        await checkpoint()
    return grading


@router.post("/start", response_model=StartSessionResponse)
async def start(
    req: SessionStartRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    key = normalize_idempotency_key(idempotency_key)
    request_payload = req.model_dump(mode="json")
    if key:
        try:
            existing = await quiz_sessions.find_start(key, request_payload)
        except QuizSessionStartConflictError as exc:
            raise IdempotencyConflictError(exc.reason) from exc
        except QuizSessionCorruptError as exc:
            raise QuizSessionApiError(
                503,
                "quiz_session_corrupt",
                "答题会话数据无效，请重新开始",
            ) from exc
        except Exception as exc:
            logger.error("durable quiz start lookup failed: %s", type(exc).__name__)
            raise QuizSessionApiError(
                503,
                "quiz_session_store_unavailable",
                "答题会话存储暂时不可用",
            ) from exc
        if existing is not None:
            return _start_response(_require_live(existing))

    try:
        session = await prepare_session(req)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail="文档不存在") from exc
    except ChromaError as exc:
        logger.error("session start document lookup failed")
        raise HTTPException(status_code=503, detail="文档存储暂时不可用") from exc
    except (InvalidQuizResponseError, ValidationError) as exc:
        logger.warning(
            "quiz provider returned invalid structured output: %s", type(exc).__name__
        )
        raise HTTPException(status_code=503, detail="模型返回的题目格式无效") from exc

    aggregate = QuizSessionAggregate(session=session)
    try:
        decision = await quiz_sessions.create(
            aggregate,
            start_key=key,
            start_request=request_payload if key else None,
        )
        return _start_response(_require_live(decision.record))
    except QuizSessionStartConflictError as exc:
        raise IdempotencyConflictError(exc.reason) from exc
    except QuizSessionPayloadTooLargeError as exc:
        raise QuizSessionApiError(
            413,
            "quiz_session_too_large",
            "本次题目内容过大，请减少题目数量",
        ) from exc
    except QuizSessionCapacityError as exc:
        raise QuizSessionApiError(
            503,
            "quiz_session_capacity",
            "当前进行中的答题会话过多，请稍后重试",
        ) from exc
    except QuizSessionAlreadyExistsError as exc:
        logger.error("generated duplicate quiz session identifier")
        raise QuizSessionApiError(
            503,
            "quiz_session_store_unavailable",
            "答题会话暂时无法创建",
        ) from exc
    except QuizSessionCorruptError as exc:
        raise QuizSessionApiError(
            503,
            "quiz_session_corrupt",
            "答题会话数据无效，请重新开始",
        ) from exc
    except Exception as exc:
        logger.error("durable quiz session create failed: %s", type(exc).__name__)
        raise QuizSessionApiError(
            503,
            "quiz_session_store_unavailable",
            "答题会话存储暂时不可用",
        ) from exc


@router.get("/{session_id}", response_model=SessionSnapshot)
async def snapshot(session_id: str):
    record = await _inspect_live(session_id)
    record = await _persist_objective_memory(session_id, record)
    return _snapshot(record)


@router.post(
    "/{session_id}/answer",
    response_model=AnswerResult,
    response_model_exclude_none=True,
)
async def answer(
    session_id: str,
    req: AnswerRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    current = await _inspect_live(session_id)
    key = normalize_idempotency_key(idempotency_key)
    key_hash = request_hash = None
    if key:
        key_hash, request_hash = _answer_request_hashes(key, session_id, req)
        replay = _replay_bound_answer(current, key_hash, request_hash, req)
        if replay is not None:
            current = await _persist_objective_memory(session_id, current)
            return _replay_bound_answer(current, key_hash, request_hash, req) or replay

    claim = None
    try:
        claim = await _claim_live(session_id, "answer")
        aggregate = claim.record.aggregate.model_copy(deep=True)
        if key_hash is not None and request_hash is not None:
            replay = _replay_bound_answer(
                claim.record,
                key_hash,
                request_hash,
                req,
            )
            if replay is not None:
                replay_record = claim.record
                await _release_claim(session_id, claim.token)
                claim = None
                repaired = await _persist_objective_memory(session_id, replay_record)
                return (
                    _replay_bound_answer(repaired, key_hash, request_hash, req)
                    or replay
                )
        response = await apply_answer_to_session(
            aggregate.session,
            req.answer,
            question_index=req.question_index,
        )
        aggregate.last_answer_index = req.question_index
        aggregate.last_answer_result = response.model_copy(deep=True)
        if key_hash is not None and request_hash is not None:
            aggregate.answer_request_hashes[key_hash] = request_hash
        committed = await _complete_claim(session_id, claim.token, aggregate)

        if response.is_last and not any(
            question.type == "short_answer" for question in aggregate.session.questions
        ):
            committed = await _persist_objective_memory(session_id, committed)

        response = _answer_response(committed, req.question_index)
        return response
    except BaseException as exc:
        if claim is not None:
            await _release_claim(session_id, claim.token)
        if isinstance(exc, asyncio.CancelledError):
            raise
        if isinstance(exc, SessionConflictError):
            raise QuizSessionApiError(
                409,
                "quiz_session_stale",
                str(exc),
                reason="stale",
            ) from exc
        raise


@router.get("/{session_id}/result", response_model=SessionResult)
async def result(session_id: str):
    record = await _inspect_live(session_id)
    record = await _persist_objective_memory(session_id, record)
    try:
        response = build_result(record.aggregate.session)
    except SessionConflictError as exc:
        raise QuizSessionApiError(
            409,
            "quiz_session_not_completed",
            "答题会话尚未完成",
            reason="not_completed",
        ) from exc
    response.revision = record.revision
    response.expires_at = record.expires_at
    return response


@router.post("/{session_id}/grade", response_model=GradingResponse)
async def grade(session_id: str):
    claim = await _claim_live(session_id, "grade")
    aggregate = claim.record.aggregate.model_copy(deep=True)
    if aggregate.session.status != "completed":
        await _release_claim(session_id, claim.token)
        raise QuizSessionApiError(
            409,
            "quiz_session_not_completed",
            "答题会话尚未完成",
            reason="not_completed",
        )
    try:
        grading = await _grade_and_write_memory(session_id, claim.token, aggregate)
        record = await _complete_claim(session_id, claim.token, aggregate)
        return GradingResponse(
            **grading.model_dump(mode="json"),
            revision=record.revision,
            expires_at=record.expires_at,
        )
    except ValidationError as exc:
        await _release_claim(session_id, claim.token)
        logger.warning("grader provider returned invalid structured output")
        raise HTTPException(status_code=503, detail="模型返回的批改格式无效") from exc
    except ValueError as exc:
        await _release_claim(session_id, claim.token)
        raise QuizSessionApiError(
            409,
            "quiz_session_stale",
            str(exc),
            reason="stale",
        ) from exc
    except BaseException:
        await _release_claim(session_id, claim.token)
        raise


@router.post("/{session_id}/report", response_model=LearningReportResponse)
async def report(session_id: str):
    claim = await _claim_live(session_id, "report")
    aggregate = claim.record.aggregate.model_copy(deep=True)
    if aggregate.session.status != "completed":
        await _release_claim(session_id, claim.token)
        raise QuizSessionApiError(
            409,
            "quiz_session_not_completed",
            "答题会话尚未完成",
            reason="not_completed",
        )
    try:
        grading = await _grade_and_write_memory(session_id, claim.token, aggregate)
        learning_report = await generate_report_for_quiz(aggregate.session, grading)
        record = await _complete_claim(session_id, claim.token, aggregate)
        return LearningReportResponse(
            **learning_report.model_dump(mode="json"),
            revision=record.revision,
            expires_at=record.expires_at,
        )
    except ValidationError as exc:
        await _release_claim(session_id, claim.token)
        logger.warning("report provider returned invalid structured output")
        raise HTTPException(status_code=503, detail="模型返回的报告格式无效") from exc
    except ValueError as exc:
        await _release_claim(session_id, claim.token)
        raise QuizSessionApiError(
            409,
            "quiz_session_stale",
            str(exc),
            reason="stale",
        ) from exc
    except BaseException:
        await _release_claim(session_id, claim.token)
        raise
