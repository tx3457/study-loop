"""Durable Adaptive learning-loop HTTP endpoints.

Every externally visible turn is backed by ``AdaptiveSessionStore``. The full
``QuizSession`` remains private inside the aggregate; HTTP responses project
only browser-safe question views.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Any

from chromadb.errors import ChromaError, NotFoundError
from fastapi import APIRouter, Header, HTTPException
from pydantic import ValidationError

from models.adaptive import AdaptiveTurn, NextStepDecision
from models.adaptive_session import (
    AdaptivePendingSubmit,
    AdaptiveQuestionFeedback,
    AdaptiveSessionAggregate,
    AdaptiveStartRequest,
    AdaptiveSubmitReceipt,
    AdaptiveSubmitRequest,
    AdaptiveTurnArtifact,
    AdaptiveTurnResponse,
)
from models.grader import GradingReport
from models.learning_path import LearningPath
from models.session import QuestionView, QuizSession
from services.adaptive_loop import decide_next_step, generate_lesson, should_terminate
from services.adaptive_sessions import (
    AdaptiveSessionAlreadyExistsError,
    AdaptiveSessionCapacityError,
    AdaptiveSessionCorruptError,
    AdaptiveSessionPayloadTooLargeError,
    AdaptiveSessionStartConflictError,
    StoredAdaptiveSession,
    adaptive_sessions,
)
from services.grader import grade_quiz_session
from services.idempotency import IdempotencyConflictError, normalize_idempotency_key
from services.injection import check_injection
from services.learning_path import generate_learning_path
from services.learning_path_store import (
    LearningPathCorruptError,
    LearningPathCreationConflictError,
    LearningPathPayloadTooLargeError,
    LearningPathRecord,
    learning_path_store,
)
from services.memory import (
    append_decision,
    commit_learning_memory,
    get_mastery,
    get_weak_points,
)
from services.quiz_sessions import QuizSessionApiError
from services.rag import generate_question
from services.session import InvalidQuizResponseError, validate_provider_questions
from services.vectorstore import ensure_document_available


router = APIRouter()
logger = logging.getLogger(__name__)
_ADAPTIVE_PATH_OPERATION = "adaptive_switch_to_plan_publish_v1"


def _not_found() -> QuizSessionApiError:
    return QuizSessionApiError(
        404,
        "adaptive_session_not_found",
        "自适应学习会话不存在",
        reason="missing",
    )


def _expired() -> QuizSessionApiError:
    return QuizSessionApiError(
        410,
        "adaptive_session_expired",
        "自适应学习会话已过期，请重新开始",
        reason="expired",
    )


def _busy() -> QuizSessionApiError:
    return QuizSessionApiError(
        409,
        "adaptive_session_busy",
        "自适应学习会话正在处理上一项操作，请稍后重试",
        reason="in_progress",
    )


def _stale() -> QuizSessionApiError:
    return QuizSessionApiError(
        409,
        "adaptive_session_stale",
        "自适应学习进度已变化，请刷新后继续",
        reason="stale",
    )


def _completed() -> QuizSessionApiError:
    return QuizSessionApiError(
        409,
        "adaptive_session_completed",
        "自适应学习会话已经结束",
        reason="completed",
    )


def _corrupt() -> QuizSessionApiError:
    return QuizSessionApiError(
        503,
        "adaptive_session_corrupt",
        "自适应学习会话数据无效，请重新开始",
        reason="corrupt",
    )


def _unavailable() -> QuizSessionApiError:
    return QuizSessionApiError(
        503,
        "adaptive_session_store_unavailable",
        "自适应学习会话存储暂时不可用",
    )


def _too_large() -> QuizSessionApiError:
    return QuizSessionApiError(
        413,
        "adaptive_session_too_large",
        "自适应学习会话内容过大，无法继续保存",
    )


def _path_too_large() -> QuizSessionApiError:
    return QuizSessionApiError(
        413,
        "adaptive_learning_path_too_large",
        "推荐学习路径内容过大，无法保存",
    )


def _path_corrupt() -> QuizSessionApiError:
    return QuizSessionApiError(
        503,
        "adaptive_learning_path_corrupt",
        "推荐学习路径数据无效，请重新开始",
        reason="corrupt",
    )


def _path_unavailable() -> QuizSessionApiError:
    return QuizSessionApiError(
        503,
        "adaptive_learning_path_store_unavailable",
        "推荐学习路径存储暂时不可用，请重试",
    )


def _canonical_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _request_hash(req: AdaptiveSubmitRequest) -> str:
    return _canonical_hash(req.model_dump(mode="json"))


def _key_hash(key: str | None) -> str | None:
    return hashlib.sha256(key.encode("utf-8")).hexdigest() if key else None


def _quiz_id(adaptive_session_id: str, turn: int) -> str:
    return f"adaptive:{adaptive_session_id}:turn:{turn}"


def _validate_aggregate(
    value: AdaptiveSessionAggregate | dict[str, Any],
) -> AdaptiveSessionAggregate:
    return AdaptiveSessionAggregate.validate_for_persistence(value)


def _validate_record(
    record: StoredAdaptiveSession | None,
) -> tuple[StoredAdaptiveSession, AdaptiveSessionAggregate]:
    if record is None:
        raise _not_found()
    if record.expired:
        raise _expired()
    try:
        aggregate = AdaptiveSessionAggregate.model_validate(
            record.aggregate.model_dump(mode="json")
        )
        aggregate = _validate_aggregate(aggregate)
    except (TypeError, ValueError, ValidationError) as exc:
        logger.error("durable adaptive session aggregate is invalid")
        raise _corrupt() from exc
    return record, aggregate


async def _inspect_live(
    session_id: str,
) -> tuple[StoredAdaptiveSession, AdaptiveSessionAggregate]:
    try:
        return _validate_record(await adaptive_sessions.inspect(session_id))
    except QuizSessionApiError:
        raise
    except AdaptiveSessionCorruptError as exc:
        logger.error("durable adaptive session envelope is invalid")
        raise _corrupt() from exc
    except Exception as exc:
        logger.error("durable adaptive session read failed: %s", type(exc).__name__)
        raise _unavailable() from exc


async def _find_start(
    key: str,
    request_payload: dict[str, Any],
) -> tuple[StoredAdaptiveSession, AdaptiveSessionAggregate] | None:
    try:
        record = await adaptive_sessions.find_start(key, request_payload)
        return None if record is None else _validate_record(record)
    except AdaptiveSessionStartConflictError as exc:
        raise IdempotencyConflictError(exc.reason) from exc
    except QuizSessionApiError:
        raise
    except AdaptiveSessionCorruptError as exc:
        raise _corrupt() from exc
    except Exception as exc:
        logger.error("durable adaptive start lookup failed: %s", type(exc).__name__)
        raise _unavailable() from exc


async def _claim_live(
    session_id: str,
) -> tuple[Any, StoredAdaptiveSession, AdaptiveSessionAggregate]:
    try:
        claim = await adaptive_sessions.claim(session_id, "adaptive_submit")
    except AdaptiveSessionCorruptError as exc:
        raise _corrupt() from exc
    except Exception as exc:
        logger.error("durable adaptive claim failed: %s", type(exc).__name__)
        raise _unavailable() from exc

    if not claim.claimed:
        if claim.reason == "missing":
            raise _not_found()
        if claim.reason == "expired":
            raise _expired()
        if claim.reason == "done":
            raise _completed()
        raise _busy()
    if claim.record is None or claim.token is None:
        raise _unavailable()
    record, aggregate = _validate_record(claim.record)
    return claim, record, aggregate


async def _release(session_id: str, token: str | None) -> None:
    if not token:
        return
    try:
        await adaptive_sessions.release(session_id, token)
    except Exception as exc:
        logger.error(
            "durable adaptive claim release failed: error_type=%s",
            type(exc).__name__,
        )


async def _checkpoint(
    record: StoredAdaptiveSession,
    token: str,
    aggregate: AdaptiveSessionAggregate,
) -> StoredAdaptiveSession:
    try:
        validated = _validate_aggregate(aggregate)
        updated = await adaptive_sessions.checkpoint(
            record.aggregate.adaptive_session_id,
            token,
            validated,
            expected_revision=record.revision,
        )
        if updated is None:
            raise _stale()
        _validate_record(updated)
        return updated
    except QuizSessionApiError:
        raise
    except AdaptiveSessionPayloadTooLargeError as exc:
        raise _too_large() from exc
    except AdaptiveSessionCorruptError as exc:
        raise _corrupt() from exc
    except (TypeError, ValueError, ValidationError) as exc:
        logger.error("adaptive checkpoint model validation failed")
        raise _corrupt() from exc
    except Exception as exc:
        logger.error("durable adaptive checkpoint failed: %s", type(exc).__name__)
        raise _unavailable() from exc


async def _complete_claim(
    record: StoredAdaptiveSession,
    token: str,
    aggregate: AdaptiveSessionAggregate,
) -> StoredAdaptiveSession:
    try:
        validated = _validate_aggregate(aggregate)
        updated = await adaptive_sessions.complete(
            record.aggregate.adaptive_session_id,
            token,
            validated,
            expected_revision=record.revision,
        )
        if updated is None:
            raise _stale()
        _validate_record(updated)
        return updated
    except QuizSessionApiError:
        raise
    except AdaptiveSessionPayloadTooLargeError as exc:
        raise _too_large() from exc
    except AdaptiveSessionCorruptError as exc:
        raise _corrupt() from exc
    except (TypeError, ValueError, ValidationError) as exc:
        logger.error("adaptive completion model validation failed")
        raise _corrupt() from exc
    except Exception as exc:
        logger.error("durable adaptive completion failed: %s", type(exc).__name__)
        raise _unavailable() from exc


def _turn_from_decision(turn: int, decision: NextStepDecision) -> AdaptiveTurn:
    return AdaptiveTurn(
        turn=turn,
        action=decision.action,
        topic=decision.topic,
        difficulty_score=decision.difficulty_score,
        reason=decision.reason,
    )


def _question_views(quiz: QuizSession) -> list[QuestionView]:
    return [
        QuestionView(
            index=index,
            question=question.question,
            options=question.options,
            type=question.type,
        )
        for index, question in enumerate(quiz.questions)
    ]


def _report_feedback(report: GradingReport | None) -> list[AdaptiveQuestionFeedback]:
    if report is None:
        return []
    return [
        AdaptiveQuestionFeedback(
            index=grade.index,
            question=grade.question,
            your_answer=grade.user_answer,
            correct_answer=grade.correct_answer,
            is_correct=grade.is_correct,
            ai_feedback=grade.ai_feedback,
            knowledge_gap=grade.knowledge_gap,
        )
        for grade in report.grades
    ]


def _report_gaps(report: GradingReport | None) -> list[str]:
    if report is None:
        return []
    return [
        grade.knowledge_gap
        for grade in report.grades
        if not grade.is_correct and grade.knowledge_gap
    ]


def _build_summary(
    history: list[AdaptiveTurn],
    mastery: float | None,
    term_reason: str,
    decision: NextStepDecision,
) -> str:
    reason_map = {
        "agent_finish": "agent 判定可结束",
        "mastery_reached": "掌握度达标",
        "max_turns": "达到最大轮次",
        "switch_to_plan": "转入系统学习路径",
    }
    trajectory = " → ".join(
        f"T{turn.turn}({turn.score:.2f})" if turn.score is not None else f"T{turn.turn}(-)"
        for turn in history
    )
    mastery_text = f"{mastery:.2f}" if mastery is not None else "未知"
    return (
        f"结束原因:{reason_map.get(term_reason, term_reason)}。"
        f"共 {len(history)} 轮,最终掌握度 {mastery_text}。"
        f"得分轨迹:{trajectory}。最后评估:{decision.reason}"
    )


def _artifact(
    *,
    session_id: str,
    turn: int,
    decision: NextStepDecision,
    trajectory: list[AdaptiveTurn],
    mastery: float | None,
    report: GradingReport | None = None,
    quiz: QuizSession | None = None,
    lesson: str | None = None,
    done: bool = False,
    terminate_reason: str = "",
    learning_path: dict[str, Any] | None = None,
) -> AdaptiveTurnArtifact:
    if done:
        turn_type = "quiz"
        questions: list[QuestionView] = []
        lesson = None
        summary = _build_summary(trajectory, mastery, terminate_reason, decision)
    else:
        turn_type = "teach" if lesson is not None else "quiz"
        questions = [] if quiz is None else _question_views(quiz)
        summary = ""
    return AdaptiveTurnArtifact(
        adaptive_session_id=session_id,
        turn=turn,
        done=done,
        turn_type=turn_type,
        questions=questions,
        lesson=lesson,
        decision=decision,
        last_report_score=report.score if report is not None else None,
        last_report_gaps=_report_gaps(report),
        last_report_feedback=_report_feedback(report),
        mastery=mastery,
        trajectory=trajectory,
        summary=summary,
        terminate_reason=terminate_reason,
        learning_path=learning_path,
    )


def _adaptive_path_binding(
    record: StoredAdaptiveSession,
    artifact: AdaptiveTurnArtifact,
) -> tuple[LearningPath, str, str] | None:
    if artifact.terminate_reason != "switch_to_plan":
        return None

    aggregate = record.aggregate
    try:
        if record.busy or aggregate.status != "completed":
            raise ValueError("adaptive path source is not a settled terminal record")
        canonical = aggregate.current_artifact
        if artifact.model_dump(mode="json") != canonical.model_dump(mode="json"):
            raise ValueError("adaptive path response is not the canonical artifact")
        path = LearningPath.model_validate(canonical.learning_path)
        if path.document_id != aggregate.document_id:
            raise ValueError("adaptive path document binding is inconsistent")
        fingerprint = _canonical_hash(
            {
                "operation": _ADAPTIVE_PATH_OPERATION,
                "adaptive_session_id": aggregate.adaptive_session_id,
                "user_id": aggregate.user_id,
                "document_id": aggregate.document_id,
                "goal": aggregate.goal,
                "source_created_at": record.updated_at,
                "learning_path": path.model_dump(mode="json"),
            }
        )
    except (TypeError, ValueError, ValidationError) as exc:
        logger.error(
            "adaptive learning path binding is invalid: error_type=%s",
            type(exc).__name__,
        )
        raise _path_corrupt() from exc

    creation_key = f"{_ADAPTIVE_PATH_OPERATION}:{aggregate.adaptive_session_id}"
    return path, creation_key, fingerprint


def _validate_published_path(
    published: LearningPathRecord,
    record: StoredAdaptiveSession,
    path: LearningPath,
) -> str:
    aggregate = record.aggregate
    try:
        if not isinstance(published, LearningPathRecord):
            raise TypeError("published learning path record is invalid")
        published_path = LearningPath.model_validate(published.path.model_dump(mode="json"))
        if (
            published.user_id != aggregate.user_id
            or published.document_id != aggregate.document_id
            or published.created_at != record.updated_at
            or published_path.model_dump(mode="json") != path.model_dump(mode="json")
        ):
            raise ValueError("published learning path binding is inconsistent")
        # Reuse the public resource model's path-id constraint without coupling
        # the Adaptive aggregate to the external resource identifier.
        if not (
            isinstance(published.path_id, str)
            and len(published.path_id) == 35
            and published.path_id.startswith("lp_")
            and all(char in "0123456789abcdef" for char in published.path_id[3:])
        ):
            raise ValueError("published learning path id is invalid")
    except (TypeError, ValueError, ValidationError) as exc:
        logger.error(
            "published adaptive learning path is invalid: error_type=%s",
            type(exc).__name__,
        )
        raise _path_corrupt() from exc
    return published.path_id


async def _publish_adaptive_learning_path(
    record: StoredAdaptiveSession,
    artifact: AdaptiveTurnArtifact,
) -> str | None:
    binding = _adaptive_path_binding(record, artifact)
    if binding is None:
        return None
    path, creation_key, fingerprint = binding
    aggregate = record.aggregate

    try:
        published = await learning_path_store.find_by_creation(
            creation_key,
            aggregate.user_id,
            aggregate.document_id,
            fingerprint,
        )
    except LearningPathCreationConflictError as exc:
        logger.error("adaptive learning path creation binding conflicts")
        raise _path_corrupt() from exc
    except LearningPathCorruptError as exc:
        logger.error("adaptive learning path record is corrupt")
        raise _path_corrupt() from exc
    except Exception as exc:
        logger.error(
            "adaptive learning path lookup failed: error_type=%s",
            type(exc).__name__,
        )
        raise _path_unavailable() from exc

    if published is None:
        create_error: Exception | None = None
        try:
            published = await learning_path_store.create(
                aggregate.user_id,
                aggregate.document_id,
                path,
                idempotency_key=creation_key,
                request_fingerprint=fingerprint,
                source_created_at=record.updated_at,
            )
        except LearningPathPayloadTooLargeError as exc:
            raise _path_too_large() from exc
        except LearningPathCreationConflictError as exc:
            logger.error("adaptive learning path creation binding conflicts")
            raise _path_corrupt() from exc
        except LearningPathCorruptError as exc:
            logger.error("adaptive learning path create found corrupt storage")
            raise _path_corrupt() from exc
        except Exception as exc:
            # A database commit can succeed even when its acknowledgement is
            # lost. Re-read the deterministic binding before reporting an
            # unavailable store so the canonical committed result wins.
            create_error = exc

        if published is None and create_error is not None:
            try:
                published = await learning_path_store.find_by_creation(
                    creation_key,
                    aggregate.user_id,
                    aggregate.document_id,
                    fingerprint,
                )
            except LearningPathCreationConflictError as exc:
                logger.error("adaptive learning path recovery binding conflicts")
                raise _path_corrupt() from exc
            except LearningPathCorruptError as exc:
                logger.error("adaptive learning path recovery found corrupt storage")
                raise _path_corrupt() from exc
            except Exception as recovery_error:
                logger.error(
                    "adaptive learning path recovery lookup failed: error_type=%s",
                    type(recovery_error).__name__,
                )
                raise _path_unavailable() from create_error
            if published is None:
                logger.error(
                    "adaptive learning path create failed without a durable receipt: error_type=%s",
                    type(create_error).__name__,
                )
                raise _path_unavailable() from create_error

    return _validate_published_path(published, record, path)


async def _response(
    record: StoredAdaptiveSession,
    artifact: AdaptiveTurnArtifact,
) -> AdaptiveTurnResponse:
    learning_path_id = await _publish_adaptive_learning_path(record, artifact)
    return AdaptiveTurnResponse(
        **artifact.model_dump(mode="json"),
        revision=record.revision,
        expires_at=record.expires_at,
        busy=record.busy,
        learning_path_id=learning_path_id,
    )


async def _generate_quiz(
    *,
    session_id: str,
    user_id: str,
    document_id: str,
    decision: NextStepDecision,
    turn: int,
) -> QuizSession:
    generated = await generate_question(
        document_id,
        decision.topic,
        decision.count,
        decision.difficulty,
        decision.question_type,
        difficulty_score=decision.difficulty_score,
        weak_points=decision.target_weak_points,
    )
    questions = getattr(generated, "questions", None)
    if not isinstance(questions, list):
        raise InvalidQuizResponseError("模型题目响应结构无效")
    validate_provider_questions(questions, decision.question_type)
    return QuizSession(
        session_id=_quiz_id(session_id, turn),
        document_id=document_id,
        user_id=user_id,
        questions=questions,
        user_answers=[],
        status="active",
    )


async def _generate_lesson(
    aggregate: AdaptiveSessionAggregate,
    decision: NextStepDecision,
) -> str:
    weak_points = decision.target_weak_points
    if not weak_points and aggregate.current_artifact.trajectory:
        weak_points = aggregate.current_artifact.trajectory[-1].knowledge_gaps
    return await generate_lesson(
        document_id=aggregate.document_id,
        topic=decision.topic,
        weak_points=weak_points or [],
        last_report=aggregate.last_report,
    )


async def _opening_state(
    req: AdaptiveStartRequest,
    session_id: str,
) -> AdaptiveSessionAggregate:
    mastery = await get_mastery(req.user_id, req.document_id)
    weak_points = await get_weak_points(req.user_id, req.document_id)
    decision = await decide_next_step(
        goal=req.goal,
        mastery=mastery,
        weak_points=weak_points,
        history=[],
        last_report=None,
    )
    history = [_turn_from_decision(1, decision)]

    if decision.action == "switch_to_plan":
        path = await generate_learning_path(req.document_id)
        artifact = _artifact(
            session_id=session_id,
            turn=1,
            decision=decision,
            trajectory=history,
            mastery=mastery,
            done=True,
            terminate_reason="switch_to_plan",
            learning_path=path.model_dump(),
        )
        return AdaptiveSessionAggregate(
            adaptive_session_id=session_id,
            user_id=req.user_id,
            document_id=req.document_id,
            goal=req.goal,
            status="completed",
            current_artifact=artifact,
        )

    terminate, reason = should_terminate(
        mastery=mastery,
        turn=1,
        decision=decision,
    )
    if terminate:
        artifact = _artifact(
            session_id=session_id,
            turn=1,
            decision=decision,
            trajectory=history,
            mastery=mastery,
            done=True,
            terminate_reason=reason,
        )
        return AdaptiveSessionAggregate(
            adaptive_session_id=session_id,
            user_id=req.user_id,
            document_id=req.document_id,
            goal=req.goal,
            status="completed",
            current_artifact=artifact,
        )

    if decision.action == "teach":
        lesson = await generate_lesson(
            document_id=req.document_id,
            topic=decision.topic,
            weak_points=decision.target_weak_points or weak_points,
            last_report=None,
        )
        artifact = _artifact(
            session_id=session_id,
            turn=1,
            decision=decision,
            trajectory=history,
            mastery=mastery,
            lesson=lesson,
        )
        return AdaptiveSessionAggregate(
            adaptive_session_id=session_id,
            user_id=req.user_id,
            document_id=req.document_id,
            goal=req.goal,
            current_artifact=artifact,
        )

    quiz = await _generate_quiz(
        session_id=session_id,
        user_id=req.user_id,
        document_id=req.document_id,
        decision=decision,
        turn=1,
    )
    artifact = _artifact(
        session_id=session_id,
        turn=1,
        decision=decision,
        trajectory=history,
        mastery=mastery,
        quiz=quiz,
    )
    return AdaptiveSessionAggregate(
        adaptive_session_id=session_id,
        user_id=req.user_id,
        document_id=req.document_id,
        goal=req.goal,
        current_quiz=quiz,
        current_artifact=artifact,
    )


async def _create_start(
    req: AdaptiveStartRequest,
    key: str | None,
    request_payload: dict[str, Any],
) -> tuple[StoredAdaptiveSession, AdaptiveSessionAggregate]:
    session_id = f"adapt_{uuid.uuid4().hex[:16]}"
    aggregate = _validate_aggregate(await _opening_state(req, session_id))
    try:
        result = await adaptive_sessions.create(
            aggregate,
            start_key=key,
            start_request=request_payload if key else None,
        )
        return _validate_record(result.record)
    except AdaptiveSessionStartConflictError as exc:
        raise IdempotencyConflictError(exc.reason) from exc
    except AdaptiveSessionPayloadTooLargeError as exc:
        raise _too_large() from exc
    except AdaptiveSessionCapacityError as exc:
        logger.warning("adaptive session capacity is full")
        raise _unavailable() from exc
    except AdaptiveSessionAlreadyExistsError as exc:
        logger.error("adaptive UUID collision")
        raise _unavailable() from exc
    except AdaptiveSessionCorruptError as exc:
        raise _corrupt() from exc
    except QuizSessionApiError:
        raise
    except Exception as exc:
        logger.error("durable adaptive create failed: %s", type(exc).__name__)
        raise _unavailable() from exc


@router.post("/agent/adaptive/start", response_model=AdaptiveTurnResponse)
async def adaptive_start(
    req: AdaptiveStartRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> AdaptiveTurnResponse:
    key = normalize_idempotency_key(idempotency_key)
    request_payload = req.model_dump(mode="json")
    if key:
        existing = await _find_start(key, request_payload)
        if existing is not None:
            record, aggregate = existing
            return await _response(record, aggregate.current_artifact)

    try:
        await ensure_document_available(req.document_id)
        injection, _reason = await check_injection(req.goal)
        if injection:
            logger.warning("adaptive start input safety check rejected")
            raise HTTPException(status_code=400, detail="输入安全检查未通过")
        record, aggregate = await _create_start(req, key, request_payload)
        return await _response(record, aggregate.current_artifact)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail="文档不存在") from exc
    except ChromaError as exc:
        logger.error(
            "adaptive start document lookup failed: error_type=%s",
            type(exc).__name__,
        )
        raise HTTPException(status_code=503, detail="文档存储暂时不可用") from exc
    except (InvalidQuizResponseError, ValidationError) as exc:
        logger.warning("adaptive provider returned invalid structured output")
        raise HTTPException(status_code=503, detail="模型返回的内容格式无效") from exc


@router.get("/agent/adaptive/{adaptive_session_id}", response_model=AdaptiveTurnResponse)
async def adaptive_snapshot(adaptive_session_id: str) -> AdaptiveTurnResponse:
    record, aggregate = await _inspect_live(adaptive_session_id)
    return await _response(record, aggregate.current_artifact)


def _find_receipt(
    aggregate: AdaptiveSessionAggregate,
    key_hash: str | None,
    request_hash: str,
) -> AdaptiveTurnArtifact | None:
    if key_hash is None:
        return None
    receipt = aggregate.submit_receipts.get(key_hash)
    if receipt is None:
        return None
    if receipt.request_hash != request_hash:
        raise IdempotencyConflictError("payload_mismatch")
    return receipt.response.model_copy(deep=True)


def _validate_submit_binding(
    req: AdaptiveSubmitRequest,
    aggregate: AdaptiveSessionAggregate,
    request_hash: str,
    key_hash: str | None,
    record: StoredAdaptiveSession,
) -> None:
    if aggregate.status == "completed":
        raise _completed()
    pending = aggregate.pending
    if pending is None:
        if req.turn != aggregate.current_artifact.turn or req.revision != record.revision:
            raise _stale()
        return
    if req.turn != pending.turn or req.revision != pending.revision:
        raise _stale()
    if pending.request_hash != request_hash:
        if pending.key_hash is not None and pending.key_hash == key_hash:
            raise IdempotencyConflictError("payload_mismatch")
        raise _busy()
    if pending.key_hash != key_hash:
        raise _busy()


async def _checkpoint_pending_input(
    req: AdaptiveSubmitRequest,
    aggregate: AdaptiveSessionAggregate,
    record: StoredAdaptiveSession,
    token: str,
    request_hash: str,
    key_hash: str | None,
) -> StoredAdaptiveSession:
    if aggregate.pending is not None:
        return record
    artifact = aggregate.current_artifact
    if artifact.turn_type == "teach":
        if req.answers:
            raise HTTPException(status_code=400, detail="讲解轮不接受题目答案")
    else:
        quiz = aggregate.current_quiz
        if quiz is None:
            raise _corrupt()
        if len(req.answers) != len(quiz.questions):
            raise HTTPException(
                status_code=400,
                detail=f"答案数 {len(req.answers)} 与题目数 {len(quiz.questions)} 不符",
            )
    aggregate.pending = AdaptivePendingSubmit(
        key_hash=key_hash,
        request_hash=request_hash,
        turn=req.turn,
        revision=req.revision,
        answers=list(req.answers),
    )
    return await _checkpoint(record, token, aggregate)


async def _checkpoint_quiz_answers(
    aggregate: AdaptiveSessionAggregate,
    record: StoredAdaptiveSession,
    token: str,
) -> StoredAdaptiveSession:
    quiz = aggregate.current_quiz
    pending = aggregate.pending
    if quiz is None or pending is None:
        raise _corrupt()
    if quiz.status == "active":
        quiz.user_answers = list(pending.answers)
        quiz.status = "completed"
        record = await _checkpoint(record, token, aggregate)
    return record


async def _grade_checkpointed_quiz(
    aggregate: AdaptiveSessionAggregate,
    record: StoredAdaptiveSession,
    token: str,
) -> tuple[StoredAdaptiveSession, GradingReport]:
    quiz = aggregate.current_quiz
    if quiz is None:
        raise _corrupt()
    if quiz.grading_report is None:

        async def checkpoint_grade() -> None:
            nonlocal record
            if quiz.grading_report is not None:
                aggregate.last_report = quiz.grading_report.model_copy(deep=True)
            record = await _checkpoint(record, token, aggregate)

        report = await grade_quiz_session(quiz, checkpoint=checkpoint_grade)
    else:
        report = quiz.grading_report.model_copy(deep=True)
    aggregate.last_report = report.model_copy(deep=True)
    record = await _checkpoint(record, token, aggregate)
    return record, report


async def _write_checkpointed_memory(
    aggregate: AdaptiveSessionAggregate,
    record: StoredAdaptiveSession,
    token: str,
    report: GradingReport,
) -> StoredAdaptiveSession:
    quiz = aggregate.current_quiz
    pending = aggregate.pending
    if quiz is None or pending is None:
        raise _corrupt()
    if quiz.profile_written:
        return record
    answered_decision = aggregate.current_artifact.decision
    decision_written = answered_decision is None

    async def record_decision() -> None:
        nonlocal decision_written
        if answered_decision is None:
            return
        await append_decision(
            aggregate.user_id,
            {
                "decision_id": (
                    f"adaptive:{aggregate.adaptive_session_id}:turn:{pending.turn}:decision"
                ),
                "agent": "adaptive_loop",
                "decision": answered_decision.action,
                "rationale": answered_decision.reason,
                "turn": pending.turn,
                "session_id": report.session_id,
                "score": report.score,
            },
        )
        decision_written = True

    await commit_learning_memory(
        aggregate.user_id,
        report,
        aggregate.document_id,
        questions=quiz.questions,
        after_write=record_decision,
        on_core_written=lambda: setattr(quiz, "profile_written", True),
    )
    # ``commit_learning_memory`` deliberately treats its audit callback as
    # fail-soft. Adaptive cannot checkpoint the shared memory marker until the
    # deterministic decision event has also succeeded, otherwise a transient
    # audit failure would be skipped forever on retry.
    if not decision_written:
        raise RuntimeError("adaptive decision audit was not written")
    return await _checkpoint(record, token, aggregate)


async def _checkpoint_mastery_and_decision(
    aggregate: AdaptiveSessionAggregate,
    record: StoredAdaptiveSession,
    token: str,
    *,
    previous_was_teach: bool,
) -> tuple[StoredAdaptiveSession, NextStepDecision]:
    pending = aggregate.pending
    if pending is None:
        raise _corrupt()
    if pending.next_decision is not None:
        return record, pending.next_decision

    mastery = await get_mastery(aggregate.user_id, aggregate.document_id)
    pending.mastery = mastery
    if not previous_was_teach:
        report = aggregate.last_report
        if report is None:
            raise _corrupt()
        history = [item.model_copy(deep=True) for item in aggregate.current_artifact.trajectory]
        current = history[-1]
        current.score = report.score
        current.mastery_after = mastery
        current.knowledge_gaps = _report_gaps(report)
        aggregate.current_artifact.trajectory = history
    record = await _checkpoint(record, token, aggregate)

    weak_points = await get_weak_points(aggregate.user_id, aggregate.document_id)
    decision = await decide_next_step(
        goal=aggregate.goal,
        mastery=mastery,
        weak_points=weak_points,
        history=aggregate.current_artifact.trajectory,
        last_report=aggregate.last_report,
        allow_teach=not previous_was_teach,
    )
    pending.next_decision = decision
    record = await _checkpoint(record, token, aggregate)
    return record, decision


async def _next_artifact(
    aggregate: AdaptiveSessionAggregate,
    decision: NextStepDecision,
) -> tuple[AdaptiveTurnArtifact, QuizSession | None, str]:
    pending = aggregate.pending
    if pending is None:
        raise _corrupt()
    mastery = pending.mastery
    history = [item.model_copy(deep=True) for item in aggregate.current_artifact.trajectory]
    report = aggregate.last_report

    # Explicit planning takes precedence over generic mastery/max-turn exits.
    if decision.action == "switch_to_plan":
        path = await generate_learning_path(aggregate.document_id)
        return (
            _artifact(
                session_id=aggregate.adaptive_session_id,
                turn=pending.turn,
                decision=decision,
                trajectory=history,
                mastery=mastery,
                report=report,
                done=True,
                terminate_reason="switch_to_plan",
                learning_path=path.model_dump(),
            ),
            aggregate.current_quiz,
            "completed",
        )

    terminate, reason = should_terminate(
        mastery=mastery,
        turn=pending.turn,
        decision=decision,
    )
    if terminate:
        return (
            _artifact(
                session_id=aggregate.adaptive_session_id,
                turn=pending.turn,
                decision=decision,
                trajectory=history,
                mastery=mastery,
                report=report,
                done=True,
                terminate_reason=reason,
            ),
            aggregate.current_quiz,
            "completed",
        )

    next_turn = pending.turn + 1
    history.append(_turn_from_decision(next_turn, decision))
    if decision.action == "teach":
        lesson = await _generate_lesson(aggregate, decision)
        artifact = _artifact(
            session_id=aggregate.adaptive_session_id,
            turn=next_turn,
            decision=decision,
            trajectory=history,
            mastery=mastery,
            report=report,
            lesson=lesson,
        )
        return artifact, None, "active"

    quiz = await _generate_quiz(
        session_id=aggregate.adaptive_session_id,
        user_id=aggregate.user_id,
        document_id=aggregate.document_id,
        decision=decision,
        turn=next_turn,
    )
    artifact = _artifact(
        session_id=aggregate.adaptive_session_id,
        turn=next_turn,
        decision=decision,
        trajectory=history,
        mastery=mastery,
        report=report,
        quiz=quiz,
    )
    return artifact, quiz, "active"


async def _execute_claimed_submit(
    req: AdaptiveSubmitRequest,
    key_hash: str | None,
    request_hash: str,
    record: StoredAdaptiveSession,
    aggregate: AdaptiveSessionAggregate,
    token: str,
) -> tuple[StoredAdaptiveSession, AdaptiveTurnArtifact]:
    _validate_submit_binding(req, aggregate, request_hash, key_hash, record)
    previous_was_teach = aggregate.current_artifact.turn_type == "teach"
    record = await _checkpoint_pending_input(
        req,
        aggregate,
        record,
        token,
        request_hash,
        key_hash,
    )

    if not previous_was_teach:
        record = await _checkpoint_quiz_answers(aggregate, record, token)
        record, report = await _grade_checkpointed_quiz(aggregate, record, token)
        record = await _write_checkpointed_memory(
            aggregate,
            record,
            token,
            report,
        )

    record, decision = await _checkpoint_mastery_and_decision(
        aggregate,
        record,
        token,
        previous_was_teach=previous_was_teach,
    )
    artifact, current_quiz, status = await _next_artifact(aggregate, decision)

    pending = aggregate.pending
    if pending is None:
        raise _corrupt()
    aggregate.current_artifact = artifact
    aggregate.current_quiz = current_quiz
    aggregate.status = status
    aggregate.pending = None
    if key_hash is not None:
        aggregate.submit_receipts[key_hash] = AdaptiveSubmitReceipt(
            key_hash=key_hash,
            request_hash=request_hash,
            turn=pending.turn,
            revision=pending.revision,
            response=artifact,
        )

    completed = await _complete_claim(record, token, aggregate)
    committed = _validate_aggregate(completed.aggregate)
    if key_hash is not None:
        return completed, committed.submit_receipts[key_hash].response.model_copy(deep=True)
    return completed, artifact


@router.post("/agent/adaptive/submit", response_model=AdaptiveTurnResponse)
async def adaptive_submit(
    req: AdaptiveSubmitRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> AdaptiveTurnResponse:
    key = normalize_idempotency_key(idempotency_key)
    request_hash = _request_hash(req)
    key_hash = _key_hash(key)

    inspected_record, inspected = await _inspect_live(req.adaptive_session_id)
    replay = _find_receipt(inspected, key_hash, request_hash)
    if replay is not None:
        return await _response(inspected_record, replay)
    _validate_submit_binding(
        req,
        inspected,
        request_hash,
        key_hash,
        inspected_record,
    )

    try:
        claim, record, aggregate = await _claim_live(req.adaptive_session_id)
    except QuizSessionApiError as exc:
        # A worker can complete after our optimistic inspect but before our
        # claim. Re-read once so a response-loss retry with the same key still
        # receives its durable receipt instead of a misleading completed error.
        if exc.code == "adaptive_session_completed" and key_hash is not None:
            raced_record, raced_aggregate = await _inspect_live(req.adaptive_session_id)
            raced_replay = _find_receipt(
                raced_aggregate,
                key_hash,
                request_hash,
            )
            if raced_replay is not None:
                return await _response(raced_record, raced_replay)
        raise
    token = claim.token
    completed = False
    try:
        replay = _find_receipt(aggregate, key_hash, request_hash)
        if replay is not None:
            await _release(req.adaptive_session_id, token)
            completed = True
            settled_record, settled_aggregate = await _inspect_live(req.adaptive_session_id)
            settled_replay = _find_receipt(
                settled_aggregate,
                key_hash,
                request_hash,
            )
            if settled_replay is None:
                raise _corrupt()
            return await _response(settled_record, settled_replay)
        final_record, artifact = await _execute_claimed_submit(
            req,
            key_hash,
            request_hash,
            record,
            aggregate,
            token,
        )
        completed = True
        return await _response(final_record, artifact)
    except (InvalidQuizResponseError, ValidationError) as exc:
        logger.warning("adaptive provider returned invalid structured output")
        raise HTTPException(
            status_code=503,
            detail="模型返回的内容格式无效",
        ) from exc
    finally:
        if not completed:
            await _release(req.adaptive_session_id, token)
