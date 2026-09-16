import logging

from fastapi import Depends, APIRouter, Header, HTTPException
from pydantic import BaseModel

from models.session import QuestionView, QuizSessionAggregate
from models.wrong_questions import WrongQuestionBank
from services.auth import require_user_id
from services.idempotency import IdempotencyConflictError, normalize_idempotency_key
from services.quiz_sessions import (
    QuizSessionAlreadyExistsError,
    QuizSessionApiError,
    QuizSessionCapacityError,
    QuizSessionCorruptError,
    QuizSessionPayloadTooLargeError,
    QuizSessionStartConflictError,
    quiz_sessions,
)
from services.session import question_views
from services.wrong_questions import (
    get_wrong_questions,
    prepare_repractice_session,
)


router = APIRouter(prefix="/wrong-questions")
logger = logging.getLogger(__name__)


class RepracticeResponse(BaseModel):
    session_id: str
    total: int
    questions: list[QuestionView]
    revision: int
    expires_at: float


@router.get("/{document_id}", response_model=WrongQuestionBank)
async def list_wrong_questions(
    document_id: str,
    user_id: str = Depends(require_user_id),
):
    # user_id 曾经是 query 参数：换个值就能读别人的错题本。现在只来自可信身份。
    return await get_wrong_questions(document_id, user_id=user_id)


def _response(record) -> RepracticeResponse:
    if record.expired:
        raise QuizSessionApiError(
            410,
            "quiz_session_expired",
            "错题重练会话已过期，请重新开始",
            reason="expired",
        )
    session = record.aggregate.session
    return RepracticeResponse(
        session_id=session.session_id,
        total=len(session.questions),
        questions=question_views(session),
        revision=record.revision,
        expires_at=record.expires_at,
    )


@router.post("/{document_id}/practice", response_model=RepracticeResponse)
async def repractice(
    document_id: str,
    user_id: str = Depends(require_user_id),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    key = normalize_idempotency_key(idempotency_key)
    request_payload = {
        "origin": "wrong_question",
        "document_id": document_id,
        "user_id": user_id,
    }
    if key:
        try:
            existing = await quiz_sessions.find_start(key, request_payload)
        except QuizSessionStartConflictError as exc:
            raise IdempotencyConflictError(exc.reason) from exc
        except QuizSessionCorruptError as exc:
            raise QuizSessionApiError(
                503,
                "quiz_session_corrupt",
                "错题重练会话数据无效，请重新开始",
            ) from exc
        except Exception as exc:
            logger.error(
                "durable wrong-practice start lookup failed: %s",
                type(exc).__name__,
            )
            raise QuizSessionApiError(
                503,
                "quiz_session_store_unavailable",
                "错题重练会话存储暂时不可用",
            ) from exc
        if existing is not None:
            return _response(existing)

    try:
        session = await prepare_repractice_session(document_id, user_id=user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="该文档暂无可重练错题") from exc

    aggregate = QuizSessionAggregate(origin="wrong_question", session=session)
    try:
        decision = await quiz_sessions.create(
            aggregate,
            start_key=key,
            start_request=request_payload if key else None,
        )
        return _response(decision.record)
    except QuizSessionPayloadTooLargeError as exc:
        raise QuizSessionApiError(
            413,
            "quiz_session_too_large",
            "错题数量过多，请先完成部分错题后再试",
        ) from exc
    except QuizSessionCapacityError as exc:
        raise QuizSessionApiError(
            503,
            "quiz_session_capacity",
            "当前进行中的答题会话过多，请稍后重试",
        ) from exc
    except QuizSessionStartConflictError as exc:
        raise IdempotencyConflictError(exc.reason) from exc
    except QuizSessionAlreadyExistsError as exc:
        logger.error("generated duplicate wrong-practice session identifier")
        raise QuizSessionApiError(
            503,
            "quiz_session_store_unavailable",
            "错题重练会话暂时无法创建",
        ) from exc
    except QuizSessionCorruptError as exc:
        raise QuizSessionApiError(
            503,
            "quiz_session_corrupt",
            "错题重练会话数据无效，请重新开始",
        ) from exc
    except Exception as exc:
        logger.error(
            "durable wrong-practice session create failed: %s",
            type(exc).__name__,
        )
        raise QuizSessionApiError(
            503,
            "quiz_session_store_unavailable",
            "错题重练会话存储暂时不可用",
        ) from exc
