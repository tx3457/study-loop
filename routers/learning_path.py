import hashlib
import json
import logging
import re

from chromadb.errors import ChromaError, NotFoundError
from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, ValidationError

from models.learning_path import (
    CreateLearningPathRequest,
    LearningPath,
    LearningPathResource,
)
from services.idempotency import IdempotencyConflictError, normalize_idempotency_key
from services.learning_path import (
    LearningPathEvidenceUnavailableError,
    generate_learning_path,
)
from services.learning_path_store import (
    LearningPathCorruptError,
    LearningPathCreationConflictError,
    LearningPathPayloadTooLargeError,
    LearningPathRecord,
    learning_path_store,
)

router = APIRouter()
logger = logging.getLogger(__name__)

_RESOURCE_OPERATION = "web_learning_path_create_v1"
_PATH_ID_PATTERN = re.compile(r"^lp_[0-9a-f]{32}$")


def _request_fingerprint(request: CreateLearningPathRequest) -> str:
    canonical = json.dumps(
        {
            "operation": _RESOURCE_OPERATION,
            "payload": request.model_dump(mode="json"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resource(record: LearningPathRecord) -> LearningPathResource:
    return LearningPathResource(
        schema_version=record.schema_version,
        learning_path_id=record.path_id,
        user_id=record.user_id,
        path=record.path,
        created_at=record.created_at,
        expires_at=None,
    )


def _store_unavailable() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="学习路径存储暂时不可用",
    )


async def _find_created_path(
    key: str,
    request: CreateLearningPathRequest,
    fingerprint: str,
) -> LearningPathRecord | None:
    try:
        return await learning_path_store.find_by_creation(
            key,
            request.user_id,
            request.document_id,
            fingerprint,
        )
    except LearningPathCreationConflictError as exc:
        raise IdempotencyConflictError(exc.reason) from exc
    except LearningPathCorruptError as exc:
        raise _store_unavailable() from exc
    except Exception as exc:
        logger.error("learning path lookup failed: %s", type(exc).__name__)
        raise _store_unavailable() from exc


@router.post("/learning-path/{document_id}", response_model=LearningPath)
async def create_learning_path(document_id: str):
    try:
        generated = await generate_learning_path(document_id)
        payload = generated.model_dump(mode="python") if isinstance(
            generated, BaseModel
        ) else generated
        return LearningPath.model_validate(payload)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail="文档不存在") from exc
    except LearningPathEvidenceUnavailableError as exc:
        raise HTTPException(
            status_code=422,
            detail="文档没有可用于生成学习路径的内容",
        ) from exc
    except ValidationError as exc:
        logger.warning("learning path provider returned invalid structured output")
        raise HTTPException(
            status_code=503,
            detail="模型返回的学习路径格式无效",
        ) from exc
    except ChromaError as exc:
        logger.exception("learning path document lookup failed")
        raise HTTPException(status_code=503, detail="文档存储暂时不可用") from exc


@router.post("/learning-paths", response_model=LearningPathResource)
async def create_learning_path_resource(
    request: CreateLearningPathRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> LearningPathResource:
    """创建或重放一个不可变、可跨刷新恢复的学习路径资源。"""
    try:
        key = normalize_idempotency_key(idempotency_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if key is None:
        raise HTTPException(status_code=400, detail="缺少 Idempotency-Key")

    fingerprint = _request_fingerprint(request)
    existing = await _find_created_path(key, request, fingerprint)
    if existing is not None:
        return _resource(existing)

    try:
        generated = await generate_learning_path(request.document_id)
        payload = generated.model_dump(mode="python") if isinstance(
            generated, BaseModel
        ) else generated
        path = LearningPath.model_validate(payload)
        try:
            record = await learning_path_store.create(
                request.user_id,
                request.document_id,
                path,
                idempotency_key=key,
                request_fingerprint=fingerprint,
            )
        except LearningPathCreationConflictError as exc:
            raise IdempotencyConflictError(exc.reason) from exc
        except LearningPathPayloadTooLargeError as exc:
            raise HTTPException(
                status_code=413,
                detail="学习路径内容超过持久化上限",
            ) from exc
        except LearningPathCorruptError as exc:
            raise _store_unavailable() from exc
        except Exception as exc:
            logger.error("learning path create failed: %s", type(exc).__name__)
            raise _store_unavailable() from exc
        return _resource(record)
    except NotFoundError as exc:
        replay = await _find_created_path(key, request, fingerprint)
        if replay is not None:
            return _resource(replay)
        raise HTTPException(status_code=404, detail="文档不存在") from exc
    except LearningPathEvidenceUnavailableError as exc:
        replay = await _find_created_path(key, request, fingerprint)
        if replay is not None:
            return _resource(replay)
        raise HTTPException(
            status_code=422,
            detail="文档没有可用于生成学习路径的内容",
        ) from exc
    except ValidationError as exc:
        replay = await _find_created_path(key, request, fingerprint)
        if replay is not None:
            return _resource(replay)
        logger.warning("learning path provider returned invalid structured output")
        raise HTTPException(
            status_code=503,
            detail="模型返回的学习路径格式无效",
        ) from exc
    except ChromaError as exc:
        replay = await _find_created_path(key, request, fingerprint)
        if replay is not None:
            return _resource(replay)
        logger.exception("learning path document lookup failed")
        raise HTTPException(status_code=503, detail="文档存储暂时不可用") from exc
    except Exception:
        # The provider may fail after another request has already committed the
        # same idempotent creation. That canonical result remains authoritative
        # for every ordinary provider failure. CancelledError is a BaseException
        # and intentionally bypasses this recovery lookup.
        replay = await _find_created_path(key, request, fingerprint)
        if replay is not None:
            return _resource(replay)
        raise


@router.get(
    "/learning-paths/current",
    response_model=LearningPathResource | None,
)
async def get_current_learning_path_resource(
    document_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=512,
        pattern=r".*\S.*",
    ),
) -> LearningPathResource | None:
    """读取默认 Web 用户最近创建的路径，供新标签页发现。"""
    try:
        record = await learning_path_store.get_current(
            "default_user",
            document_id,
        )
    except (LearningPathCorruptError, ValueError) as exc:
        raise _store_unavailable() from exc
    except Exception as exc:
        logger.error("current learning path read failed: %s", type(exc).__name__)
        raise _store_unavailable() from exc
    return None if record is None else _resource(record)


@router.get(
    "/learning-paths/{learning_path_id}",
    response_model=LearningPathResource,
)
async def get_learning_path_resource(
    learning_path_id: str,
) -> LearningPathResource:
    """读取持久路径；不依赖原材料是否仍保留。"""
    if not _PATH_ID_PATTERN.fullmatch(learning_path_id):
        raise HTTPException(status_code=404, detail="学习路径不存在")
    try:
        record = await learning_path_store.get(learning_path_id)
    except (LearningPathCorruptError, ValueError) as exc:
        raise _store_unavailable() from exc
    except Exception as exc:
        logger.error("learning path read failed: %s", type(exc).__name__)
        raise _store_unavailable() from exc
    if record is None:
        raise HTTPException(status_code=404, detail="学习路径不存在")
    return _resource(record)
