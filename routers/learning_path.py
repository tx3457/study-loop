import logging

from chromadb.errors import ChromaError, NotFoundError
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

from models.learning_path import LearningPath
from services.learning_path import (
    LearningPathEvidenceUnavailableError,
    generate_learning_path,
)

router = APIRouter()
logger = logging.getLogger(__name__)


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
