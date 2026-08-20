"""
文档上传 / 列表 / 删除

parser 自动识别 .pdf/.docx/.txt/.md/图片(OCR)，chunker 自适应切分。
"""

import os
import logging
from pathlib import Path

from chromadb.errors import ChromaError, NotFoundError
from fastapi import APIRouter, File, HTTPException, UploadFile

from models.quiz import Quiz
from services.chunker import default_chunker
from services.parser import DocumentParseError, UnsupportedFileError, parse_upload
from services.vectorstore import (
    DocumentAlreadyExistsError,
    deal_document,
    delete_document,
    get_all_document,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# 上传安全限制:扩展名白名单(fail-fast,读文件前就拒绝) + 大小上限(防 OOM)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024
ALLOWED_UPLOAD_EXTS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
}
PARSE_ERROR_DETAIL = "文档无法解析，请确认文件未损坏且格式正确"
INVALID_FILENAME_DETAIL = "文件名无效，请使用不含路径或控制字符的文件名"


def _validate_filename(filename: str) -> None:
    if (
        len(filename) > 255
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or any(ord(char) < 32 for char in filename)
    ):
        raise HTTPException(status_code=422, detail=INVALID_FILENAME_DETAIL)


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""
    _validate_filename(filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        raise HTTPException(
            status_code=415, detail=f"不支持的文件类型：{ext or '(无扩展名)'}"
        )

    # 流式读取 + 大小上限,避免一次性把超大文件读进内存导致 OOM
    buf = bytearray()
    while data := await file.read(1024 * 1024):
        buf += data
        if len(buf) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"文件超过大小上限 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB",
            )
    content = bytes(buf)
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")

    try:
        docs = await parse_upload(content, filename)
    except UnsupportedFileError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DocumentParseError as e:
        raise HTTPException(status_code=422, detail=PARSE_ERROR_DETAIL) from e

    # OCR/parser failures are represented as error-marked Documents so that one
    # bad page does not crash a batch parser. They are diagnostics, never source
    # material: indexing them would teach the model error messages and may leak
    # local paths or provider details.
    if not docs or any("error" in (doc.metadata or {}) for doc in docs):
        logger.warning("文档解析未产生可安全索引的内容: %s", filename)
        raise HTTPException(status_code=422, detail=PARSE_ERROR_DETAIL)

    chunks = default_chunker.split_documents(docs)
    chunk_texts = [c.page_content for c in chunks if c.page_content.strip()]
    if not chunk_texts:
        raise HTTPException(
            status_code=400,
            detail="文档解析后内容为空（可能是扫描件 OCR 失败或文件损坏）",
        )

    try:
        chunk_count = await deal_document(filename, filename, chunk_texts)
    except DocumentAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ChromaError as e:
        logger.error(
            "文档索引写入失败: error_type=%s",
            type(e).__name__,
        )
        raise HTTPException(status_code=503, detail="文档存储暂时不可用") from e
    return Quiz(
        document_id=filename,
        filename=filename,
        chunks=chunk_count,
        status="indexed",
    )


@router.get("/documents")
async def get_documents():
    try:
        collections = await get_all_document()
    except ChromaError as e:
        logger.error(
            "文档列表读取失败: error_type=%s",
            type(e).__name__,
        )
        raise HTTPException(status_code=503, detail="文档存储暂时不可用") from e
    document_ids = []
    for collection in collections:
        metadata = collection.metadata if isinstance(collection.metadata, dict) else {}
        document_ids.append(
            metadata.get("source_document_id")
            or metadata.get("source_filename")
            or collection.name
        )
    return {"documents": document_ids}


@router.delete("/documents/{document_id}")
async def delete_document_by_id(document_id: str):
    try:
        status = await delete_document(document_id)
    except NotFoundError as e:
        # Includes attempts to address a Unicode document through its internal
        # collection alias rather than the public filename.
        raise HTTPException(status_code=404, detail="文档不存在") from e
    except (ChromaError, RuntimeError) as e:
        logger.error(
            "文档删除失败: error_type=%s",
            type(e).__name__,
        )
        raise HTTPException(status_code=503, detail="文档存储暂时不可用") from e
    return {
        "status": status,
        "document_id": document_id,
        "scope": "material_only",
        "learning_data_retained": True,
        "document_id_reusable": False,
    }
