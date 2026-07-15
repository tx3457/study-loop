"""
文档上传 / 列表 / 删除（Phase 8 升级：接入多格式 parser + AdaptiveChunker）

升级前：只支持 utf-8 文本，content.decode("utf-8")，PDF 上传会 UnicodeDecodeError。
升级后：parser 自动识别 .pdf/.docx/.txt/.md/图片(OCR)，chunker 自适应切分。
"""
import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from models.quiz import Quiz
from services.chunker import default_chunker
from services.parser import UnsupportedFileError, parse_upload
from services.vectorstore import deal_document, delete_document, get_all_document

router = APIRouter()

# 上传安全限制:扩展名白名单(fail-fast,读文件前就拒绝) + 大小上限(防 OOM)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024
ALLOWED_UPLOAD_EXTS = {".pdf", ".docx", ".txt", ".md",
                       ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif"}


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        raise HTTPException(status_code=415, detail=f"不支持的文件类型：{ext or '(无扩展名)'}")

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
        docs = await parse_upload(content, file.filename)
    except UnsupportedFileError as e:
        raise HTTPException(status_code=400, detail=str(e))

    chunks = default_chunker.split_documents(docs)
    chunk_texts = [c.page_content for c in chunks if c.page_content.strip()]
    if not chunk_texts:
        raise HTTPException(status_code=400, detail="文档解析后内容为空（可能是扫描件 OCR 失败或文件损坏）")

    chunk_count = await deal_document(file.filename, file.filename, chunk_texts)
    return Quiz(
        document_id=file.filename,
        filename=file.filename,
        chunks=chunk_count,
        status="indexed",
    )


@router.get("/documents")
async def get_documents():
    collections = await get_all_document()
    return {"documents": [c.name for c in collections]}


@router.delete("/documents/{document_id}")
async def delete_document_by_id(document_id: str):
    await delete_document(document_id)
    return {"status": "deleted", "document_id": document_id}
