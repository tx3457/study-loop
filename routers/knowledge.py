"""Authenticated, bounded public gateway to the internal knowledge service."""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from chromadb.errors import ChromaError, NotFoundError
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Path as ApiPath,
    Query,
    UploadFile,
)
from fastapi.responses import JSONResponse

from models.knowledge import (
    CorrectionRequest,
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeCapabilities,
    LegacyImportRequest,
    RevisionRequest,
    WebImportRequest,
)
from routers.documents import (
    ALLOWED_UPLOAD_EXTS,
    MAX_UPLOAD_BYTES,
    PARSE_ERROR_DETAIL,
    _validate_filename,
)
from services.auth import require_user_id
from services.idempotency import InvalidIdempotencyKeyError, normalize_idempotency_key
from services.knowledge_client import KnowledgeServiceError, knowledge_client
from services.mcp_client import mcp_registry
from services.parser import DocumentParseError, UnsupportedFileError, parse_upload
from services.vectorstore import _get_bm25_index, _get_public_document_collection


router = APIRouter()
logger = logging.getLogger(__name__)

_RESOURCE_ID = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _enabled() -> bool:
    return os.getenv("KNOWLEDGE_BASES_ENABLED", "false").strip().lower() in _TRUE_VALUES


def _require_enabled() -> None:
    if not _enabled():
        raise HTTPException(status_code=503, detail="知识库功能未启用")


def _write_headers(idempotency_key: str | None) -> dict[str, str]:
    try:
        key = normalize_idempotency_key(idempotency_key)
    except InvalidIdempotencyKeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if key is None:
        raise HTTPException(status_code=400, detail="缺少 Idempotency-Key")
    return {"Idempotency-Key": key}


def _json_response(response) -> JSONResponse:
    try:
        payload = response.json()
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=503, detail="知识库服务返回无效数据") from exc
    headers = {}
    request_id = response.headers.get("X-Request-ID")
    if request_id:
        headers["X-Request-ID"] = request_id
    return JSONResponse(status_code=response.status_code, content=payload, headers=headers)


def _raise_service_error(exc: KnowledgeServiceError) -> None:
    headers = {"X-Request-ID": exc.request_id} if exc.request_id else None
    raise HTTPException(
        status_code=exc.status_code, detail=exc.detail, headers=headers
    ) from exc


async def _proxy(method: str, path: str, *, owner_id: str, json_body=None, headers=None):
    _require_enabled()
    try:
        response = await knowledge_client.request(
            method,
            path,
            owner_id=owner_id,
            json=json_body,
            headers=headers,
        )
    except KnowledgeServiceError as exc:
        _raise_service_error(exc)
    return _json_response(response)


def _safe_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or len(value) > 100:
        raise HTTPException(status_code=422, detail=PARSE_ERROR_DETAIL)
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=PARSE_ERROR_DETAIL) from exc
    if len(encoded.encode("utf-8")) > 64 * 1024:
        raise HTTPException(status_code=422, detail=PARSE_ERROR_DETAIL)
    return value


async def _read_upload(file: UploadFile) -> tuple[str, bytes, list[dict[str, Any]]]:
    filename = file.filename or ""
    _validate_filename(filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        raise HTTPException(
            status_code=415, detail=f"不支持的文件类型：{ext or '(无扩展名)'}"
        )
    content = bytearray()
    while data := await file.read(1024 * 1024):
        content += data
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"文件超过大小上限 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB",
            )
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    try:
        documents = await parse_upload(bytes(content), filename)
    except UnsupportedFileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentParseError as exc:
        raise HTTPException(status_code=422, detail=PARSE_ERROR_DETAIL) from exc
    if not documents or any("error" in (doc.metadata or {}) for doc in documents):
        logger.warning("knowledge upload parse failed safely: filename=%s", filename)
        raise HTTPException(status_code=422, detail=PARSE_ERROR_DETAIL)
    blocks = [
        {"text": doc.page_content.strip(), "metadata": _safe_metadata(doc.metadata or {})}
        for doc in documents
        if isinstance(doc.page_content, str) and doc.page_content.strip()
    ]
    if not blocks:
        raise HTTPException(status_code=400, detail="文档解析后内容为空")
    return filename, bytes(content), blocks


async def _read_legacy_snapshot(
    document_id: str, owner_id: str
) -> tuple[list[str], list[str]]:
    _validate_filename(document_id)
    try:
        collection = await _get_public_document_collection(document_id, owner_id)
        index = await _get_bm25_index(collection, document_id, owner_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail="旧文档不存在") from exc
    except (ChromaError, RuntimeError) as exc:
        logger.error("legacy snapshot read failed: error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="旧文档存储暂时不可用") from exc
    documents = list(index.get("all_docs") or [])
    ids = list(index.get("all_ids") or [])
    if len(documents) != len(ids) or not documents:
        raise HTTPException(status_code=422, detail="旧文档没有可复制的内容")
    return documents, ids


@router.get("/knowledge-bases/capabilities", response_model=KnowledgeCapabilities)
async def capabilities(owner_id: str = Depends(require_user_id)) -> KnowledgeCapabilities:
    if not _enabled():
        return KnowledgeCapabilities(
            enabled=False, available=False, web_search_available=False
        )
    try:
        response = await knowledge_client.request(
            "GET", "/knowledge-bases/capabilities", owner_id=owner_id
        )
        payload = response.json()
        return KnowledgeCapabilities(
            enabled=True,
            available=bool(payload.get("available", True)),
            web_search_available=(
                bool(payload.get("available", True))
                and mcp_registry.has("mcp_ddg_search")
            ),
        )
    except (KnowledgeServiceError, ValueError, AttributeError):
        return KnowledgeCapabilities(
            enabled=True, available=False, web_search_available=False
        )


@router.get("/knowledge-bases")
async def list_knowledge_bases(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0, le=100_000),
    owner_id: str = Depends(require_user_id),
):
    return await _proxy(
        "GET",
        "/knowledge-bases?" + urlencode({"limit": limit, "offset": offset}),
        owner_id=owner_id,
    )


@router.post("/knowledge-bases")
async def create_knowledge_base(
    body: KnowledgeBaseCreate,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    owner_id: str = Depends(require_user_id),
):
    return await _proxy(
        "POST",
        "/knowledge-bases",
        owner_id=owner_id,
        json_body=body.model_dump(mode="json"),
        headers=_write_headers(idempotency_key),
    )


@router.get("/knowledge-bases/{kb_id}")
async def get_knowledge_base(
    kb_id: str = ApiPath(pattern=_RESOURCE_ID),
    owner_id: str = Depends(require_user_id),
):
    return await _proxy("GET", f"/knowledge-bases/{kb_id}", owner_id=owner_id)


@router.patch("/knowledge-bases/{kb_id}")
async def update_knowledge_base(
    body: KnowledgeBaseUpdate,
    kb_id: str = ApiPath(pattern=_RESOURCE_ID),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    owner_id: str = Depends(require_user_id),
):
    return await _proxy(
        "PATCH",
        f"/knowledge-bases/{kb_id}",
        owner_id=owner_id,
        json_body=body.model_dump(mode="json", exclude_none=True),
        headers=_write_headers(idempotency_key),
    )


@router.delete("/knowledge-bases/{kb_id}")
async def delete_knowledge_base(
    body: RevisionRequest,
    kb_id: str = ApiPath(pattern=_RESOURCE_ID),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    owner_id: str = Depends(require_user_id),
):
    return await _proxy(
        "DELETE",
        f"/knowledge-bases/{kb_id}?" + urlencode(
            {"expected_revision": body.expected_revision}
        ),
        owner_id=owner_id,
        headers=_write_headers(idempotency_key),
    )


@router.get("/knowledge-bases/{kb_id}/documents")
async def list_documents(
    kb_id: str = ApiPath(pattern=_RESOURCE_ID),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0, le=100_000),
    owner_id: str = Depends(require_user_id),
):
    path = f"/knowledge-bases/{kb_id}/documents?" + urlencode(
        {"limit": limit, "offset": offset}
    )
    return await _proxy("GET", path, owner_id=owner_id)


async def _proxy_file(
    *,
    method: str,
    path: str,
    kb_id: str,
    file: UploadFile,
    expected_revision: int,
    idempotency_key: str | None,
    owner_id: str,
):
    _require_enabled()
    filename, content, blocks = await _read_upload(file)
    return await _proxy(
        method,
        path,
        owner_id=owner_id,
        json_body={
            "name": filename,
            "kind": "file",
            "content_base64": base64.b64encode(content).decode("ascii"),
            "parsed_blocks": blocks,
            "expected_revision": expected_revision,
        },
        headers=_write_headers(idempotency_key),
    )


@router.post("/knowledge-bases/{kb_id}/documents/upload")
async def upload_document(
    kb_id: str = ApiPath(pattern=_RESOURCE_ID),
    file: UploadFile = File(...),
    expected_revision: int = Form(..., ge=0),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    owner_id: str = Depends(require_user_id),
):
    return await _proxy_file(
        method="POST",
        path=f"/knowledge-bases/{kb_id}/documents",
        kb_id=kb_id,
        file=file,
        expected_revision=expected_revision,
        idempotency_key=idempotency_key,
        owner_id=owner_id,
    )


@router.put("/knowledge-bases/{kb_id}/documents/{document_id}")
async def replace_document(
    kb_id: str = ApiPath(pattern=_RESOURCE_ID),
    document_id: str = ApiPath(pattern=_RESOURCE_ID),
    file: UploadFile = File(...),
    expected_revision: int = Form(..., ge=0),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    owner_id: str = Depends(require_user_id),
):
    return await _proxy_file(
        method="PUT",
        path=f"/knowledge-bases/{kb_id}/documents/{document_id}",
        kb_id=kb_id,
        file=file,
        expected_revision=expected_revision,
        idempotency_key=idempotency_key,
        owner_id=owner_id,
    )


@router.delete("/knowledge-bases/{kb_id}/documents/{document_id}")
async def delete_document(
    body: RevisionRequest,
    kb_id: str = ApiPath(pattern=_RESOURCE_ID),
    document_id: str = ApiPath(pattern=_RESOURCE_ID),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    owner_id: str = Depends(require_user_id),
):
    return await _proxy(
        "DELETE",
        f"/knowledge-bases/{kb_id}/documents/{document_id}?" + urlencode(
            {"expected_revision": body.expected_revision}
        ),
        owner_id=owner_id,
        headers=_write_headers(idempotency_key),
    )


@router.post("/knowledge-bases/{kb_id}/documents/import-legacy")
async def import_legacy_document(
    body: LegacyImportRequest,
    kb_id: str = ApiPath(pattern=_RESOURCE_ID),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    owner_id: str = Depends(require_user_id),
):
    _require_enabled()
    chunks, chunk_ids = await _read_legacy_snapshot(body.legacy_document_id, owner_id)
    joined = "\n\n".join(chunks).encode("utf-8")
    if len(joined) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="旧文档快照超过知识库导入上限")
    return await _proxy(
        "POST",
        f"/knowledge-bases/{kb_id}/documents",
        owner_id=owner_id,
        json_body={
            "name": body.legacy_document_id,
            "kind": "legacy_copy",
            "content_base64": base64.b64encode(joined).decode("ascii"),
            "parsed_blocks": [
                {"text": text, "metadata": {"legacy_chunk_id": chunk_id}}
                for text, chunk_id in zip(chunks, chunk_ids, strict=True)
            ],
            "expected_revision": body.expected_revision,
            "legacy_document_id": body.legacy_document_id,
        },
        headers=_write_headers(idempotency_key),
    )


@router.get("/knowledge-bases/{kb_id}/graph")
async def get_graph(
    kb_id: str = ApiPath(pattern=_RESOURCE_ID),
    node_limit: int | None = Query(default=None, ge=1, le=200),
    edge_limit: int | None = Query(default=None, ge=1, le=400),
    limit_nodes: int | None = Query(default=None, ge=1, le=200),
    limit_edges: int | None = Query(default=None, ge=1, le=400),
    search: str | None = Query(default=None, max_length=500),
    focus_id: str | None = Query(default=None, max_length=128),
    focus: str | None = Query(default=None, max_length=128),
    owner_id: str = Depends(require_user_id),
):
    if node_limit is not None and limit_nodes is not None and node_limit != limit_nodes:
        raise HTTPException(status_code=422, detail="图谱节点上限参数冲突")
    if edge_limit is not None and limit_edges is not None and edge_limit != limit_edges:
        raise HTTPException(status_code=422, detail="图谱关系上限参数冲突")
    if focus_id is not None and focus is not None and focus_id != focus:
        raise HTTPException(status_code=422, detail="图谱聚焦参数冲突")
    resolved_node_limit = limit_nodes if limit_nodes is not None else node_limit or 100
    resolved_edge_limit = limit_edges if limit_edges is not None else edge_limit or 200
    resolved_focus = focus if focus is not None else focus_id
    params: dict[str, Any] = {
        "node_limit": resolved_node_limit,
        "edge_limit": resolved_edge_limit,
    }
    if search is not None:
        params["search"] = search
    if resolved_focus is not None:
        params["focus_id"] = resolved_focus
    return await _proxy(
        "GET",
        f"/knowledge-bases/{kb_id}/graph?" + urlencode(params),
        owner_id=owner_id,
    )


@router.get("/knowledge-bases/{kb_id}/sources/{source_version_id}")
async def get_source(
    kb_id: str = ApiPath(pattern=_RESOURCE_ID),
    source_version_id: str = ApiPath(pattern=_RESOURCE_ID),
    max_chars: int = Query(20_000, ge=1, le=50_000),
    owner_id: str = Depends(require_user_id),
):
    path = f"/knowledge-bases/{kb_id}/sources/{source_version_id}?" + urlencode(
        {"max_chars": max_chars}
    )
    return await _proxy("GET", path, owner_id=owner_id)


@router.post("/knowledge-bases/{kb_id}/corrections")
async def create_correction(
    body: CorrectionRequest,
    kb_id: str = ApiPath(pattern=_RESOURCE_ID),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    owner_id: str = Depends(require_user_id),
):
    return await _proxy(
        "POST",
        f"/knowledge-bases/{kb_id}/corrections",
        owner_id=owner_id,
        json_body=body.model_dump(mode="json", exclude_none=True),
        headers=_write_headers(idempotency_key),
    )


@router.post("/knowledge-bases/{kb_id}/web-import")
async def import_web_snapshot(
    body: WebImportRequest,
    kb_id: str = ApiPath(pattern=_RESOURCE_ID),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    owner_id: str = Depends(require_user_id),
):
    return await _proxy(
        "POST",
        f"/knowledge-bases/{kb_id}/web-import",
        owner_id=owner_id,
        json_body=body.model_dump(mode="json"),
        headers=_write_headers(idempotency_key),
    )


@router.post("/knowledge-bases/{kb_id}/rebuild")
async def rebuild_knowledge_base(
    body: RevisionRequest,
    kb_id: str = ApiPath(pattern=_RESOURCE_ID),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    owner_id: str = Depends(require_user_id),
):
    return await _proxy(
        "POST",
        f"/knowledge-bases/{kb_id}/rebuild",
        owner_id=owner_id,
        json_body=body.model_dump(mode="json"),
        headers=_write_headers(idempotency_key),
    )


@router.get("/knowledge-jobs/{job_id}")
async def get_job(
    job_id: str = ApiPath(pattern=_RESOURCE_ID),
    owner_id: str = Depends(require_user_id),
):
    return await _proxy("GET", f"/knowledge-jobs/{job_id}", owner_id=owner_id)


@router.post("/knowledge-jobs/{job_id}/retry")
async def retry_job(
    body: RevisionRequest,
    job_id: str = ApiPath(pattern=_RESOURCE_ID),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    owner_id: str = Depends(require_user_id),
):
    return await _proxy(
        "POST",
        f"/knowledge-jobs/{job_id}/retry",
        owner_id=owner_id,
        json_body=body.model_dump(mode="json"),
        headers=_write_headers(idempotency_key),
    )
