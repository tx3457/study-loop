from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from .config import Settings
from .engine import LightRAGEngine, index_config_hash
from .errors import ServiceError
from .materials import MaterialStore
from .repository import Repository
from .service import KnowledgeService
from .worker import WriteWorker


def create_app(
    settings: Settings,
    *,
    service: KnowledgeService | None = None,
    worker: WriteWorker | None = None,
    manage_lifespan: bool = True,
) -> FastAPI:
    state: dict[str, Any] = {"service": service, "worker": worker, "pool": None, "task": None}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["service"] is None:
            pool = await asyncpg.create_pool(
                settings.database_url,
                password=os.environ.get("PGPASSWORD") or None,
                min_size=1,
                max_size=10,
            )
            repository = Repository(pool)
            await repository.migrate()
            await repository.mark_interrupted_dirty()
            pinned_index_hash = index_config_hash(settings)
            await repository.mark_config_drift_dirty(pinned_index_hash)
            engine = LightRAGEngine(settings)
            graph_service = KnowledgeService(
                repository,
                MaterialStore(settings.materials_dir),
                engine,
                index_hash=pinned_index_hash,
                query_timeout_seconds=settings.query_timeout_seconds,
            )
            write_worker = WriteWorker(
                repository,
                graph_service,
                engine,
                operation_timeout_seconds=settings.mutation_timeout_seconds,
            )
            state.update(service=graph_service, worker=write_worker, pool=pool)
        if manage_lifespan:
            state["task"] = asyncio.create_task(state["worker"].run())
        try:
            yield
        finally:
            if state["task"]:
                state["worker"].stop()
                state["task"].cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await state["task"]
            engine = getattr(state["service"], "engine", None)
            if hasattr(engine, "close"):
                await engine.close()
            if state["pool"]:
                await state["pool"].close()

    app = FastAPI(title="StudyLoop Knowledge Service", lifespan=lifespan)

    @app.exception_handler(ServiceError)
    async def service_error(_request: Request, error: ServiceError):
        return JSONResponse(
            status_code=error.status_code,
            content={"error_code": error.code, "detail": str(error)},
        )

    @app.exception_handler(KeyError)
    async def not_found(_request: Request, _error: KeyError):
        return JSONResponse(status_code=404, content={"error_code": "not_found"})

    @app.exception_handler(ValueError)
    async def invalid_request(_request: Request, error: ValueError):
        return JSONResponse(
            status_code=422,
            content={"error_code": "invalid_request", "detail": str(error)},
        )

    def current_service() -> KnowledgeService:
        graph_service = state["service"]
        if graph_service is None:
            raise HTTPException(503, detail={"code": "unavailable"})
        return graph_service

    async def subject(
        authorization: str | None = Header(default=None),
        x_studyloop_subject: str | None = Header(default=None),
    ) -> str:
        expected = f"Bearer {settings.internal_token}"
        if (
            authorization is None
            or not secrets.compare_digest(authorization, expected)
            or not x_studyloop_subject
        ):
            raise HTTPException(401, detail={"code": "unauthorized"})
        return x_studyloop_subject

    def key(idempotency_key: str | None) -> str:
        if not idempotency_key:
            raise HTTPException(400, detail={"code": "idempotency_key_required"})
        return idempotency_key

    @app.get("/health/live")
    async def live():
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready():
        graph_service = state["service"]
        if graph_service is None:
            raise HTTPException(503, detail={"code": "not_ready"})
        write_worker = state["worker"]
        worker_task = state["task"]
        if manage_lifespan and (
            write_worker is None
            or not write_worker.healthy
            or worker_task is None
            or worker_task.done()
        ):
            raise HTTPException(503, detail={"code": "worker_unavailable"})
        try:
            await graph_service.repository.pool.fetchval("SELECT 1")
        except Exception as error:
            raise HTTPException(503, detail={"code": "database_unavailable"}) from error
        return {"status": "ready"}

    @app.post("/knowledge-bases", status_code=status.HTTP_201_CREATED)
    async def create_kb(
        payload: dict,
        owner: str = Depends(subject),
        idempotency_key: str | None = Header(default=None),
    ):
        return await current_service().create_knowledge_base(
            owner,
            payload.get("name", ""),
            payload.get("description", ""),
            idempotency_key=key(idempotency_key),
        )

    @app.get("/knowledge-bases")
    async def list_kbs(
        owner: str = Depends(subject),
        limit: int = Query(50, ge=1, le=100),
        offset: int = Query(0, ge=0, le=100_000),
    ):
        return await current_service().list_knowledge_bases(owner, limit, offset)

    @app.get("/knowledge-bases/capabilities")
    async def capabilities(_owner: str = Depends(subject)):
        return {"enabled": True, "available": True, "web_search_available": False}

    @app.get("/knowledge-bases/{kb_id}")
    async def get_kb(kb_id: str, owner: str = Depends(subject)):
        return await current_service().get_knowledge_base(owner, kb_id)

    @app.patch("/knowledge-bases/{kb_id}")
    async def patch_kb(
        kb_id: str,
        payload: dict,
        owner: str = Depends(subject),
        idempotency_key: str | None = Header(default=None),
    ):
        return await current_service().update_knowledge_base(
            owner, kb_id, payload, idempotency_key=key(idempotency_key)
        )

    @app.delete("/knowledge-bases/{kb_id}", status_code=202)
    async def delete_kb(
        kb_id: str,
        expected_revision: int,
        owner: str = Depends(subject),
        idempotency_key: str | None = Header(default=None),
    ):
        return await current_service().delete_knowledge_base(
            owner, kb_id, expected_revision, key(idempotency_key)
        )

    @app.post("/knowledge-bases/{kb_id}/documents", status_code=202)
    async def add_document(
        kb_id: str,
        payload: dict,
        owner: str = Depends(subject),
        idempotency_key: str | None = Header(default=None),
    ):
        return await current_service().ingest_document(
            owner, kb_id, payload, idempotency_key=key(idempotency_key)
        )

    @app.get("/knowledge-bases/{kb_id}/documents")
    async def documents(
        kb_id: str,
        owner: str = Depends(subject),
        limit: int = Query(50, ge=1, le=100),
        offset: int = Query(0, ge=0, le=100_000),
    ):
        return await current_service().list_documents(owner, kb_id, limit, offset)

    @app.put("/knowledge-bases/{kb_id}/documents/{document_id}", status_code=202)
    async def replace_document(
        kb_id: str,
        document_id: str,
        payload: dict,
        owner: str = Depends(subject),
        idempotency_key: str | None = Header(default=None),
    ):
        return await current_service().ingest_document(
            owner,
            kb_id,
            payload,
            idempotency_key=key(idempotency_key),
            document_id=document_id,
        )

    @app.delete("/knowledge-bases/{kb_id}/documents/{document_id}", status_code=202)
    async def remove_document(
        kb_id: str,
        document_id: str,
        expected_revision: int,
        owner: str = Depends(subject),
        idempotency_key: str | None = Header(default=None),
    ):
        return await current_service().delete_document(
            owner, kb_id, document_id, expected_revision, key(idempotency_key)
        )

    @app.post("/knowledge-bases/{kb_id}/documents/import-legacy", status_code=202)
    async def import_legacy(
        kb_id: str,
        payload: dict,
        owner: str = Depends(subject),
        idempotency_key: str | None = Header(default=None),
    ):
        payload = {**payload, "kind": "legacy_copy"}
        return await current_service().ingest_document(
            owner, kb_id, payload, idempotency_key=key(idempotency_key)
        )

    @app.get("/knowledge-bases/{kb_id}/scope")
    async def scope(kb_id: str, owner: str = Depends(subject)):
        return await current_service().get_scope(owner, kb_id)

    @app.post("/knowledge-bases/{kb_id}/query")
    async def query_kb(
        kb_id: str, payload: dict, request: Request, owner: str = Depends(subject)
    ):
        task = asyncio.create_task(
            current_service().query(
                owner,
                kb_id,
                payload["query"],
                int(payload["expected_revision"]),
                int(payload["expected_epoch"]),
            )
        )
        try:
            while not task.done():
                if await request.is_disconnected():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                    raise HTTPException(499, detail={"code": "client_disconnected"})
                await asyncio.sleep(0.1)
            return await task
        finally:
            if not task.done():
                task.cancel()

    @app.get("/knowledge-bases/{kb_id}/graph")
    async def graph(
        kb_id: str,
        owner: str = Depends(subject),
        search: str | None = None,
        focus_id: str | None = None,
        node_limit: int = Query(200, ge=1, le=200),
        edge_limit: int = Query(400, ge=1, le=400),
    ):
        return await current_service().graph(
            owner, kb_id, search, node_limit, edge_limit, focus_id
        )

    @app.get("/knowledge-bases/{kb_id}/sources/{version_id}")
    async def source(
        kb_id: str,
        version_id: str,
        owner: str = Depends(subject),
        max_chars: int = Query(20_000, ge=1, le=50_000),
    ):
        return await current_service().get_source(owner, kb_id, version_id, max_chars)

    @app.post("/knowledge-bases/{kb_id}/corrections", status_code=202)
    async def correction(
        kb_id: str,
        payload: dict,
        owner: str = Depends(subject),
        idempotency_key: str | None = Header(default=None),
    ):
        return await current_service().create_correction(
            owner, kb_id, payload, idempotency_key=key(idempotency_key)
        )

    @app.post("/knowledge-bases/{kb_id}/web-import", status_code=202)
    async def web_import(
        kb_id: str,
        payload: dict,
        owner: str = Depends(subject),
        idempotency_key: str | None = Header(default=None),
    ):
        return await current_service().import_web_snapshot(
            owner,
            kb_id,
            payload["snapshot_id"],
            expected_revision=int(payload["expected_revision"]),
            idempotency_key=key(idempotency_key),
        )

    @app.post("/web-snapshots", status_code=201)
    async def create_snapshot(payload: dict, owner: str = Depends(subject)):
        return await current_service().create_web_snapshot(owner, payload)

    @app.get("/web-snapshots/{snapshot_id}")
    async def get_snapshot(
        snapshot_id: str, session_id: str, owner: str = Depends(subject)
    ):
        return await current_service().get_web_snapshot(owner, snapshot_id, session_id)

    @app.get("/knowledge-jobs/{job_id}")
    async def get_job(job_id: str, owner: str = Depends(subject)):
        return await current_service().get_job(owner, job_id)

    @app.post("/knowledge-jobs/{job_id}/retry", status_code=202)
    async def retry_job(job_id: str, payload: dict, owner: str = Depends(subject)):
        return await current_service().retry_job(
            owner, job_id, int(payload["expected_revision"])
        )

    @app.post("/knowledge-bases/{kb_id}/rebuild", status_code=202)
    async def rebuild(
        kb_id: str,
        payload: dict,
        owner: str = Depends(subject),
        idempotency_key: str | None = Header(default=None),
    ):
        return await current_service().create_rebuild_job(
            owner, kb_id, int(payload["expected_revision"]), key(idempotency_key)
        )

    return app


def application() -> FastAPI:
    return create_app(Settings.from_env())
