from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from .engine import GraphEngine
from .repository import Repository


LOGGER = logging.getLogger(__name__)
ADVISORY_LOCK_ID = 0x53545544594C4F4F


class WriteWorker:
    def __init__(
        self,
        repository: Repository,
        service,
        engine: GraphEngine,
        *,
        operation_timeout_seconds: int = 300,
    ) -> None:
        self.repository = repository
        self.service = service
        self.engine = engine
        self.operation_timeout_seconds = operation_timeout_seconds
        self._stop = asyncio.Event()
        self.healthy = True
        self.last_error_type: str | None = None
        self._needs_recovery = False
        self._next_snapshot_cleanup = 0.0

    async def process_one(self) -> str | None:
        async with self.repository.pool.acquire() as connection:
            locked = await connection.fetchval("SELECT pg_try_advisory_lock($1)", ADVISORY_LOCK_ID)
            if not locked:
                return None
            try:
                job = await self.repository.claim_job(connection)
                if job is None:
                    return None
                payload = job["payload"]
                if isinstance(payload, str):
                    payload = json.loads(payload)
                job["payload"] = payload
                kb = await self.repository.fetchrow(
                    "SELECT * FROM sl_knowledge_bases WHERE id=$1", job["knowledge_base_id"]
                )
                try:
                    async with self.service.locks.write(str(job["knowledge_base_id"])):
                        if job["operation"] != "rebuild":
                            self.service.assert_index_config(kb)
                        async with asyncio.timeout(self.operation_timeout_seconds):
                            await self._execute(job, kb)
                        await self.repository.finish_job(job)
                except Exception as error:
                    LOGGER.error(
                        "knowledge indexing job failed job_id=%s error_type=%s",
                        job["id"],
                        type(error).__name__,
                    )
                    await self.repository.fail_job(job, "index_failed")
                return str(job["id"])
            finally:
                await connection.execute("SELECT pg_advisory_unlock($1)", ADVISORY_LOCK_ID)

    async def _execute(self, job: dict[str, Any], kb: dict[str, Any]) -> None:
        operation = job["operation"]
        payload = job["payload"]
        workspace = kb["workspace"]
        if operation == "ingest_document":
            version = await self.repository.fetchrow(
                "SELECT * FROM sl_source_versions WHERE id=$1 AND knowledge_base_id=$2",
                uuid.UUID(payload["version_id"]),
                job["knowledge_base_id"],
            )
            if version is None:
                raise RuntimeError("canonical source version is absent")
            await self.engine.insert(
                workspace,
                str(version["id"]),
                version["parsed_text"],
                version["source_token"],
            )
            await self._replay_corrections(job["knowledge_base_id"], workspace)
        elif operation == "replace_document":
            corrections = await self._corrections(job["knowledge_base_id"])
            if self._has_identity_corrections(corrections):
                documents, corrections, _selected = await self._rebuild_inputs(job)
                await self.engine.rebuild(workspace, documents, corrections)
            else:
                if payload.get("previous_version_id"):
                    await self.engine.delete_document(workspace, payload["previous_version_id"])
                version = await self.repository.fetchrow(
                    "SELECT * FROM sl_source_versions WHERE id=$1 AND knowledge_base_id=$2",
                    uuid.UUID(payload["version_id"]),
                    job["knowledge_base_id"],
                )
                if version is None:
                    raise RuntimeError("canonical source version is absent")
                await self.engine.insert(
                    workspace,
                    str(version["id"]),
                    version["parsed_text"],
                    version["source_token"],
                )
                await self._apply_corrections(workspace, corrections)
        elif operation == "reuse_document":
            return
        elif operation == "delete_document":
            corrections = await self._corrections(job["knowledge_base_id"])
            if self._has_identity_corrections(corrections):
                documents, corrections, _selected = await self._rebuild_inputs(job)
                await self.engine.rebuild(workspace, documents, corrections)
            else:
                await self.engine.delete_document(workspace, payload["version_id"])
                await self._apply_corrections(workspace, corrections)
        elif operation == "correction":
            await self.engine.apply_correction(workspace, payload["kind"], payload["engine_payload"])
        elif operation == "rebuild":
            all_versions, corrections, selected = await self._rebuild_inputs(job)
            await self.engine.rebuild(
                workspace,
                all_versions,
                corrections,
                clear_llm_cache=kb["index_config_hash"] != self.service.index_hash,
            )
            job["payload"]["active_versions"] = selected
            await self.repository.pool.execute(
                "UPDATE sl_knowledge_bases SET index_config_hash=$2 WHERE id=$1",
                job["knowledge_base_id"],
                self.service.index_hash,
            )
        elif operation == "delete_knowledge_base":
            await self.engine.rebuild(
                workspace, [], [], clear_llm_cache=True
            )
        else:
            raise RuntimeError(f"unsupported job operation: {operation}")

    async def _replay_corrections(self, kb_id, workspace: str) -> None:
        await self._apply_corrections(workspace, await self._corrections(kb_id))

    async def _corrections(self, kb_id) -> list[dict[str, Any]]:
        rows = await self.repository.fetch(
            "SELECT kind,payload FROM sl_corrections WHERE knowledge_base_id=$1 ORDER BY sequence",
            kb_id,
        )
        for row in rows:
            payload = row["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            row["payload"] = payload.get("engine_payload", payload)
        return rows

    async def _apply_corrections(
        self, workspace: str, corrections: list[dict[str, Any]]
    ) -> None:
        for correction in corrections:
            await self.engine.apply_correction(
                workspace, correction["kind"], correction["payload"]
            )

    @staticmethod
    def _has_identity_corrections(corrections: list[dict[str, Any]]) -> bool:
        return any(
            correction["kind"] in {"rename_entity", "merge_entities"}
            for correction in corrections
        )

    async def _rebuild_inputs(
        self, job: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
        all_versions = await self.repository.fetch(
            "SELECT v.id AS version_id,v.document_id,v.parsed_text,v.source_token,"
            "v.status AS version_status,v.created_at,d.status AS document_status,"
            "d.current_version_id FROM sl_source_versions v JOIN sl_documents d "
            "ON d.id=v.document_id WHERE v.knowledge_base_id=$1 ORDER BY v.created_at",
            job["knowledge_base_id"],
        )
        selected: dict[str, str] = {}
        by_document: dict[str, list[dict[str, Any]]] = {}
        for row in all_versions:
            by_document.setdefault(str(row["document_id"]), []).append(row)
        for document_id, versions in by_document.items():
            if versions[0]["document_status"] == "deleted":
                continue
            pending = [row for row in versions if row["version_status"] == "pending"]
            chosen = pending[-1] if pending else next(
                (
                    row
                    for row in versions
                    if row["version_id"] == row["current_version_id"]
                ),
                versions[-1],
            )
            selected[document_id] = str(chosen["version_id"])
        payload = job["payload"]
        if job["operation"] == "delete_document":
            selected.pop(str(payload["document_id"]), None)
        elif job["operation"] == "replace_document":
            selected[str(payload["document_id"])] = str(payload["version_id"])
        for row in all_versions:
            row["version_id"] = str(row["version_id"])
            row["active"] = selected.get(str(row["document_id"])) == row["version_id"]
        return all_versions, await self._corrections(job["knowledge_base_id"]), selected

    async def run(self) -> None:
        while not self._stop.is_set():
            try:
                if self._needs_recovery:
                    await self.repository.mark_interrupted_dirty()
                    self._needs_recovery = False
                now = asyncio.get_running_loop().time()
                if now >= self._next_snapshot_cleanup:
                    await self.repository.cleanup_expired_snapshots(limit=100)
                    self._next_snapshot_cleanup = now + 60.0
                processed = await self.process_one()
                self.healthy = True
                self.last_error_type = None
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.healthy = False
                self.last_error_type = type(error).__name__
                self._needs_recovery = True
                LOGGER.error(
                    "knowledge worker supervisor recovered error_type=%s",
                    self.last_error_type,
                )
                processed = None
            if processed is None:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=0.5)
                except TimeoutError:
                    pass

    def stop(self) -> None:
        self._stop.set()
