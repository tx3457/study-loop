from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from .engine import DeletionRefused, GraphEngine
from .repository import Repository


LOGGER = logging.getLogger(__name__)
ADVISORY_LOCK_ID = 0x53545544594C4F4F
PURGING_OPERATIONS = frozenset({"delete_document", "delete_knowledge_base"})
# Jobs after which some extraction-cache entries may no longer belong to a live chunk.
CACHE_ORPHANING_OPERATIONS = frozenset({"delete_document", "replace_document", "rebuild"})


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
        # Workspaces still to sweep once after start; None until listed.
        self._orphan_cache_backlog: list[str] | None = None
        self._material_cursor = None

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
                else:
                    await self._after_success(job, kb)
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
                await self._rebuild_for(job, workspace)
            else:
                if payload.get("previous_version_id"):
                    try:
                        await self.engine.delete_document(
                            workspace, payload["previous_version_id"]
                        )
                    except DeletionRefused:
                        await self._rebuild_for(job, workspace)
                        return
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
                await self._rebuild_for(job, workspace)
            else:
                try:
                    await self.engine.delete_document(workspace, payload["version_id"])
                except DeletionRefused:
                    await self._rebuild_for(job, workspace)
                    return
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

    async def _rebuild_for(self, job: dict[str, Any], workspace: str) -> None:
        # Rebuilding from the canonical versions this job leaves live is the one
        # removal that works whatever state the SDK's per-document records are in.
        documents, corrections, _selected = await self._rebuild_inputs(job)
        await self.engine.rebuild(workspace, documents, corrections)

    async def _after_success(self, job: dict[str, Any], kb: dict[str, Any]) -> None:
        # Cleanup that cannot finish now must not fail a committed job or hold up
        # indexing; the periodic sweep finishes it.
        try:
            if job["operation"] in CACHE_ORPHANING_OPERATIONS:
                await self.repository.drop_orphan_extraction_cache(kb["workspace"])
            if job["operation"] == "delete_document":
                # Its own files now, whatever backlog the shared cursor is working through.
                await self._unlink_materials(
                    await self.repository.document_materials(job["payload"]["document_id"])
                )
            if job["operation"] in PURGING_OPERATIONS:
                await self.purge_deleted_data()
        except Exception as error:
            LOGGER.error(
                "post-job cleanup deferred job_id=%s error_type=%s",
                job["id"],
                type(error).__name__,
            )

    async def purge_deleted_data(self, *, max_material_pages: int = 10) -> None:
        """Remove what deleted knowledge bases and documents still hold on disk and in rows.

        Runs after each delete job and on the periodic sweep, so a crash between a
        delete job committing and its files going away is finished on the next pass.
        Only positively deleted data is touched; an upload still writing its file is
        never mistaken for an orphan.
        """
        for kb in await self.repository.knowledge_bases_pending_purge():
            try:
                # Files first, by recorded path: the rows are the only record of
                # where each one was written.
                for path in await self.repository.knowledge_base_materials(kb["id"]):
                    self.service.materials.delete(path)
                await self.repository.purge_knowledge_base_rows(kb["id"])
                if not await self.engine.forget(kb["workspace"]):
                    continue
                self.service.materials.delete_knowledge_base(str(kb["id"]))
                await self.repository.mark_knowledge_base_purged(kb["id"])
            except Exception as error:
                LOGGER.error(
                    "knowledge base purge deferred kb_id=%s error_type=%s",
                    kb["id"],
                    type(error).__name__,
                )
        # Bounded per pass so a large backlog cannot hold up indexing; the cursor
        # carries on from there next time and wraps at the end, so files that keep
        # failing are retried without starving the rest.
        for _page in range(max_material_pages):
            rows = await self.repository.deleted_materials(self._material_cursor)
            if not rows:
                self._material_cursor = None
                break
            await self._unlink_materials(rows)
            self._material_cursor = rows[-1]["id"]

    async def _unlink_materials(self, rows: list[dict[str, Any]]) -> None:
        purged = []
        for row in rows:
            try:
                self.service.materials.delete(row["material_path"])
            except OSError as error:
                LOGGER.error(
                    "source material purge deferred version_id=%s error_type=%s",
                    row["id"],
                    type(error).__name__,
                )
            else:
                purged.append(row["id"])
        if purged:
            await self.repository.mark_materials_purged(purged)

    async def _periodic_cleanup(self) -> None:
        try:
            await self.purge_deleted_data()
            # Once per process, one workspace per pass: caches left by failed jobs or
            # older releases. A workspace that errors is not retried until restart;
            # its next successful delete, replace or rebuild sweeps it anyway.
            if self._orphan_cache_backlog is None:
                self._orphan_cache_backlog = await self.repository.ready_workspaces()
            if self._orphan_cache_backlog:
                await self.repository.drop_orphan_extraction_cache(
                    self._orphan_cache_backlog.pop()
                )
        except Exception as error:
            LOGGER.error("periodic cleanup deferred error_type=%s", type(error).__name__)

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
                    await self._periodic_cleanup()
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
