from __future__ import annotations

import base64
import binascii
import asyncio
import hashlib
import json
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any

from .engine import GraphEngine
from .errors import Conflict, Expired, TooLarge, Unavailable
from .locks import KeyedRWLocks
from .materials import MaterialStore
from .repository import Repository


MAX_MATERIAL_BYTES = 20 * 1024 * 1024
MAX_SOURCE_BLOCKS = 100
MAX_SOURCE_METADATA_BYTES = 16 * 1024


def _json_value(value: Any) -> Any:
    if isinstance(value, (uuid.UUID, datetime)):
        return value.isoformat() if isinstance(value, datetime) else str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _request_hash(operation: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"operation": operation, "payload": payload}, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _material_request_hash(
    kb_id: str, payload: dict[str, Any], *,
    document_id: str | None = None, snapshot_id: str | None = None,
) -> str:
    operation = (
        "replace_document" if document_id else
        "import_web_snapshot" if snapshot_id else "ingest_document"
    )
    material = {
        key: payload.get(key)
        for key in (
            "name", "kind", "content_base64", "parsed_blocks", "expected_revision",
            "legacy_document_id", "source_url", "source_fetched_at",
        )
    }
    return _request_hash(operation, {
        "kb_id": kb_id, "document_id": document_id,
        "snapshot_id": snapshot_id, "material": material,
    })


def _workspace(owner: str, kb_id: str) -> str:
    return "kb_" + hashlib.sha256(f"{owner}\0{kb_id}".encode()).hexdigest()[:40]


def _bounded_source_blocks(
    blocks: list[dict[str, Any]], max_chars: int
) -> tuple[list[dict[str, Any]], bool]:
    projected: list[dict[str, Any]] = []
    cursor = 0
    metadata_used = 0
    truncated = False
    for index, block in enumerate(blocks):
        if index >= MAX_SOURCE_BLOCKS or cursor >= max_chars:
            truncated = True
            break
        text = str(block.get("text", ""))
        visible = text[: max(0, max_chars - cursor)]
        metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
        bounded_metadata: dict[str, Any] = {}
        for key in sorted(metadata):
            candidate = {**bounded_metadata, key: metadata[key]}
            encoded = json.dumps(
                candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
            ).encode()
            if metadata_used + len(encoded) > MAX_SOURCE_METADATA_BYTES:
                truncated = True
                continue
            bounded_metadata = candidate
        metadata_used += len(
            json.dumps(
                bounded_metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        )
        projected.append({"text": visible, "metadata": bounded_metadata})
        if len(visible) < len(text):
            truncated = True
            break
        cursor += len(text) + 2
    if len(projected) < len(blocks):
        truncated = True
    return projected, truncated


class KnowledgeService:
    def __init__(
        self,
        repository: Repository,
        materials: MaterialStore,
        engine: GraphEngine,
        *,
        index_hash: str | None = None,
        locks: KeyedRWLocks | None = None,
        query_timeout_seconds: int = 30,
    ) -> None:
        self.repository = repository
        self.materials = materials
        self.engine = engine
        self.index_hash = index_hash or hashlib.sha256(b"test-index-config").hexdigest()
        self.locks = locks or KeyedRWLocks()
        self.query_timeout_seconds = query_timeout_seconds

    def assert_index_config(self, kb: dict[str, Any]) -> None:
        if kb["index_config_hash"] != self.index_hash:
            raise Unavailable("knowledge base index configuration changed; rebuild required")

    async def create_knowledge_base(
        self,
        owner: str,
        name: str,
        description: str = "",
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if not name.strip():
            raise ValueError("name is required")
        kb_id = uuid.uuid4()
        async def mutate(connection):
            row = dict(
                await connection.fetchrow(
                    "INSERT INTO sl_knowledge_bases(id,owner,workspace,name,description,index_config_hash) "
                    "VALUES($1,$2,$3,$4,$5,$6) RETURNING *",
                    kb_id,
                    owner,
                    _workspace(owner, str(kb_id)),
                    name.strip(),
                    description.strip(),
                    self.index_hash,
                )
            )
            return {
                "id": str(row["id"]),
                "name": row["name"],
                "description": row["description"],
                "status": row["status"],
                "revision": 0,
                "epoch": 0,
                "document_count": 0,
                "created_at": row["created_at"].isoformat(),
            }

        return await self.repository.idempotent_mutation(
            owner,
            idempotency_key or f"internal:create:{uuid.uuid4()}",
            _request_hash("create_knowledge_base", {"name": name, "description": description}),
            mutate,
        )

    async def _kb_view(self, row: dict[str, Any]) -> dict[str, Any]:
        count = await self.repository.pool.fetchval(
            "SELECT count(*) FROM sl_documents WHERE knowledge_base_id=$1 AND status!='deleted'",
            row["id"],
        )
        return {
            "id": str(row["id"]),
            "name": row["name"],
            "description": row["description"],
            "status": (
                row["status"]
                if row["index_config_hash"] == self.index_hash
                else "dirty"
            ),
            "revision": row["revision"],
            "epoch": row["epoch"],
            "document_count": count,
            "created_at": row["created_at"].isoformat(),
        }

    async def get_knowledge_base(self, owner: str, kb_id: str) -> dict[str, Any]:
        row = await self.repository.get_kb(owner, kb_id)
        if row["status"] == "deleting":
            raise KeyError(kb_id)
        return await self._kb_view(row)

    async def list_knowledge_bases(
        self, owner: str, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        rows = await self.repository.fetch(
            "SELECT * FROM sl_knowledge_bases WHERE owner=$1 AND status!='deleting' "
            "ORDER BY created_at LIMIT $2 OFFSET $3",
            owner,
            limit,
            offset,
        )
        return {"knowledge_bases": [await self._kb_view(row) for row in rows]}

    async def update_knowledge_base(
        self,
        owner: str,
        kb_id: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        expected = payload.get("expected_revision")
        async with self.locks.write(kb_id):
            async def mutate(connection):
                updated = await connection.fetchrow(
                    "UPDATE sl_knowledge_bases SET name=COALESCE($4,name),"
                    "description=COALESCE($5,description),revision=revision+1,epoch=epoch+1,"
                    "updated_at=now() WHERE id=$1 AND owner=$2 AND revision=$3 AND status='ready' "
                    "RETURNING *",
                    uuid.UUID(kb_id),
                    owner,
                    expected,
                    payload.get("name"),
                    payload.get("description"),
                )
                if updated is None:
                    existing = await connection.fetchrow(
                        "SELECT index_config_hash FROM sl_knowledge_bases "
                        "WHERE id=$1 AND owner=$2",
                        uuid.UUID(kb_id),
                        owner,
                    )
                    if existing and existing["index_config_hash"] != self.index_hash:
                        raise Unavailable(
                            "knowledge base index configuration changed; rebuild required"
                        )
                    raise Conflict("knowledge base revision or status changed")
                count = await connection.fetchval(
                    "SELECT count(*) FROM sl_documents WHERE knowledge_base_id=$1 AND status!='deleted'",
                    uuid.UUID(kb_id),
                )
                return {
                    "id": str(updated["id"]),
                    "name": updated["name"],
                    "description": updated["description"],
                    "status": updated["status"],
                    "revision": updated["revision"],
                    "epoch": updated["epoch"],
                    "document_count": count,
                    "created_at": updated["created_at"].isoformat(),
                }

            return await self.repository.idempotent_mutation(
                owner,
                idempotency_key,
                _request_hash("update_knowledge_base", {"kb_id": kb_id, **payload}),
                mutate,
            )

    async def delete_knowledge_base(
        self,
        owner: str,
        kb_id: str,
        expected_revision: int,
        idempotency_key: str,
    ) -> dict[str, str]:
        async def setup(_connection):
            return None

        async with self.locks.write(kb_id):
            return await self.repository.begin_job(
                owner=owner,
                kb_id=kb_id,
                expected_revision=expected_revision,
                idempotency_key=idempotency_key,
                request_hash=_request_hash(
                    "delete_knowledge_base",
                    {"knowledge_base_id": kb_id, "expected_revision": expected_revision},
                ),
                job_id=str(uuid.uuid4()),
                operation="delete_knowledge_base",
                payload={},
                setup=setup,
                index_config_hash=self.index_hash,
            )

    def _decode_material(self, payload: dict[str, Any]) -> tuple[bytes, str, list[dict]]:
        try:
            raw = base64.b64decode(payload["content_base64"], validate=True)
        except (KeyError, binascii.Error, ValueError) as error:
            raise ValueError("content_base64 must be valid base64") from error
        if len(raw) > MAX_MATERIAL_BYTES:
            raise TooLarge("material exceeds 20 MiB")
        blocks = payload.get("parsed_blocks")
        if not isinstance(blocks, list) or not blocks:
            raise ValueError("parsed_blocks must be a non-empty list")
        text_parts = []
        normalized = []
        for block in blocks:
            text = block.get("text") if isinstance(block, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError("every parsed block must contain non-empty text")
            metadata = block.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("parsed block metadata must be an object")
            text_parts.append(text.strip())
            normalized.append({"text": text.strip(), "metadata": metadata})
        return raw, "\n\n".join(text_parts), normalized

    async def ingest_document(
        self,
        owner: str,
        kb_id: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
        document_id: str | None = None,
        snapshot_id: str | None = None,
    ) -> dict[str, str]:
        raw, parsed_text, blocks = self._decode_material(payload)
        kind = payload.get("kind", "file")
        if kind not in {"file", "web", "legacy_copy"}:
            raise ValueError("invalid document kind")
        operation = "replace_document" if document_id else "ingest_document"
        request_hash = _material_request_hash(
            kb_id, payload, document_id=document_id, snapshot_id=snapshot_id,
        )
        replay = await self.repository.replay_idempotency(
            owner, idempotency_key, request_hash
        )
        if replay is not None:
            return replay
        content_hash = hashlib.sha256(raw).hexdigest()
        if document_id is None:
            duplicate = await self.repository.fetchrow(
                "SELECT d.id AS document_id,v.id AS version_id FROM sl_source_versions v "
                "JOIN sl_documents d ON d.id=v.document_id JOIN sl_knowledge_bases k "
                "ON k.id=v.knowledge_base_id WHERE v.knowledge_base_id=$1 AND k.owner=$2 "
                "AND v.content_hash=$3 AND v.status='active' AND d.status='active' "
                "AND d.current_version_id=v.id LIMIT 1",
                uuid.UUID(kb_id),
                owner,
                content_hash,
            )
            if duplicate:
                async def no_setup(_connection):
                    return None

                async with self.locks.write(kb_id):
                    return await self.repository.begin_job(
                        owner=owner,
                        kb_id=kb_id,
                        expected_revision=int(payload["expected_revision"]),
                        idempotency_key=idempotency_key,
                        request_hash=request_hash,
                        job_id=str(uuid.uuid4()),
                        operation="reuse_document",
                        payload={
                            "document_id": str(duplicate["document_id"]),
                            "version_id": str(duplicate["version_id"]),
                        },
                        setup=no_setup,
                        index_config_hash=self.index_hash,
                    )
        document_uuid = uuid.UUID(document_id) if document_id else uuid.uuid4()
        version_uuid = uuid.uuid4()
        job_uuid = uuid.uuid4()
        source_token = "src_" + secrets.token_hex(20)
        material_path = self.materials.put(kb_id, str(version_uuid), raw)
        job_payload = {
            "document_id": str(document_uuid),
            "version_id": str(version_uuid),
            "previous_version_id": None,
        }
        async with self.locks.write(kb_id):
            async def setup(connection):
                previous = None
                if document_id:
                    existing = await connection.fetchrow(
                        "SELECT * FROM sl_documents WHERE id=$1 AND knowledge_base_id=$2 AND status!='deleted'",
                        document_uuid,
                        uuid.UUID(kb_id),
                    )
                    if existing is None:
                        raise KeyError(document_id)
                    previous = existing["current_version_id"]
                else:
                    await connection.execute(
                        "INSERT INTO sl_documents(id,knowledge_base_id,name,kind,legacy_document_id,"
                        "source_url,source_fetched_at) VALUES($1,$2,$3,$4,$5,$6,$7)",
                        document_uuid,
                        uuid.UUID(kb_id),
                        payload["name"],
                        kind,
                        payload.get("legacy_document_id"),
                        payload.get("source_url"),
                        datetime.fromisoformat(payload["source_fetched_at"])
                        if payload.get("source_fetched_at")
                        else None,
                    )
                job_payload["previous_version_id"] = str(previous) if previous else None
                await connection.execute(
                    "INSERT INTO sl_source_versions(id,document_id,knowledge_base_id,content_hash,"
                    "material_path,parsed_text,parsed_blocks,source_token) "
                    "VALUES($1,$2,$3,$4,$5,$6,$7::jsonb,$8)",
                    version_uuid,
                    document_uuid,
                    uuid.UUID(kb_id),
                    content_hash,
                    material_path,
                    parsed_text,
                    json.dumps(blocks),
                    source_token,
                )

            try:
                response = await self.repository.begin_job(
                    owner=owner,
                    kb_id=kb_id,
                    expected_revision=int(payload["expected_revision"]),
                    idempotency_key=idempotency_key,
                    request_hash=request_hash,
                    job_id=str(job_uuid),
                    operation=operation,
                    payload=job_payload,
                    setup=setup,
                    index_config_hash=self.index_hash,
                )
            except Exception:
                self.materials.delete(material_path)
                raise
            if response["job_id"] != str(job_uuid):
                self.materials.delete(material_path)
            return response

    async def delete_document(
        self,
        owner: str,
        kb_id: str,
        document_id: str,
        expected_revision: int,
        idempotency_key: str,
    ) -> dict[str, str]:
        job_id = str(uuid.uuid4())
        payload: dict[str, Any] = {"document_id": document_id, "version_id": None}
        async with self.locks.write(kb_id):
            async def setup(connection):
                document = await connection.fetchrow(
                    "SELECT current_version_id FROM sl_documents WHERE id=$1 AND knowledge_base_id=$2 "
                    "AND status!='deleted'",
                    uuid.UUID(document_id),
                    uuid.UUID(kb_id),
                )
                if document is None:
                    raise KeyError(document_id)
                payload["version_id"] = str(document["current_version_id"])

            return await self.repository.begin_job(
                owner=owner,
                kb_id=kb_id,
                expected_revision=expected_revision,
                idempotency_key=idempotency_key,
                request_hash=_request_hash("delete_document", {
                    "kb_id": kb_id, "document_id": document_id,
                    "expected_revision": expected_revision,
                }),
                job_id=job_id,
                operation="delete_document",
                payload=payload,
                setup=setup,
                index_config_hash=self.index_hash,
            )

    async def list_documents(
        self, owner: str, kb_id: str, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        async with self.locks.read(kb_id):
            await self.repository.get_kb(owner, kb_id)
            rows = await self.repository.fetch(
                "SELECT d.*,v.content_hash FROM sl_documents d LEFT JOIN sl_source_versions v "
                "ON v.id=d.current_version_id WHERE d.knowledge_base_id=$1 AND d.status!='deleted' "
                "ORDER BY d.created_at,d.id LIMIT $2 OFFSET $3",
                uuid.UUID(kb_id), limit, offset,
            )
            total = await self.repository.pool.fetchval(
                "SELECT count(*) FROM sl_documents WHERE knowledge_base_id=$1 AND status!='deleted'",
                uuid.UUID(kb_id),
            )
        return {
            "documents": [self._document_view(row) for row in rows],
            "total": total, "limit": limit, "offset": offset,
        }

    def _document_view(self, row: dict[str, Any]) -> dict[str, Any]:
        result = {
            "id": str(row["id"]),
            "knowledge_base_id": str(row["knowledge_base_id"]),
            "name": row["name"],
            "kind": row["kind"],
            "version_id": str(row["current_version_id"]) if row["current_version_id"] else None,
            "status": row["status"],
            "content_hash": row.get("content_hash"),
            "created_at": row["created_at"].isoformat(),
        }
        if row.get("source_url"):
            result["source_url"] = row["source_url"]
        if row.get("source_fetched_at"):
            result["source_fetched_at"] = row["source_fetched_at"].isoformat()
        if row.get("legacy_document_id"):
            result["legacy_document_id"] = row["legacy_document_id"]
        return result

    async def get_scope(self, owner: str, kb_id: str) -> dict[str, Any]:
        kb = await self.repository.get_kb(owner, kb_id)
        scope_status = (
            kb["status"] if kb["index_config_hash"] == self.index_hash else "dirty"
        )
        return {
            "knowledge_base_id": kb_id,
            "revision": kb["revision"],
            "epoch": kb["epoch"],
            "status": scope_status,
        }

    async def query(
        self, owner: str, kb_id: str, query: str, expected_revision: int, expected_epoch: int
    ) -> dict[str, Any]:
        # Check availability before queueing on the per-KB lock. An indexing job
        # holds that lock for the whole run, so the identical checks below are
        # unreachable exactly when they matter most: without this the caller
        # blocks until its own HTTP timeout instead of being told the index is
        # busy. The in-lock checks stay authoritative; this one only fails fast.
        #
        # Scaffolding, not a fix: a job can still take the write lock between
        # this check and the one below. The deadline below now bounds that wait,
        # so the worst case is a clear timeout rather than an indefinite queue,
        # but failing fast here still beats waiting out the whole deadline.
        # Delete this block once the read lock is gone -- keeping both would
        # leave a permanent double read of the same row.
        preflight = await self.repository.get_kb(owner, kb_id)
        self.assert_index_config(preflight)
        if preflight["status"] != "ready":
            raise Unavailable(f"knowledge base is {preflight['status']}")
        if (
            preflight["revision"] != expected_revision
            or preflight["epoch"] != expected_epoch
        ):
            raise Conflict("knowledge base scope changed")
        # The deadline covers waiting for the lock, not just the engine call. An
        # indexing job holds the write lock for its whole run, so a deadline
        # scoped to the engine alone left the caller queued behind it with no
        # bound of its own -- the mutation timeout, not this one, decided how
        # long a reader waited.
        async with asyncio.timeout(self.query_timeout_seconds):
            async with self.locks.read(kb_id):
                kb = await self.repository.get_kb(owner, kb_id)
                self.assert_index_config(kb)
                if kb["status"] != "ready":
                    raise Unavailable(f"knowledge base is {kb['status']}")
                if kb["revision"] != expected_revision or kb["epoch"] != expected_epoch:
                    raise Conflict("knowledge base scope changed")
                result = await self.engine.query(kb["workspace"], query)
                after = await self.repository.get_kb(owner, kb_id)
                if after["status"] != "ready" or after["epoch"] != expected_epoch:
                    raise Conflict("knowledge base changed during query")
                chunks = result.get("chunks", [])
                # Resolve every cited version in one round trip. One statement per
                # chunk put up to 20 sequential queries inside the lock, against a
                # pool of 10 connections shared with the indexing worker.
                wanted: list[uuid.UUID] = []
                for chunk in chunks:
                    try:
                        wanted.append(uuid.UUID(str(chunk["full_doc_id"])))
                    except (KeyError, TypeError, ValueError):
                        # One malformed id from the index must not fail the whole
                        # query; that chunk simply resolves to no live source below.
                        continue
                sources: dict[str, dict[str, Any]] = {}
                if wanted:
                    rows = await self.repository.fetch(
                        "SELECT v.*,d.name,d.id AS document_id,d.source_url,d.source_fetched_at "
                        "FROM sl_source_versions v "
                        "JOIN sl_documents d ON d.id=v.document_id "
                        "WHERE v.id=ANY($1::uuid[]) AND v.knowledge_base_id=$2 "
                        "AND v.status='active' "
                        "AND d.status='active' AND d.current_version_id=v.id",
                        list(dict.fromkeys(wanted)),
                        uuid.UUID(kb_id),
                    )
                    sources = {str(row["id"]): row for row in rows}
                evidence = []
                for chunk in chunks:
                    try:
                        version_key = str(uuid.UUID(str(chunk["full_doc_id"])))
                    except (KeyError, TypeError, ValueError):
                        continue
                    source = sources.get(version_key)
                    if source is None:
                        continue
                    text = chunk.get("content", "")
                    locator = None
                    blocks = source["parsed_blocks"]
                    if isinstance(blocks, str):
                        blocks = json.loads(blocks)
                    for block in blocks:
                        if text and text in block["text"]:
                            locator = block.get("metadata") or None
                            break
                    item = {
                        "evidence_id": f"kb:{kb_id}:{chunk['chunk_id']}",
                        "kind": "kb_chunk",
                        "knowledge_base_id": kb_id,
                        "document_id": str(source["document_id"]),
                        "source_version_id": str(source["id"]),
                        "chunk_id": chunk["chunk_id"],
                        "title": source["name"],
                        "snippet": text[:500],
                        "text": text,
                        # Provenance tier, not a truth claim: material the user
                        # uploaded is more accountable than a page this deployment
                        # later ingested from the public web.
                        "origin": "web_import" if source.get("source_url") else "user_upload",
                    }
                    if source.get("source_url"):
                        item["source_url"] = source["source_url"]
                    if source.get("source_fetched_at"):
                        item["source_fetched_at"] = source["source_fetched_at"].isoformat()
                    if locator:
                        item["locator"] = locator
                    evidence.append(item)
                return {
                    "scope": {"knowledge_base_id": kb_id, "revision": kb["revision"], "epoch": kb["epoch"]},
                    "evidence": evidence,
                    "entities": result.get("entities", []),
                    "relationships": result.get("relationships", []),
                }

    async def graph(
        self,
        owner: str,
        kb_id: str,
        search: str | None,
        node_limit: int,
        edge_limit: int,
        focus_id: str | None = None,
    ) -> dict[str, Any]:
        if node_limit > 200 or edge_limit > 400:
            raise ValueError("graph limits exceed 200 nodes or 400 edges")
        async with self.locks.read(kb_id):
            kb = await self.repository.get_kb(owner, kb_id)
            self.assert_index_config(kb)
            if kb["status"] != "ready":
                raise Unavailable(f"knowledge base is {kb['status']}")
            if focus_id:
                focused = await self.repository.fetchrow(
                    "SELECT label FROM sl_entities WHERE id=$1 AND knowledge_base_id=$2 "
                    "AND NOT tombstoned",
                    uuid.UUID(focus_id),
                    uuid.UUID(kb_id),
                )
                if focused is None:
                    raise KeyError(focus_id)
                search = focused["label"]
            raw = await self.engine.graph(kb["workspace"], search, node_limit)
            raw_nodes = raw.get("nodes", [])[:node_limit]
            raw_edges = raw.get("edges", [])[:edge_limit]
            candidates = {
                candidate
                for item in [*raw_nodes, *raw_edges]
                for candidate in item.get("source_version_ids", [])
            }
            candidate_ids = []
            for candidate in candidates:
                try:
                    candidate_ids.append(uuid.UUID(candidate))
                except ValueError:
                    continue
            verified_sources = {
                str(row["id"])
                for row in await self.repository.fetch(
                    "SELECT v.id FROM sl_source_versions v JOIN sl_documents d ON d.id=v.document_id "
                    "WHERE v.id=ANY($1::uuid[]) AND v.knowledge_base_id=$2 AND v.status='active' "
                    "AND d.status='active' AND d.current_version_id=v.id",
                    candidate_ids,
                    uuid.UUID(kb_id),
                )
            }
            nodes, by_label = await self._canonical_nodes(
                uuid.UUID(kb_id), raw_nodes, verified_sources
            )
            edges = await self._canonical_edges(
                uuid.UUID(kb_id), raw_edges, by_label, verified_sources
            )
            return {
                "nodes": nodes,
                "edges": edges,
                "truncated": bool(raw.get("truncated") or len(raw.get("nodes", [])) > node_limit or len(raw.get("edges", [])) > edge_limit),
                "revision": kb["revision"],
                "epoch": kb["epoch"],
            }

    async def _canonical_nodes(
        self,
        kb_id: uuid.UUID,
        raw_nodes: list[dict[str, Any]],
        verified_sources: set[str],
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        nodes: list[dict[str, Any]] = []
        by_label: dict[str, dict[str, Any]] = {}
        emitted: set[uuid.UUID] = set()
        for raw in raw_nodes:
            raw_label = str(raw.get("id") or raw.get("label") or "").strip()
            if not raw_label:
                continue
            canonical = await self.repository.fetchrow(
                "SELECT e.* FROM sl_entity_aliases a JOIN sl_entities e ON e.id=a.entity_id "
                "WHERE a.knowledge_base_id=$1 AND a.label=$2",
                kb_id,
                raw_label,
            )
            if canonical is None:
                canonical = await self.repository.fetchrow(
                    "INSERT INTO sl_entities(id,knowledge_base_id,label) VALUES($1,$2,$3) "
                    "ON CONFLICT(knowledge_base_id,label) DO UPDATE SET label=EXCLUDED.label RETURNING *",
                    uuid.uuid4(),
                    kb_id,
                    raw_label,
                )
                await self.repository.pool.execute(
                    "INSERT INTO sl_entity_aliases(knowledge_base_id,label,entity_id) "
                    "VALUES($1,$2,$3) ON CONFLICT(knowledge_base_id,label) DO NOTHING",
                    kb_id,
                    raw_label,
                    canonical["id"],
                )
                owner = await self.repository.fetchrow(
                    "SELECT e.* FROM sl_entity_aliases a JOIN sl_entities e ON e.id=a.entity_id "
                    "WHERE a.knowledge_base_id=$1 AND a.label=$2",
                    kb_id,
                    raw_label,
                )
                if owner["id"] != canonical["id"]:
                    await self.repository.pool.execute(
                        "DELETE FROM sl_entities e WHERE e.id=$1 AND NOT EXISTS "
                        "(SELECT 1 FROM sl_entity_aliases a WHERE a.entity_id=e.id)",
                        canonical["id"],
                    )
                    canonical = owner
            by_label[raw_label] = canonical
            if canonical["tombstoned"] or canonical["id"] in emitted:
                continue
            emitted.add(canonical["id"])
            aliases = canonical["aliases"]
            if isinstance(aliases, str):
                aliases = json.loads(aliases)
            nodes.append(
                {
                    "id": str(canonical["id"]),
                    "label": canonical["label"],
                    "aliases": aliases,
                    "type": (raw.get("labels") or [None])[0],
                    "properties": raw.get("properties", {}),
                    "source_version_ids": [
                        value
                        for value in raw.get("source_version_ids", [])
                        if value in verified_sources
                    ],
                }
            )
        return nodes, by_label

    async def _canonical_edges(
        self,
        kb_id: uuid.UUID,
        raw_edges: list[dict[str, Any]],
        by_label: dict[str, dict[str, Any]],
        verified_sources: set[str],
    ) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        for raw in raw_edges:
            source_label, target_label = str(raw.get("source", "")), str(raw.get("target", ""))
            source, target = by_label.get(source_label), by_label.get(target_label)
            if not source or not target or source["tombstoned"] or target["tombstoned"]:
                continue
            edge = await self.repository.fetchrow(
                "INSERT INTO sl_edges(id,knowledge_base_id,source_entity_id,target_entity_id,"
                "source_label,target_label) VALUES($1,$2,$3,$4,$5,$6) "
                "ON CONFLICT(knowledge_base_id,source_entity_id,target_entity_id) DO UPDATE SET "
                "source_label=EXCLUDED.source_label,target_label=EXCLUDED.target_label "
                "RETURNING *",
                uuid.uuid4(),
                kb_id,
                source["id"],
                target["id"],
                source["label"],
                target["label"],
            )
            await self.repository.pool.execute(
                "INSERT INTO sl_edge_aliases(knowledge_base_id,alias_id,edge_id) "
                "VALUES($1,$2,$2) ON CONFLICT(knowledge_base_id,alias_id) DO NOTHING",
                kb_id,
                edge["id"],
            )
            if edge["tombstoned"]:
                continue
            edges.append(
                {
                    "id": str(edge["id"]),
                    "source": str(source["id"]),
                    "target": str(target["id"]),
                    "type": raw.get("type"),
                    "properties": raw.get("properties", {}),
                    "source_version_ids": [
                        value
                        for value in raw.get("source_version_ids", [])
                        if value in verified_sources
                    ],
                }
            )
        return edges

    async def create_correction(
        self,
        owner: str,
        kb_id: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
    ) -> dict[str, str]:
        kind = payload.get("kind")
        if kind not in {"rename_entity", "merge_entities", "delete_entity", "delete_relation"}:
            raise ValueError("invalid correction kind")
        expected_revision = int(payload["expected_revision"])
        request_hash = _request_hash("correction", {"kb_id": kb_id, "correction": payload})
        replay = await self.repository.replay_idempotency(
            owner, idempotency_key, request_hash
        )
        if replay is not None:
            return replay
        job_id = str(uuid.uuid4())
        correction_id = uuid.uuid4()
        job_payload: dict[str, Any] = {"kind": kind, "correction_id": str(correction_id)}

        async def setup(connection):
            engine_payload: dict[str, Any]
            if kind == "rename_entity":
                entity = await connection.fetchrow(
                    "SELECT * FROM sl_entities WHERE id=$1 AND knowledge_base_id=$2 AND NOT tombstoned",
                    uuid.UUID(payload["entity_id"]),
                    uuid.UUID(kb_id),
                )
                if entity is None:
                    raise KeyError(payload["entity_id"])
                collision = await connection.fetchrow(
                    "SELECT entity_id FROM sl_entity_aliases WHERE knowledge_base_id=$1 "
                    "AND label=$2 AND entity_id!=$3",
                    uuid.UUID(kb_id),
                    payload["label"],
                    entity["id"],
                )
                if collision:
                    raise Conflict("entity label already exists; use an explicit merge")
                aliases = entity["aliases"]
                if isinstance(aliases, str):
                    aliases = json.loads(aliases)
                aliases = list(dict.fromkeys([*aliases, entity["label"]]))
                await connection.execute(
                    "UPDATE sl_entities SET label=$2,aliases=$3::jsonb WHERE id=$1",
                    entity["id"],
                    payload["label"],
                    json.dumps(aliases),
                )
                await connection.execute(
                    "INSERT INTO sl_entity_aliases(knowledge_base_id,label,entity_id) "
                    "VALUES($1,$2,$3) ON CONFLICT(knowledge_base_id,label) DO NOTHING",
                    uuid.UUID(kb_id),
                    payload["label"],
                    entity["id"],
                )
                await connection.execute(
                    "UPDATE sl_edges SET source_label=$2 WHERE knowledge_base_id=$1 "
                    "AND source_entity_id=$3",
                    uuid.UUID(kb_id),
                    payload["label"],
                    entity["id"],
                )
                await connection.execute(
                    "UPDATE sl_edges SET target_label=$2 WHERE knowledge_base_id=$1 "
                    "AND target_entity_id=$3",
                    uuid.UUID(kb_id),
                    payload["label"],
                    entity["id"],
                )
                engine_payload = {
                    "entity_label": entity["label"],
                    "label": payload["label"],
                }
                if payload["label"] in aliases:
                    engine_payload["target_is_owned_alias"] = True
            elif kind == "merge_entities":
                source_ids = [uuid.UUID(value) for value in payload["entity_ids"]]
                target_id = uuid.UUID(payload["target_id"])
                if (
                    not source_ids
                    or target_id in source_ids
                    or len(source_ids) != len(set(source_ids))
                ):
                    raise Conflict("merge aliases must be acyclic and distinct")
                target = await connection.fetchrow(
                    "SELECT * FROM sl_entities WHERE id=$1 AND knowledge_base_id=$2 AND NOT tombstoned",
                    target_id,
                    uuid.UUID(kb_id),
                )
                sources = await connection.fetch(
                    "SELECT * FROM sl_entities WHERE id=ANY($1::uuid[]) AND knowledge_base_id=$2 "
                    "AND NOT tombstoned",
                    source_ids,
                    uuid.UUID(kb_id),
                )
                if target is None or len(sources) != len(source_ids):
                    raise KeyError("merge entity")
                source_labels = [row["label"] for row in sources]
                owned_aliases = await connection.fetch(
                    "SELECT label FROM sl_entity_aliases WHERE knowledge_base_id=$1 "
                    "AND entity_id=ANY($2::uuid[])",
                    uuid.UUID(kb_id),
                    [*source_ids, target_id],
                )
                target_aliases = [
                    row["label"] for row in owned_aliases if row["label"] != target["label"]
                ]
                await connection.execute(
                    "UPDATE sl_entities SET aliases=$2::jsonb WHERE id=$1",
                    target_id,
                    json.dumps(target_aliases),
                )
                await connection.execute(
                    "UPDATE sl_entities SET tombstoned=true WHERE id=ANY($1::uuid[])", source_ids
                )
                await connection.execute(
                    "UPDATE sl_entity_aliases SET entity_id=$3 WHERE knowledge_base_id=$1 "
                    "AND entity_id=ANY($2::uuid[])",
                    uuid.UUID(kb_id),
                    source_ids,
                    target_id,
                )
                affected_edges = await connection.fetch(
                    "SELECT * FROM sl_edges WHERE knowledge_base_id=$1 AND "
                    "(source_entity_id=ANY($2::uuid[]) OR target_entity_id=ANY($2::uuid[])) "
                    "ORDER BY id",
                    uuid.UUID(kb_id),
                    source_ids,
                )
                for old_edge in affected_edges:
                    desired_source = (
                        target_id
                        if old_edge["source_entity_id"] in source_ids
                        else old_edge["source_entity_id"]
                    )
                    desired_target = (
                        target_id
                        if old_edge["target_entity_id"] in source_ids
                        else old_edge["target_entity_id"]
                    )
                    labels = await connection.fetchrow(
                        "SELECT s.label AS source_label,t.label AS target_label "
                        "FROM sl_entities s CROSS JOIN sl_entities t WHERE s.id=$1 AND t.id=$2",
                        desired_source,
                        desired_target,
                    )
                    survivor = await connection.fetchrow(
                        "SELECT * FROM sl_edges WHERE knowledge_base_id=$1 "
                        "AND source_entity_id=$2 AND target_entity_id=$3",
                        uuid.UUID(kb_id),
                        desired_source,
                        desired_target,
                    )
                    if survivor and survivor["id"] != old_edge["id"]:
                        await connection.execute(
                            "UPDATE sl_edges SET tombstoned=(tombstoned OR $2),"
                            "source_label=$3,target_label=$4 WHERE id=$1",
                            survivor["id"],
                            old_edge["tombstoned"],
                            labels["source_label"],
                            labels["target_label"],
                        )
                        await connection.execute(
                            "UPDATE sl_edge_aliases SET edge_id=$2 WHERE edge_id=$1",
                            old_edge["id"],
                            survivor["id"],
                        )
                        await connection.execute("DELETE FROM sl_edges WHERE id=$1", old_edge["id"])
                    else:
                        await connection.execute(
                            "UPDATE sl_edges SET source_entity_id=$2,target_entity_id=$3,"
                            "source_label=$4,target_label=$5 WHERE id=$1",
                            old_edge["id"],
                            desired_source,
                            desired_target,
                            labels["source_label"],
                            labels["target_label"],
                        )
                engine_payload = {"source_labels": source_labels, "target_label": target["label"]}
            elif kind == "delete_entity":
                entity = await connection.fetchrow(
                    "SELECT * FROM sl_entities WHERE id=$1 AND knowledge_base_id=$2",
                    uuid.UUID(payload["entity_id"]),
                    uuid.UUID(kb_id),
                )
                if entity is None:
                    raise KeyError(payload["entity_id"])
                await connection.execute("UPDATE sl_entities SET tombstoned=true WHERE id=$1", entity["id"])
                engine_payload = {"entity_label": entity["label"]}
            else:
                edge = await connection.fetchrow(
                    "SELECT e.* FROM sl_edge_aliases a JOIN sl_edges e ON e.id=a.edge_id "
                    "WHERE a.alias_id=$1 AND a.knowledge_base_id=$2",
                    uuid.UUID(payload["edge_id"]),
                    uuid.UUID(kb_id),
                )
                if edge is None:
                    raise KeyError(payload["edge_id"])
                await connection.execute("UPDATE sl_edges SET tombstoned=true WHERE id=$1", edge["id"])
                engine_payload = {
                    "source_label": edge["source_label"],
                    "target_label": edge["target_label"],
                }
            sequence = await connection.fetchval(
                "SELECT COALESCE(max(sequence),0)+1 FROM sl_corrections WHERE knowledge_base_id=$1",
                uuid.UUID(kb_id),
            )
            stored = {"engine_payload": engine_payload, "request": payload}
            await connection.execute(
                "INSERT INTO sl_corrections(id,knowledge_base_id,sequence,kind,payload) "
                "VALUES($1,$2,$3,$4,$5::jsonb)",
                correction_id,
                uuid.UUID(kb_id),
                sequence,
                kind,
                json.dumps(stored),
            )
            job_payload["engine_payload"] = engine_payload

        async with self.locks.write(kb_id):
            if kind == "rename_entity":
                kb = await self.repository.get_kb(owner, kb_id)
                self.assert_index_config(kb)
                if kb["revision"] != expected_revision or kb["status"] != "ready":
                    raise Conflict("knowledge base revision or status changed")
                entity_id = uuid.UUID(payload["entity_id"])
                source = await self.repository.fetchrow(
                    "SELECT id,label FROM sl_entities WHERE id=$1 AND knowledge_base_id=$2 "
                    "AND NOT tombstoned",
                    entity_id,
                    uuid.UUID(kb_id),
                )
                if source is None:
                    raise KeyError(payload["entity_id"])
                alias_owner = await self.repository.fetchrow(
                    "SELECT entity_id FROM sl_entity_aliases WHERE knowledge_base_id=$1 "
                    "AND label=$2",
                    uuid.UUID(kb_id),
                    payload["label"],
                )
                if alias_owner and alias_owner["entity_id"] != entity_id:
                    raise Conflict("entity label already exists; use an explicit merge")
                if payload["label"] == source["label"]:
                    raise Conflict("entity already has the requested label")
                target_exists = await self.engine.entity_exists(kb["workspace"], payload["label"])
                if target_exists and alias_owner is None:
                    raise Conflict("entity label already exists; use an explicit merge")
            return await self.repository.begin_job(
                owner=owner,
                kb_id=kb_id,
                expected_revision=expected_revision,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                job_id=job_id,
                operation="correction",
                payload=job_payload,
                setup=setup,
                index_config_hash=self.index_hash,
            )

    async def get_source(
        self, owner: str, kb_id: str, version_id: str, max_chars: int = 20_000
    ) -> dict[str, Any]:
        await self.repository.get_kb(owner, kb_id)
        row = await self.repository.fetchrow(
            "SELECT v.*,d.id AS document_id,d.name,d.status AS document_status,d.current_version_id,"
            "d.source_url,d.source_fetched_at FROM sl_source_versions v "
            "JOIN sl_documents d ON d.id=v.document_id "
            "WHERE v.id=$1 AND v.knowledge_base_id=$2",
            uuid.UUID(version_id),
            uuid.UUID(kb_id),
        )
        if row is None:
            raise KeyError(version_id)
        parsed_blocks = row["parsed_blocks"]
        if isinstance(parsed_blocks, str):
            parsed_blocks = json.loads(parsed_blocks)
        bounded_blocks, blocks_truncated = _bounded_source_blocks(parsed_blocks, max_chars)
        if row["document_status"] == "deleted" or row["status"] == "deleted":
            source_status = "deleted"
        elif row["status"] == "active" and row["id"] == row.get("current_version_id"):
            source_status = "current"
        else:
            source_status = "historical"
        return {
            "source_version_id": version_id,
            "document_id": str(row["document_id"]),
            "title": row["name"],
            "status": row["document_status"],
            "source_status": source_status,
            "content_hash": row["content_hash"],
            "text": row["parsed_text"][:max_chars],
            "parsed_blocks": _json_value(bounded_blocks),
            "truncated": len(row["parsed_text"]) > max_chars or blocks_truncated,
            "total_blocks": len(parsed_blocks),
            "returned_blocks": len(bounded_blocks),
            "source_url": row["source_url"],
            "source_fetched_at": _json_value(row["source_fetched_at"]),
        }

    async def create_web_snapshot(self, owner: str, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot_id = uuid.uuid4()
        actual_hash = hashlib.sha256(payload["text"].encode()).hexdigest()
        if not secrets.compare_digest(actual_hash, payload["content_hash"]):
            raise ValueError("web snapshot content_hash does not match text")
        fetched = datetime.fromisoformat(payload["fetched_at"])
        expires = datetime.fromisoformat(payload["expires_at"])
        row = await self.repository.fetchrow(
            "INSERT INTO sl_web_snapshots(id,owner,session_id,url,title,body,content_hash,fetched_at,expires_at) "
            "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING *",
            snapshot_id,
            owner,
            payload["session_id"],
            payload["url"],
            payload["title"],
            payload["text"],
            payload["content_hash"],
            fetched,
            expires,
        )
        return _json_value({"id": row["id"], **payload})

    async def get_web_snapshot(
        self, owner: str, snapshot_id: str, session_id: str
    ) -> dict[str, Any]:
        row = await self.repository.fetchrow(
            "SELECT * FROM sl_web_snapshots WHERE id=$1 AND owner=$2 AND session_id=$3",
            uuid.UUID(snapshot_id),
            owner,
            session_id,
        )
        if row is None:
            raise KeyError(snapshot_id)
        if row["expires_at"] <= datetime.now(timezone.utc):
            raise Expired("web snapshot expired")
        row["text"] = row.pop("body")
        return _json_value(row)

    async def import_web_snapshot(
        self,
        owner: str,
        kb_id: str,
        snapshot_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
    ) -> dict[str, str]:
        row = await self.repository.fetchrow(
            "SELECT * FROM sl_web_snapshots WHERE id=$1 AND owner=$2",
            uuid.UUID(snapshot_id),
            owner,
        )
        if row is None:
            raise KeyError(snapshot_id)
        snapshot = _json_value({**row, "text": row["body"]})
        body = snapshot["text"].encode()
        material_payload = {
            "name": snapshot["title"],
            "kind": "web",
            "content_base64": base64.b64encode(body).decode(),
            "parsed_blocks": [
                {"text": snapshot["text"], "metadata": {"url": snapshot["url"]}}
            ],
            "expected_revision": expected_revision,
            "source_url": snapshot["url"],
            "source_fetched_at": snapshot["fetched_at"],
        }
        replay = await self.repository.replay_idempotency(
            owner,
            key=idempotency_key,
            request_hash=_material_request_hash(kb_id, material_payload, snapshot_id=snapshot_id),
        )
        if replay is not None:
            return replay
        if row["expires_at"] <= datetime.now(timezone.utc):
            raise Expired("web snapshot expired")
        response = await self.ingest_document(
            owner,
            kb_id,
            material_payload,
            idempotency_key=idempotency_key,
            snapshot_id=snapshot_id,
        )
        await self.repository.pool.execute(
            "UPDATE sl_web_snapshots SET imported_at=COALESCE(imported_at,now()) "
            "WHERE id=$1 AND owner=$2",
            uuid.UUID(snapshot_id),
            owner,
        )
        return response

    async def get_job(self, owner: str, job_id: str) -> dict[str, Any]:
        row = await self.repository.fetchrow(
            "SELECT id,knowledge_base_id,status,stage,revision,error_code FROM sl_jobs "
            "WHERE id=$1 AND owner=$2",
            uuid.UUID(job_id),
            owner,
        )
        if row is None:
            raise KeyError(job_id)
        return _json_value(row)

    async def retry_job(
        self, owner: str, job_id: str, expected_revision: int
    ) -> dict[str, str]:
        failed = await self.repository.fetchrow(
            "SELECT * FROM sl_jobs WHERE id=$1 AND owner=$2 AND status='failed'",
            uuid.UUID(job_id),
            owner,
        )
        if failed is None:
            raise Conflict("only failed jobs can be retried")
        payload = failed["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        new_id = str(uuid.uuid4())
        kb_id = str(failed["knowledge_base_id"])

        async def setup(_connection):
            return None

        async with self.locks.write(kb_id):
            return await self.repository.begin_job(
                owner=owner,
                kb_id=kb_id,
                expected_revision=expected_revision,
                idempotency_key=f"retry:{job_id}:{expected_revision}",
                request_hash=_request_hash("retry", {"job_id": job_id, "expected_revision": expected_revision}),
                job_id=new_id,
                operation=failed["operation"],
                payload=payload,
                setup=setup,
                allow_dirty=True,
                index_config_hash=self.index_hash,
            )

    async def create_rebuild_job(
        self, owner: str, kb_id: str, expected_revision: int, idempotency_key: str
    ) -> dict[str, str]:
        async def setup(_connection):
            return None

        async with self.locks.write(kb_id):
            return await self.repository.begin_job(
                owner=owner,
                kb_id=kb_id,
                expected_revision=expected_revision,
                idempotency_key=idempotency_key,
                request_hash=_request_hash("rebuild", {"kb_id": kb_id, "expected_revision": expected_revision}),
                job_id=str(uuid.uuid4()),
                operation="rebuild",
                payload={},
                setup=setup,
                allow_dirty=True,
                index_config_hash=self.index_hash,
                allow_config_mismatch=True,
            )
