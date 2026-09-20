from __future__ import annotations

import json
from typing import Any

import asyncpg

from .errors import Conflict, Unavailable


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sl_knowledge_bases (
    id uuid PRIMARY KEY, owner text NOT NULL, workspace text NOT NULL UNIQUE,
    name text NOT NULL, description text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'ready', revision bigint NOT NULL DEFAULT 0,
    epoch bigint NOT NULL DEFAULT 0, index_config_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS sl_kb_owner_idx ON sl_knowledge_bases(owner, created_at);
CREATE TABLE IF NOT EXISTS sl_documents (
    id uuid PRIMARY KEY, knowledge_base_id uuid NOT NULL REFERENCES sl_knowledge_bases(id) ON DELETE CASCADE,
    name text NOT NULL, kind text NOT NULL, current_version_id uuid,
    status text NOT NULL DEFAULT 'pending', legacy_document_id text,
    source_url text, source_fetched_at timestamptz, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS sl_source_versions (
    id uuid PRIMARY KEY, document_id uuid NOT NULL REFERENCES sl_documents(id) ON DELETE CASCADE,
    knowledge_base_id uuid NOT NULL REFERENCES sl_knowledge_bases(id) ON DELETE CASCADE,
    content_hash text NOT NULL, material_path text NOT NULL, parsed_text text NOT NULL,
    parsed_blocks jsonb NOT NULL, source_token text NOT NULL UNIQUE,
    status text NOT NULL DEFAULT 'pending', created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS sl_versions_kb_idx ON sl_source_versions(knowledge_base_id, id);
CREATE TABLE IF NOT EXISTS sl_jobs (
    id uuid PRIMARY KEY, knowledge_base_id uuid NOT NULL REFERENCES sl_knowledge_bases(id) ON DELETE CASCADE,
    owner text NOT NULL, operation text NOT NULL, payload jsonb NOT NULL,
    status text NOT NULL DEFAULT 'queued', stage text NOT NULL DEFAULT 'queued',
    revision bigint, error_code text, parent_job_id uuid,
    created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS sl_jobs_queue_idx ON sl_jobs(status, created_at);
CREATE TABLE IF NOT EXISTS sl_idempotency (
    owner text NOT NULL, key text NOT NULL, request_hash text NOT NULL,
    response jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY(owner, key)
);
CREATE TABLE IF NOT EXISTS sl_corrections (
    id uuid PRIMARY KEY, knowledge_base_id uuid NOT NULL REFERENCES sl_knowledge_bases(id) ON DELETE CASCADE,
    sequence bigint NOT NULL, kind text NOT NULL, payload jsonb NOT NULL,
    status text NOT NULL DEFAULT 'active', created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(knowledge_base_id, sequence)
);
CREATE TABLE IF NOT EXISTS sl_entities (
    id uuid PRIMARY KEY, knowledge_base_id uuid NOT NULL REFERENCES sl_knowledge_bases(id) ON DELETE CASCADE,
    label text NOT NULL, aliases jsonb NOT NULL DEFAULT '[]'::jsonb, tombstoned boolean NOT NULL DEFAULT false,
    UNIQUE(knowledge_base_id, label)
);
CREATE TABLE IF NOT EXISTS sl_entity_aliases (
    knowledge_base_id uuid NOT NULL REFERENCES sl_knowledge_bases(id) ON DELETE CASCADE,
    label text NOT NULL,
    entity_id uuid NOT NULL REFERENCES sl_entities(id) ON DELETE CASCADE,
    PRIMARY KEY(knowledge_base_id, label)
);
INSERT INTO sl_entity_aliases(knowledge_base_id,label,entity_id)
SELECT knowledge_base_id,label,id FROM sl_entities
ON CONFLICT(knowledge_base_id,label) DO NOTHING;
INSERT INTO sl_entity_aliases(knowledge_base_id,label,entity_id)
SELECT e.knowledge_base_id,a.label,e.id FROM sl_entities e
CROSS JOIN LATERAL jsonb_array_elements_text(e.aliases) AS a(label)
ON CONFLICT(knowledge_base_id,label) DO NOTHING;
CREATE TABLE IF NOT EXISTS sl_edges (
    id uuid PRIMARY KEY, knowledge_base_id uuid NOT NULL REFERENCES sl_knowledge_bases(id) ON DELETE CASCADE,
    source_entity_id uuid NOT NULL, target_entity_id uuid NOT NULL,
    source_label text NOT NULL, target_label text NOT NULL, tombstoned boolean NOT NULL DEFAULT false
);
ALTER TABLE sl_edges
DROP CONSTRAINT IF EXISTS sl_edges_knowledge_base_id_source_label_target_label_key;
CREATE UNIQUE INDEX IF NOT EXISTS sl_edges_canonical_endpoints_idx
ON sl_edges(knowledge_base_id,source_entity_id,target_entity_id);
CREATE TABLE IF NOT EXISTS sl_edge_aliases (
    knowledge_base_id uuid NOT NULL REFERENCES sl_knowledge_bases(id) ON DELETE CASCADE,
    alias_id uuid NOT NULL,
    edge_id uuid NOT NULL REFERENCES sl_edges(id) ON DELETE CASCADE,
    PRIMARY KEY(knowledge_base_id,alias_id)
);
INSERT INTO sl_edge_aliases(knowledge_base_id,alias_id,edge_id)
SELECT knowledge_base_id,id,id FROM sl_edges
ON CONFLICT(knowledge_base_id,alias_id) DO NOTHING;
CREATE TABLE IF NOT EXISTS sl_web_snapshots (
    id uuid PRIMARY KEY, owner text NOT NULL, session_id text NOT NULL, url text NOT NULL,
    title text NOT NULL, body text NOT NULL, content_hash text NOT NULL,
    fetched_at timestamptz NOT NULL, expires_at timestamptz NOT NULL,
    imported_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE sl_web_snapshots ADD COLUMN IF NOT EXISTS imported_at timestamptz;
CREATE INDEX IF NOT EXISTS sl_snapshots_scope_idx ON sl_web_snapshots(owner, session_id, id);
"""


def _row(row: asyncpg.Record | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class Repository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def migrate(self) -> None:
        await self.pool.execute(SCHEMA_SQL)

    async def fetchrow(self, query: str, *args) -> dict[str, Any] | None:
        return _row(await self.pool.fetchrow(query, *args))

    async def fetch(self, query: str, *args) -> list[dict[str, Any]]:
        return [dict(row) for row in await self.pool.fetch(query, *args)]

    async def get_kb(self, owner: str, kb_id: str) -> dict[str, Any]:
        row = await self.fetchrow(
            "SELECT * FROM sl_knowledge_bases WHERE id=$1 AND owner=$2", kb_id, owner
        )
        if row is None:
            raise KeyError(kb_id)
        return row

    async def idempotent_mutation(
        self, owner: str, key: str, request_hash: str, mutate
    ) -> dict[str, Any]:
        async with self.pool.acquire() as connection, connection.transaction():
            replay = await connection.fetchrow(
                "SELECT request_hash,response FROM sl_idempotency WHERE owner=$1 AND key=$2",
                owner,
                key,
            )
            if replay:
                if replay["request_hash"] != request_hash:
                    raise Conflict("idempotency key was already used for a different request")
                response = replay["response"]
                return json.loads(response) if isinstance(response, str) else dict(response)
            response = await mutate(connection)
            await connection.execute(
                "INSERT INTO sl_idempotency(owner,key,request_hash,response) VALUES($1,$2,$3,$4::jsonb)",
                owner,
                key,
                request_hash,
                json.dumps(response),
            )
            return response

    async def replay_idempotency(
        self, owner: str, key: str, request_hash: str
    ) -> dict[str, Any] | None:
        row = await self.pool.fetchrow(
            "SELECT request_hash,response FROM sl_idempotency WHERE owner=$1 AND key=$2",
            owner,
            key,
        )
        if row is None:
            return None
        if row["request_hash"] != request_hash:
            raise Conflict("idempotency key was already used for a different request")
        response = row["response"]
        return json.loads(response) if isinstance(response, str) else dict(response)

    async def begin_job(
        self,
        *,
        owner: str,
        kb_id: str,
        expected_revision: int,
        idempotency_key: str,
        request_hash: str,
        job_id: str,
        operation: str,
        payload: dict[str, Any],
        setup,
        allow_dirty: bool = False,
        index_config_hash: str | None = None,
        allow_config_mismatch: bool = False,
    ) -> dict[str, str]:
        async with self.pool.acquire() as connection, connection.transaction():
            replay = await connection.fetchrow(
                "SELECT request_hash, response FROM sl_idempotency WHERE owner=$1 AND key=$2",
                owner,
                idempotency_key,
            )
            if replay:
                if replay["request_hash"] != request_hash:
                    raise Conflict("idempotency key was already used for a different request")
                response = replay["response"]
                return json.loads(response) if isinstance(response, str) else dict(response)
            kb = await connection.fetchrow(
                "SELECT * FROM sl_knowledge_bases WHERE id=$1 AND owner=$2 FOR UPDATE",
                kb_id,
                owner,
            )
            if kb is None:
                raise KeyError(kb_id)
            if kb["revision"] != expected_revision:
                raise Conflict("knowledge base revision changed")
            if (
                index_config_hash is not None
                and kb["index_config_hash"] != index_config_hash
                and not allow_config_mismatch
            ):
                raise Unavailable("knowledge base index configuration changed; rebuild required")
            allowed = {"ready"} | ({"dirty"} if allow_dirty else set())
            if kb["status"] not in allowed:
                raise Conflict(f"knowledge base is {kb['status']}")
            await setup(connection)
            await connection.execute(
                "INSERT INTO sl_jobs(id, knowledge_base_id, owner, operation, payload) "
                "VALUES($1,$2,$3,$4,$5::jsonb)",
                job_id,
                kb_id,
                owner,
                operation,
                json.dumps(payload),
            )
            await connection.execute(
                "UPDATE sl_knowledge_bases SET epoch=epoch+1,status='updating',updated_at=now() "
                "WHERE id=$1",
                kb_id,
            )
            response = {"job_id": job_id, "knowledge_base_id": kb_id}
            await connection.execute(
                "INSERT INTO sl_idempotency(owner,key,request_hash,response) VALUES($1,$2,$3,$4::jsonb)",
                owner,
                idempotency_key,
                request_hash,
                json.dumps(response),
            )
            return response

    async def claim_job(self, connection: asyncpg.Connection) -> dict[str, Any] | None:
        async with connection.transaction():
            row = await connection.fetchrow(
                "SELECT * FROM sl_jobs WHERE status='queued' ORDER BY created_at "
                "FOR UPDATE SKIP LOCKED LIMIT 1"
            )
            if row is None:
                return None
            await connection.execute(
                "UPDATE sl_jobs SET status='running',stage='indexing',updated_at=now() WHERE id=$1",
                row["id"],
            )
            return dict(row)

    async def finish_job(self, job: dict[str, Any]) -> int:
        async with self.pool.acquire() as connection, connection.transaction():
            final_status = "deleting" if job["operation"] == "delete_knowledge_base" else "ready"
            kb = await connection.fetchrow(
                "UPDATE sl_knowledge_bases SET revision=revision+1,status=$2,updated_at=now() "
                "WHERE id=$1 AND status='updating' RETURNING revision",
                job["knowledge_base_id"],
                final_status,
            )
            if kb is None:
                raise RuntimeError("knowledge base left updating state during job")
            revision = kb["revision"]
            payload = job["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            version_id = payload.get("version_id")
            document_id = payload.get("document_id")
            if version_id:
                version_status = "deleted" if job["operation"] == "delete_document" else "active"
                await connection.execute(
                    "UPDATE sl_source_versions SET status=$2 WHERE id=$1",
                    version_id,
                    version_status,
                )
            if job["operation"] == "replace_document" and payload.get("previous_version_id"):
                await connection.execute(
                    "UPDATE sl_source_versions SET status='superseded' WHERE id=$1",
                    payload["previous_version_id"],
                )
            if document_id:
                status = "deleted" if job["operation"] == "delete_document" else "active"
                await connection.execute(
                    "UPDATE sl_documents SET status=$2, current_version_id=COALESCE($3,current_version_id) "
                    "WHERE id=$1",
                    document_id,
                    status,
                    version_id,
                )
            if job["operation"] == "delete_knowledge_base":
                await connection.execute(
                    "UPDATE sl_documents SET status='deleted' WHERE knowledge_base_id=$1",
                    job["knowledge_base_id"],
                )
            if job["operation"] == "rebuild":
                active_versions = payload.get("active_versions", {})
                await connection.execute(
                    "UPDATE sl_source_versions SET status='superseded' "
                    "WHERE knowledge_base_id=$1 AND status!='deleted'",
                    job["knowledge_base_id"],
                )
                for active_document_id, active_version_id in active_versions.items():
                    await connection.execute(
                        "UPDATE sl_documents SET status='active',current_version_id=$2 WHERE id=$1",
                        active_document_id,
                        active_version_id,
                    )
                    await connection.execute(
                        "UPDATE sl_source_versions SET status='active' WHERE id=$1",
                        active_version_id,
                    )
            await connection.execute(
                "UPDATE sl_jobs SET status='succeeded',stage='published',revision=$2,error_code=NULL,"
                "updated_at=now() WHERE id=$1",
                job["id"],
                revision,
            )
            return revision

    async def fail_job(self, job: dict[str, Any], code: str) -> None:
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "UPDATE sl_jobs SET status='failed',stage='failed',error_code=$2,updated_at=now() "
                "WHERE id=$1",
                job["id"],
                code,
            )
            await connection.execute(
                "UPDATE sl_knowledge_bases SET status='dirty',updated_at=now() WHERE id=$1",
                job["knowledge_base_id"],
            )

    async def mark_interrupted_dirty(self) -> None:
        async with self.pool.acquire() as connection, connection.transaction():
            rows = await connection.fetch(
                "UPDATE sl_jobs SET status='failed',stage='failed',error_code='interrupted',updated_at=now() "
                "WHERE status='running' RETURNING knowledge_base_id"
            )
            if rows:
                await connection.execute(
                    "UPDATE sl_knowledge_bases SET status='dirty',updated_at=now() "
                    "WHERE id=ANY($1::uuid[])",
                    [row["knowledge_base_id"] for row in rows],
                )

    async def mark_config_drift_dirty(self, index_config_hash: str) -> None:
        await self.pool.execute(
            "UPDATE sl_knowledge_bases SET status='dirty',epoch=epoch+1,updated_at=now() "
            "WHERE index_config_hash!=$1 AND status='ready'",
            index_config_hash,
        )

    async def cleanup_expired_snapshots(self, limit: int = 100) -> int:
        result = await self.pool.execute(
            "WITH expired AS (SELECT id FROM sl_web_snapshots "
            "WHERE expires_at<=now() AND imported_at IS NULL AND body!='' ORDER BY expires_at "
            "FOR UPDATE SKIP LOCKED LIMIT $1) UPDATE sl_web_snapshots s SET body='' "
            "FROM expired e WHERE s.id=e.id",
            limit,
        )
        return int(result.rsplit(" ", 1)[-1])
