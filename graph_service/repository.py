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
-- Every snapshot that fed a document, including an import that reused an existing
-- document with the same text.
CREATE TABLE IF NOT EXISTS sl_document_snapshots (
    document_id uuid NOT NULL REFERENCES sl_documents(id) ON DELETE CASCADE,
    snapshot_id uuid NOT NULL,
    PRIMARY KEY(document_id, snapshot_id)
);
CREATE INDEX IF NOT EXISTS sl_document_snapshots_snapshot_idx
ON sl_document_snapshots(snapshot_id);
-- Foreign-key lookups for deletion cascades and the purge.
CREATE INDEX IF NOT EXISTS sl_documents_kb_idx ON sl_documents(knowledge_base_id);
CREATE INDEX IF NOT EXISTS sl_versions_document_idx ON sl_source_versions(document_id);
CREATE INDEX IF NOT EXISTS sl_jobs_kb_idx ON sl_jobs(knowledge_base_id);
CREATE INDEX IF NOT EXISTS sl_entity_aliases_entity_idx ON sl_entity_aliases(entity_id);
CREATE INDEX IF NOT EXISTS sl_edge_aliases_edge_idx ON sl_edge_aliases(edge_id);
-- Pages imported before links were recorded: every imported snapshot of the owner
-- with the text of one of the document's versions. Over-linking only delays a
-- clear until every linked document is deleted.
INSERT INTO sl_document_snapshots(document_id,snapshot_id)
SELECT DISTINCT v.document_id,s.id FROM sl_source_versions v
JOIN sl_documents d ON d.id=v.document_id AND d.kind='web'
JOIN sl_knowledge_bases k ON k.id=v.knowledge_base_id
JOIN sl_web_snapshots s ON s.owner=k.owner AND s.content_hash=v.content_hash
WHERE s.imported_at IS NOT NULL AND v.content_hash!=''
AND NOT EXISTS (SELECT 1 FROM sl_document_snapshots x WHERE x.document_id=d.id)
ON CONFLICT DO NOTHING;
-- Documents deleted before deletion cleared their content; each a no-op once applied.
UPDATE sl_web_snapshots s SET body='',url='',title='' FROM sl_document_snapshots l
JOIN sl_documents d ON d.id=l.document_id
WHERE d.status='deleted' AND s.id=l.snapshot_id AND (s.body!='' OR s.url!='')
AND NOT EXISTS (SELECT 1 FROM sl_document_snapshots l2 JOIN sl_documents o ON o.id=l2.document_id
                WHERE l2.snapshot_id=s.id AND o.status!='deleted');
UPDATE sl_source_versions v SET parsed_text='',parsed_blocks='[]'::jsonb,content_hash=''
FROM sl_documents d WHERE d.id=v.document_id AND d.status='deleted' AND v.parsed_text!='';
UPDATE sl_documents SET source_url=NULL,source_fetched_at=NULL,legacy_document_id=NULL
WHERE status='deleted'
AND (source_url IS NOT NULL OR source_fetched_at IS NOT NULL OR legacy_document_id IS NOT NULL);
"""


# An imported page's snapshot keeps its body past expiry so an import can replay
# idempotently, which makes it a second copy of the document. It is cleared once no
# live document was imported from it; the same page in another knowledge base is
# another document and keeps it.
_BLANK_LINKED_SNAPSHOTS = (
    "UPDATE sl_web_snapshots s SET body='',url='',title='' FROM sl_document_snapshots l "
    "JOIN sl_documents d ON d.id=l.document_id "
    "WHERE d.{scope}=$1 AND s.id=l.snapshot_id AND (s.body!='' OR s.url!='') "
    "AND NOT EXISTS (SELECT 1 FROM sl_document_snapshots l2 JOIN sl_documents o "
    "ON o.id=l2.document_id WHERE l2.snapshot_id=s.id AND o.status!='deleted' "
    "AND o.id!=d.id)"
)

# 'deleting': the index is gone and the purge below has not finished yet.
# 'deleted': a tombstone kept only so the delete job stays observable.
DELETED_KB_STATUSES = frozenset({"deleting", "deleted"})


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
        if row is None or row["status"] in DELETED_KB_STATUSES:
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
            if kb is None or kb["status"] in DELETED_KB_STATUSES:
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
            if job["operation"] == "delete_document" and document_id:
                # Every version of the document, not only the one just removed from
                # the index. The row stays so old citations still resolve to
                # "deleted"; the original file goes in WriteWorker.purge_deleted_data.
                await connection.execute(
                    _BLANK_LINKED_SNAPSHOTS.format(scope="id"), document_id
                )
                await connection.execute(
                    "UPDATE sl_source_versions SET parsed_text='',parsed_blocks='[]'::jsonb,"
                    "content_hash='' WHERE document_id=$1",
                    document_id,
                )
                await connection.execute(
                    "UPDATE sl_documents SET source_url=NULL,source_fetched_at=NULL,"
                    "legacy_document_id=NULL WHERE id=$1",
                    document_id,
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

    async def knowledge_bases_pending_purge(self) -> list[dict[str, Any]]:
        return await self.fetch(
            "SELECT id,workspace FROM sl_knowledge_bases WHERE status='deleting' ORDER BY updated_at"
        )

    async def purge_knowledge_base_rows(self, kb_id) -> None:
        """Delete everything a deleted knowledge base holds, keeping a nameless tombstone.

        The tombstone row and its delete job survive so a client polling that job
        still sees it succeed. Idempotent: a crash part-way is finished by the next sweep.
        """
        async with self.pool.acquire() as connection, connection.transaction():
            kb = await connection.fetchrow(
                "SELECT id,owner FROM sl_knowledge_bases WHERE id=$1 AND status='deleting' "
                "FOR UPDATE",
                kb_id,
            )
            if kb is None:
                return
            await connection.execute(
                _BLANK_LINKED_SNAPSHOTS.format(scope="knowledge_base_id"), kb_id
            )
            # Source versions and alias rows go with these by cascade.
            for table in ("sl_documents", "sl_corrections", "sl_entities", "sl_edges"):
                await connection.execute(
                    f"DELETE FROM {table} WHERE knowledge_base_id=$1", kb_id
                )
            await connection.execute(
                "DELETE FROM sl_jobs WHERE knowledge_base_id=$1 "
                "AND operation!='delete_knowledge_base'",
                kb_id,
            )
            # Create/update replays carry the name and description. Keeping the id
            # alone lets a late retry of the create replay instead of making a new one.
            await connection.execute(
                "UPDATE sl_idempotency SET response=jsonb_build_object('id',response->'id') "
                "WHERE owner=$1 AND response->>'id'=$2",
                kb["owner"],
                str(kb_id),
            )
            await connection.execute(
                "UPDATE sl_knowledge_bases SET name='',description='',updated_at=now() "
                "WHERE id=$1",
                kb_id,
            )

    async def mark_knowledge_base_purged(self, kb_id) -> None:
        await self.pool.execute(
            "UPDATE sl_knowledge_bases SET status='deleted',updated_at=now() "
            "WHERE id=$1 AND status='deleting'",
            kb_id,
        )

    async def knowledge_base_materials(self, kb_id) -> list[str]:
        rows = await self.fetch(
            "SELECT material_path FROM sl_source_versions "
            "WHERE knowledge_base_id=$1 AND material_path!=''",
            kb_id,
        )
        return [row["material_path"] for row in rows]

    async def document_materials(self, document_id) -> list[dict[str, Any]]:
        return await self.fetch(
            "SELECT id,material_path FROM sl_source_versions "
            "WHERE document_id=$1 AND material_path!=''",
            document_id,
        )

    async def deleted_materials(self, after=None, limit: int = 100) -> list[dict[str, Any]]:
        # Keyset pages, so files that keep failing to unlink cannot starve the rest.
        return await self.fetch(
            "SELECT v.id,v.material_path FROM sl_source_versions v JOIN sl_documents d "
            "ON d.id=v.document_id WHERE d.status='deleted' AND v.material_path!='' "
            "AND ($1::uuid IS NULL OR v.id>$1) ORDER BY v.id LIMIT $2",
            after,
            limit,
        )

    async def drop_orphan_extraction_cache(self, workspace: str) -> int:
        """Delete LightRAG extraction-cache entries no live chunk still uses.

        Their prompts embed chunk text, so a replaced or deleted source is not gone
        while they remain. Computed from what is stored rather than remembered by the
        job, so it also finishes after a failed job's retry and for data left by older
        releases. Reads LightRAG's PostgreSQL tables directly (lightrag-hku 1.5.7,
        pinned by the index configuration hash); run only on a consistent workspace.
        """
        if await self.pool.fetchval("SELECT to_regclass('lightrag_llm_cache')") is None:
            return 0
        async with self.pool.acquire() as connection, connection.transaction():
            # Bounded: the worker awaits this between jobs. A timeout rolls back and
            # the next job or sweep tries again.
            await connection.execute("SET LOCAL statement_timeout = '30s'")
            result = await connection.execute(
                "WITH live AS (SELECT DISTINCT e.id FROM lightrag_doc_chunks k "
                "CROSS JOIN LATERAL jsonb_array_elements_text(k.llm_cache_list) AS e(id) "
                "WHERE k.workspace=$1 AND jsonb_typeof(k.llm_cache_list)='array') "
                "DELETE FROM lightrag_llm_cache c WHERE c.workspace=$1 "
                "AND c.cache_type='extract' "
                # A dirty workspace may be mid-rebuild with chunks not yet
                # re-inserted; its retry needs the cache it would lose here.
                "AND EXISTS (SELECT 1 FROM sl_knowledge_bases b "
                "WHERE b.workspace=$1 AND b.status='ready') "
                "AND NOT EXISTS (SELECT 1 FROM lightrag_doc_chunks k "
                "WHERE k.workspace=$1 AND k.id=c.chunk_id) "
                "AND NOT EXISTS (SELECT 1 FROM live WHERE live.id=c.id)",
                workspace,
            )
        return int(result.rsplit(" ", 1)[-1])

    async def ready_workspaces(self) -> list[str]:
        rows = await self.fetch("SELECT workspace FROM sl_knowledge_bases WHERE status='ready'")
        return [row["workspace"] for row in rows]

    async def mark_materials_purged(self, version_ids: list) -> None:
        await self.pool.execute(
            "UPDATE sl_source_versions SET material_path='' WHERE id=ANY($1::uuid[])",
            version_ids,
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
