from __future__ import annotations

import base64
import hashlib
import asyncio
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("asyncpg")

from graph_service.errors import Conflict, Expired, Unavailable
from graph_service.engine import LightRAGEngine
from graph_service.materials import MaterialStore
from graph_service.service import KnowledgeService
from graph_service.worker import WriteWorker

from fakes import FakeEngine


pytestmark = pytest.mark.asyncio


def material_payload(text: str = "Alpha marker", *, revision: int = 0) -> dict:
    return {
        "name": "notes.txt",
        "kind": "file",
        "content_base64": base64.b64encode(text.encode()).decode(),
        "parsed_blocks": [{"text": text, "metadata": {"page": 1}}],
        "expected_revision": revision,
    }


async def make_service(repository, settings, engine=None):
    engine = engine or FakeEngine()
    service = KnowledgeService(repository, MaterialStore(settings.materials_dir), engine)
    worker = WriteWorker(repository, service, engine)
    return service, worker, engine


async def test_owner_scope_and_epoch_are_enforced(repository, settings) -> None:
    service, worker, _ = await make_service(repository, settings)
    kb = await service.create_knowledge_base("alice", "Medicine", "private")

    with pytest.raises(KeyError):
        await service.get_knowledge_base("bob", kb["id"])

    queued = await service.ingest_document(
        "alice", kb["id"], material_payload(), idempotency_key="upload-1"
    )
    scope_during_write = await service.get_scope("alice", kb["id"])
    assert scope_during_write == {
        "knowledge_base_id": kb["id"],
        "revision": 0,
        "epoch": 1,
        "status": "updating",
    }
    with pytest.raises(Unavailable):
        await service.query("alice", kb["id"], "Alpha", 0, 1)

    assert await worker.process_one() == queued["job_id"]
    scope = await service.get_scope("alice", kb["id"])
    assert scope == {
        "knowledge_base_id": kb["id"],
        "revision": 1,
        "epoch": 1,
        "status": "ready",
    }


async def test_metadata_writes_are_idempotent_and_advance_scope(repository, settings) -> None:
    service, _, _ = await make_service(repository, settings)
    created = await service.create_knowledge_base(
        "alice", "Original", "", idempotency_key="create-metadata"
    )
    replay = await service.create_knowledge_base(
        "alice", "Original", "", idempotency_key="create-metadata"
    )
    assert replay == created
    updated = await service.update_knowledge_base(
        "alice",
        created["id"],
        {"name": "Renamed", "expected_revision": 0},
        idempotency_key="patch-metadata",
    )
    assert (updated["name"], updated["revision"], updated["epoch"]) == ("Renamed", 1, 1)
    assert await service.update_knowledge_base(
        "alice",
        created["id"],
        {"name": "Renamed", "expected_revision": 0},
        idempotency_key="patch-metadata",
    ) == updated
    with pytest.raises(Conflict):
        await service.update_knowledge_base(
            "alice",
            created["id"],
            {"description": "stale", "expected_revision": 0},
            idempotency_key="stale-patch",
        )


async def test_idempotency_replay_and_conflicting_body(repository, settings) -> None:
    service, _, _ = await make_service(repository, settings)
    kb = await service.create_knowledge_base("alice", "KB", "")
    first = await service.ingest_document(
        "alice", kb["id"], material_payload(), idempotency_key="same-key"
    )
    replay = await service.ingest_document(
        "alice", kb["id"], material_payload(), idempotency_key="same-key"
    )
    assert replay == first

    with pytest.raises(Conflict):
        await service.ingest_document(
            "alice",
            kb["id"],
            material_payload("Different"),
            idempotency_key="same-key",
        )


async def test_same_kb_duplicate_content_reuses_existing_document(repository, settings) -> None:
    service, worker, engine = await make_service(repository, settings)
    kb = await service.create_knowledge_base("alice", "KB", "")
    await service.ingest_document(
        "alice", kb["id"], material_payload("same content"), idempotency_key="first"
    )
    await worker.process_one()
    await service.ingest_document(
        "alice",
        kb["id"],
        material_payload("same content", revision=1),
        idempotency_key="duplicate",
    )
    await worker.process_one()
    assert len((await service.list_documents("alice", kb["id"]))["documents"]) == 1
    assert len(engine.documents[next(iter(engine.documents))]) == 1


async def test_query_maps_chunks_to_immutable_source_versions(repository, settings) -> None:
    service, worker, _ = await make_service(repository, settings)
    kb = await service.create_knowledge_base("alice", "KB", "")
    await service.ingest_document(
        "alice", kb["id"], material_payload("Alpha marker"), idempotency_key="one"
    )
    await worker.process_one()

    result = await service.query("alice", kb["id"], "Alpha", 1, 1)
    documents = (await service.list_documents("alice", kb["id"]))["documents"]
    assert result["scope"] == {
        "knowledge_base_id": kb["id"],
        "revision": 1,
        "epoch": 1,
    }
    assert result["evidence"] == [
        {
            "evidence_id": f"kb:{kb['id']}:chunk-{documents[0]['version_id']}",
            "kind": "kb_chunk",
            "knowledge_base_id": kb["id"],
            "document_id": documents[0]["id"],
            "source_version_id": documents[0]["version_id"],
            "chunk_id": f"chunk-{documents[0]['version_id']}",
            "title": "notes.txt",
            "snippet": "Alpha marker",
            "text": "Alpha marker",
            "locator": {"page": 1},
        }
    ]


async def test_failed_write_marks_dirty_and_retry_is_explicit(repository, settings) -> None:
    engine = FakeEngine()
    engine.fail_next = True
    service, worker, _ = await make_service(repository, settings, engine)
    kb = await service.create_knowledge_base("alice", "KB", "")
    queued = await service.ingest_document(
        "alice", kb["id"], material_payload(), idempotency_key="failure"
    )

    assert await worker.process_one() == queued["job_id"]
    job = await service.get_job("alice", queued["job_id"])
    assert job["status"] == "failed"
    assert job["error_code"] == "index_failed"
    assert (await service.get_scope("alice", kb["id"]))["status"] == "dirty"
    with pytest.raises(Unavailable):
        await service.query("alice", kb["id"], "Alpha", 0, 1)

    retry = await service.retry_job("alice", queued["job_id"], expected_revision=0)
    assert retry["job_id"] != queued["job_id"]
    await worker.process_one()
    assert (await service.get_scope("alice", kb["id"]))["status"] == "ready"


async def test_rebuild_promotes_failed_initial_ingest_without_ghost_document(
    repository, settings
) -> None:
    engine = FakeEngine()
    engine.fail_next = True
    service, worker, _ = await make_service(repository, settings, engine)
    kb = await service.create_knowledge_base("alice", "KB", "")
    await service.ingest_document(
        "alice", kb["id"], material_payload("recover marker"), idempotency_key="failed"
    )
    await worker.process_one()
    assert (await service.get_scope("alice", kb["id"]))["status"] == "dirty"

    await service.create_rebuild_job("alice", kb["id"], 0, "rebuild-failed-ingest")
    await worker.process_one()
    assert engine.rebuild_clear_flags == [False]
    documents = (await service.list_documents("alice", kb["id"]))["documents"]
    assert len(documents) == 1
    assert documents[0]["status"] == "active"
    assert documents[0]["version_id"] is not None
    result = await service.query("alice", kb["id"], "recover", 1, 2)
    assert result["evidence"][0]["document_id"] == documents[0]["id"]


async def test_snapshot_owner_session_expiry_and_import(repository, settings) -> None:
    service, worker, _ = await make_service(repository, settings)
    now = datetime.now(timezone.utc)
    snapshot = await service.create_web_snapshot(
        "alice",
        {
            "session_id": "session-a",
            "url": "https://example.test/page",
            "title": "Example",
            "text": "Alpha web marker",
            "content_hash": hashlib.sha256(b"Alpha web marker").hexdigest(),
            "fetched_at": now.isoformat(),
            "expires_at": (now + timedelta(days=1)).isoformat(),
        },
    )
    with pytest.raises(KeyError):
        await service.get_web_snapshot("alice", snapshot["id"], "wrong-session")

    kb = await service.create_knowledge_base("alice", "KB", "")
    job = await service.import_web_snapshot(
        "alice",
        kb["id"],
        snapshot["id"],
        expected_revision=0,
        idempotency_key="web-1",
    )
    await worker.process_one()
    assert job["knowledge_base_id"] == kb["id"]
    doc = (await service.list_documents("alice", kb["id"]))["documents"][0]
    assert doc["kind"] == "web"
    assert doc["source_url"] == "https://example.test/page"
    await repository.pool.execute(
        "UPDATE sl_web_snapshots SET expires_at=now()-interval '1 second' WHERE id=$1",
        snapshot["id"],
    )
    assert await service.import_web_snapshot(
        "alice",
        kb["id"],
        snapshot["id"],
        expected_revision=0,
        idempotency_key="web-1",
    ) == job
    with pytest.raises(Expired):
        await service.import_web_snapshot(
            "alice",
            kb["id"],
            snapshot["id"],
            expected_revision=1,
            idempotency_key="web-expired-new-key",
        )

    expired = await service.create_web_snapshot(
        "alice",
        {
            "session_id": "session-a",
            "url": "https://example.test/old",
            "title": "Old",
            "text": "old",
            "content_hash": hashlib.sha256(b"old").hexdigest(),
            "fetched_at": (now - timedelta(days=2)).isoformat(),
            "expires_at": (now - timedelta(days=1)).isoformat(),
        },
    )
    with pytest.raises(Expired):
        await service.import_web_snapshot(
            "alice",
            kb["id"],
            expired["id"],
            expected_revision=1,
            idempotency_key="web-old",
        )
    assert await repository.cleanup_expired_snapshots(limit=100) == 1
    assert await repository.pool.fetchval(
        "SELECT count(*) FROM sl_web_snapshots WHERE id=$1 AND body=''", expired["id"]
    ) == 1
    with pytest.raises(Expired):
        await service.get_web_snapshot("alice", expired["id"], "session-a")
    with pytest.raises(Expired):
        await service.import_web_snapshot(
            "alice",
            kb["id"],
            expired["id"],
            expected_revision=1,
            idempotency_key="web-old-after-cleanup",
        )
    assert await repository.pool.fetchval(
        "SELECT count(*) FROM sl_web_snapshots WHERE id=$1 AND imported_at IS NOT NULL",
        snapshot["id"],
    ) == 1


async def test_graph_uses_stable_ids_and_replays_rename(repository, settings) -> None:
    engine = FakeEngine()
    engine.nodes = [
        {"id": "Alpha", "labels": ["Concept"], "properties": {}},
        {"id": "Beta", "labels": ["Concept"], "properties": {}},
    ]
    engine.edges = [
        {"id": "sdk-edge", "source": "Alpha", "target": "Beta", "properties": {}}
    ]
    service, worker, _ = await make_service(repository, settings, engine)
    kb = await service.create_knowledge_base("alice", "KB", "")
    graph = await service.graph("alice", kb["id"], None, 200, 400)
    alpha = next(node for node in graph["nodes"] if node["label"] == "Alpha")
    beta = next(node for node in graph["nodes"] if node["label"] == "Beta")
    edge = graph["edges"][0]
    assert edge["source"] == alpha["id"]
    assert edge["target"] == beta["id"]

    queued = await service.create_correction(
        "alice",
        kb["id"],
        {"kind": "rename_entity", "entity_id": alpha["id"], "label": "Gamma", "expected_revision": 0},
        idempotency_key="rename-alpha",
    )
    await worker.process_one()
    assert queued["knowledge_base_id"] == kb["id"]
    assert await service.create_correction(
        "alice",
        kb["id"],
        {"kind": "rename_entity", "entity_id": alpha["id"], "label": "Gamma", "expected_revision": 0},
        idempotency_key="rename-alpha",
    ) == queued
    assert engine.corrections[-1][1:] == (
        "rename_entity",
        {"entity_label": "Alpha", "label": "Gamma"},
    )

    with pytest.raises(Conflict):
        await service.create_correction(
            "alice",
            kb["id"],
            {"kind": "rename_entity", "entity_id": alpha["id"], "label": "Beta", "expected_revision": 1},
            idempotency_key="rename-collision",
        )


async def test_rename_rejects_unmaterialized_sdk_target_before_persistence(
    repository, settings
) -> None:
    engine = FakeEngine()
    engine.nodes = [{"id": "Alpha", "labels": ["Concept"], "properties": {}}]
    service, _, _ = await make_service(repository, settings, engine)
    kb = await service.create_knowledge_base("alice", "KB", "")
    graph = await service.graph("alice", kb["id"], None, 200, 400)
    alpha_id = graph["nodes"][0]["id"]
    engine.nodes.append({"id": "Gamma", "labels": ["Concept"], "properties": {}})

    with pytest.raises(Conflict):
        await service.create_correction(
            "alice",
            kb["id"],
            {
                "kind": "rename_entity",
                "entity_id": alpha_id,
                "label": "Gamma",
                "expected_revision": 0,
            },
            idempotency_key="unmaterialized-target",
        )
    assert engine.corrections == []
    assert await service.get_scope("alice", kb["id"]) == {
        "knowledge_base_id": kb["id"],
        "revision": 0,
        "epoch": 0,
        "status": "ready",
    }


async def test_same_label_rename_creates_no_job_and_keeps_kb_ready(
    repository, settings
) -> None:
    engine = FakeEngine()
    engine.nodes = [{"id": "Alpha", "labels": ["Concept"], "properties": {}}]
    service, _, _ = await make_service(repository, settings, engine)
    kb = await service.create_knowledge_base("alice", "KB", "")
    alpha = (await service.graph("alice", kb["id"], None, 200, 400))["nodes"][0]
    with pytest.raises(Conflict):
        await service.create_correction(
            "alice",
            kb["id"],
            {
                "kind": "rename_entity",
                "entity_id": alpha["id"],
                "label": "Alpha",
                "expected_revision": 0,
            },
            idempotency_key="same-label",
        )
    assert await repository.pool.fetchval(
        "SELECT count(*) FROM sl_jobs WHERE knowledge_base_id=$1", kb["id"]
    ) == 0
    assert (await service.get_scope("alice", kb["id"]))["status"] == "ready"


async def test_chained_rename_merge_owns_all_aliases_after_reindex(
    repository, settings
) -> None:
    engine = FakeEngine()
    engine.nodes = [
        {"id": label, "labels": ["Concept"], "properties": {}}
        for label in ("Alpha", "Beta", "Delta")
    ]
    service, worker, _ = await make_service(repository, settings, engine)
    kb = await service.create_knowledge_base("alice", "KB", "")
    initial = await service.graph("alice", kb["id"], None, 200, 400)
    ids = {node["label"]: node["id"] for node in initial["nodes"]}
    await service.create_correction(
        "alice",
        kb["id"],
        {"kind": "rename_entity", "entity_id": ids["Alpha"], "label": "Gamma", "expected_revision": 0},
        idempotency_key="rename-chain",
    )
    await worker.process_one()
    await service.create_correction(
        "alice",
        kb["id"],
        {
            "kind": "merge_entities",
            "entity_ids": [ids["Alpha"]],
            "target_id": ids["Beta"],
            "expected_revision": 1,
        },
        idempotency_key="merge-chain",
    )
    await worker.process_one()
    await service.ingest_document(
        "alice",
        kb["id"],
        material_payload("reindex aliases", revision=2),
        idempotency_key="reindex-aliases",
    )
    await worker.process_one()

    engine.nodes = [
        {"id": label, "labels": ["Concept"], "properties": {}}
        for label in ("Alpha", "Gamma", "Beta", "Delta")
    ]
    graph = await service.graph("alice", kb["id"], None, 200, 400)
    assert [node["label"] for node in graph["nodes"]].count("Beta") == 1
    beta = next(node for node in graph["nodes"] if node["label"] == "Beta")
    assert {"Alpha", "Gamma"}.issubset(set(beta["aliases"]))
    with pytest.raises(Conflict):
        await service.create_correction(
            "alice",
            kb["id"],
            {
                "kind": "rename_entity",
                "entity_id": ids["Delta"],
                "label": "Gamma",
                "expected_revision": 3,
            },
            idempotency_key="alias-collision",
        )


async def test_merge_rejects_empty_sources_without_creating_job(repository, settings) -> None:
    engine = FakeEngine()
    engine.nodes = [{"id": "Target", "labels": ["Concept"], "properties": {}}]
    service, _, _ = await make_service(repository, settings, engine)
    kb = await service.create_knowledge_base("alice", "KB", "")
    target = (await service.graph("alice", kb["id"], None, 200, 400))["nodes"][0]
    with pytest.raises(Conflict):
        await service.create_correction(
            "alice",
            kb["id"],
            {
                "kind": "merge_entities",
                "entity_ids": [],
                "target_id": target["id"],
                "expected_revision": 0,
            },
            idempotency_key="empty-merge",
        )
    assert await repository.pool.fetchval(
        "SELECT count(*) FROM sl_jobs WHERE knowledge_base_id=$1", kb["id"]
    ) == 0


async def test_edge_identity_survives_rename_and_old_id_resolves_after_merge(
    repository, settings
) -> None:
    engine = FakeEngine()
    engine.nodes = [
        {"id": label, "labels": ["Concept"], "properties": {}}
        for label in ("Alpha", "Beta", "Gamma")
    ]
    engine.edges = [
        {"id": "sdk-a", "source": "Alpha", "target": "Beta", "properties": {}},
        {"id": "sdk-g", "source": "Gamma", "target": "Beta", "properties": {}},
    ]
    service, worker, _ = await make_service(repository, settings, engine)
    kb = await service.create_knowledge_base("alice", "KB", "")
    first = await service.graph("alice", kb["id"], None, 200, 400)
    ids = {node["label"]: node["id"] for node in first["nodes"]}
    alpha_edge_id = next(
        edge["id"] for edge in first["edges"] if edge["source"] == ids["Alpha"]
    )

    await service.create_correction(
        "alice",
        kb["id"],
        {
            "kind": "rename_entity",
            "entity_id": ids["Alpha"],
            "label": "RenamedAlpha",
            "expected_revision": 0,
        },
        idempotency_key="edge-rename",
    )
    await worker.process_one()
    engine.nodes[0]["id"] = "RenamedAlpha"
    engine.edges[0]["source"] = "RenamedAlpha"
    renamed = await service.graph("alice", kb["id"], None, 200, 400)
    assert any(edge["id"] == alpha_edge_id for edge in renamed["edges"])

    await service.create_correction(
        "alice",
        kb["id"],
        {
            "kind": "merge_entities",
            "entity_ids": [ids["Alpha"]],
            "target_id": ids["Gamma"],
            "expected_revision": 1,
        },
        idempotency_key="edge-merge",
    )
    await worker.process_one()
    await service.create_correction(
        "alice",
        kb["id"],
        {"kind": "delete_relation", "edge_id": alpha_edge_id, "expected_revision": 2},
        idempotency_key="edge-delete-old-id",
    )
    await worker.process_one()
    assert engine.corrections[-1][1:] == (
        "delete_relation",
        {"source_label": "Gamma", "target_label": "Beta"},
    )


async def test_rename_then_delete_old_edge_id_targets_renamed_sdk_edge_without_graph_refresh(
    repository, settings
) -> None:
    class MutatingEngine(FakeEngine):
        async def apply_correction(self, workspace: str, kind: str, payload: dict) -> None:
            await super().apply_correction(workspace, kind, payload)
            if kind == "rename_entity":
                for node in self.nodes:
                    if node["id"] == payload["entity_label"]:
                        node["id"] = payload["label"]
                for edge in self.edges:
                    if edge["source"] == payload["entity_label"]:
                        edge["source"] = payload["label"]
                    if edge["target"] == payload["entity_label"]:
                        edge["target"] = payload["label"]
            elif kind == "delete_relation":
                self.edges = [
                    edge
                    for edge in self.edges
                    if not (
                        edge["source"] == payload["source_label"]
                        and edge["target"] == payload["target_label"]
                    )
                ]

    engine = MutatingEngine()
    engine.nodes = [
        {"id": label, "labels": ["Concept"], "properties": {}}
        for label in ("Alpha", "Beta")
    ]
    engine.edges = [
        {"id": "sdk-edge", "source": "Alpha", "target": "Beta", "properties": {}}
    ]
    service, worker, _ = await make_service(repository, settings, engine)
    kb = await service.create_knowledge_base("alice", "KB", "")
    initial = await service.graph("alice", kb["id"], None, 200, 400)
    alpha = next(node for node in initial["nodes"] if node["label"] == "Alpha")
    edge_id = initial["edges"][0]["id"]
    await service.create_correction(
        "alice",
        kb["id"],
        {
            "kind": "rename_entity",
            "entity_id": alpha["id"],
            "label": "Gamma",
            "expected_revision": 0,
        },
        idempotency_key="direct-edge-rename",
    )
    await worker.process_one()
    await service.create_correction(
        "alice",
        kb["id"],
        {"kind": "delete_relation", "edge_id": edge_id, "expected_revision": 1},
        idempotency_key="direct-edge-delete",
    )
    await worker.process_one()
    assert engine.corrections[-1][1:] == (
        "delete_relation",
        {"source_label": "Gamma", "target_label": "Beta"},
    )
    assert engine.edges == []


async def test_source_view_reports_current_historical_and_deleted(repository, settings) -> None:
    service, worker, _ = await make_service(repository, settings)
    kb = await service.create_knowledge_base("alice", "KB", "")
    await service.ingest_document(
        "alice", kb["id"], material_payload("version one"), idempotency_key="v1"
    )
    await worker.process_one()
    first = (await service.list_documents("alice", kb["id"]))["documents"][0]
    assert (await service.get_source("alice", kb["id"], first["version_id"]))[
        "source_status"
    ] == "current"
    bounded = await service.get_source("alice", kb["id"], first["version_id"], 4)
    assert bounded["text"] == "vers"
    assert bounded["parsed_blocks"] == [{"text": "vers", "metadata": {"page": 1}}]
    assert bounded["truncated"] is True
    assert (bounded["returned_blocks"], bounded["total_blocks"]) == (1, 1)

    await service.ingest_document(
        "alice",
        kb["id"],
        material_payload("version two", revision=1),
        idempotency_key="v2",
        document_id=first["id"],
    )
    await worker.process_one()
    second = (await service.list_documents("alice", kb["id"]))["documents"][0]
    assert (await service.get_source("alice", kb["id"], first["version_id"]))[
        "source_status"
    ] == "historical"
    assert (await service.get_source("alice", kb["id"], second["version_id"]))[
        "source_status"
    ] == "current"

    await service.delete_document(
        "alice", kb["id"], first["id"], 2, "delete-source-document"
    )
    await worker.process_one()
    assert (await service.get_source("alice", kb["id"], second["version_id"]))[
        "source_status"
    ] == "deleted"
    assert not (await service.query("alice", kb["id"], "version", 3, 3))["evidence"]


async def test_engine_cache_never_evicts_an_active_workspace(settings) -> None:
    class IdleRag:
        def __init__(self) -> None:
            self.finalized = 0

        async def finalize_storages(self) -> None:
            self.finalized += 1

    engine = LightRAGEngine(replace(settings, max_cached_instances=1))
    first, second = IdleRag(), IdleRag()
    engine._instances = OrderedDict([("first", first), ("second", second)])
    engine._active = {"first": 0, "second": 0}

    async with engine._use("first"):
        async with engine._use("second"):
            assert first.finalized == second.finalized == 0
        assert "first" in engine._instances
        assert first.finalized == 0
        assert second.finalized == 1
    await engine.close()
    assert first.finalized == 1


async def test_query_timeout_releases_per_kb_guard(repository, settings) -> None:
    class SlowEngine(FakeEngine):
        async def query(self, workspace: str, query: str) -> dict:
            await asyncio.sleep(1)
            return await super().query(workspace, query)

    engine = SlowEngine()
    service = KnowledgeService(
        repository,
        MaterialStore(settings.materials_dir),
        engine,
        query_timeout_seconds=0.01,
    )
    kb = await service.create_knowledge_base("alice", "KB", "")
    with pytest.raises(TimeoutError):
        await service.query("alice", kb["id"], "marker", 0, 0)
    async with service.locks.write(kb["id"]):
        pass


async def test_restored_index_config_drift_blocks_everything_except_rebuild(
    repository, settings
) -> None:
    original, _, engine = await make_service(repository, settings)
    kb = await original.create_knowledge_base("alice", "KB", "")
    restored = KnowledgeService(
        repository,
        MaterialStore(settings.materials_dir),
        engine,
        index_hash="changed-index-config",
    )
    await repository.mark_config_drift_dirty(restored.index_hash)

    assert (await restored.get_knowledge_base("alice", kb["id"]))["status"] == "dirty"
    assert (await restored.list_knowledge_bases("alice"))["knowledge_bases"][0][
        "status"
    ] == "dirty"
    assert (await restored.get_scope("alice", kb["id"]))["epoch"] == 1
    with pytest.raises(Unavailable):
        await restored.query("alice", kb["id"], "anything", 0, 1)
    with pytest.raises(Unavailable):
        await restored.ingest_document(
            "alice",
            kb["id"],
            material_payload("blocked"),
            idempotency_key="blocked-by-drift",
        )
    assert await repository.pool.fetchval(
        "SELECT count(*) FROM sl_jobs WHERE knowledge_base_id=$1", kb["id"]
    ) == 0

    queued = await restored.create_rebuild_job(
        "alice", kb["id"], 0, "repair-config-drift"
    )
    worker = WriteWorker(repository, restored, engine)
    assert await worker.process_one() == queued["job_id"]
    assert await restored.get_scope("alice", kb["id"]) == {
        "knowledge_base_id": kb["id"],
        "revision": 1,
        "epoch": 2,
        "status": "ready",
    }
    assert engine.rebuild_clear_flags == [True]


async def test_worker_supervisor_recovers_without_replaying_running_job() -> None:
    class Repo:
        def __init__(self) -> None:
            self.recoveries = 0

        async def mark_interrupted_dirty(self) -> None:
            self.recoveries += 1

        async def cleanup_expired_snapshots(self, limit: int = 100) -> int:
            return 0

    class RecoveringWorker(WriteWorker):
        def __init__(self, repo) -> None:
            super().__init__(repo, object(), FakeEngine())
            self.calls = 0

        async def process_one(self) -> str | None:
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("database unavailable")
            self.stop()
            return None

    repo = Repo()
    worker = RecoveringWorker(repo)
    await asyncio.wait_for(worker.run(), timeout=2)
    assert repo.recoveries == 1
    assert worker.calls == 2
    assert worker.healthy is True
