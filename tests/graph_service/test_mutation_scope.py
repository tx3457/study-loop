"""Owner-global receipt keys must still identify one exact mutation target."""

import base64
import hashlib
import uuid
from datetime import datetime, timedelta, timezone

import httpx
import pytest

pytest.importorskip("asyncpg")

from graph_service.api import create_app
from graph_service.errors import Conflict, Expired
from graph_service.materials import MaterialStore
from graph_service.service import KnowledgeService
from graph_service.worker import WriteWorker
from fakes import FakeEngine


pytestmark = pytest.mark.asyncio


def material(text="shared body", revision=0):
    return {"name": "notes.txt", "kind": "file",
            "content_base64": base64.b64encode(text.encode()).decode(),
            "parsed_blocks": [{"text": text, "metadata": {}}],
            "expected_revision": revision}


def build(repository, settings):
    engine = FakeEngine()
    service = KnowledgeService(repository, MaterialStore(settings.materials_dir), engine)
    return service, WriteWorker(repository, service, engine)


async def test_same_upload_key_cannot_replay_another_knowledge_base(repository, settings):
    service, worker = build(repository, settings)
    a = await service.create_knowledge_base("alice", "A")
    b = await service.create_knowledge_base("alice", "B")
    app = create_app(settings, service=service, worker=worker, manage_lifespan=False)
    headers = {"Authorization": "Bearer test-token", "X-StudyLoop-Subject": "alice",
               "Idempotency-Key": "one-upload-intent"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        first = await client.post(f"/knowledge-bases/{a['id']}/documents", headers=headers, json=material())
        assert first.status_code == 202
        retry = await client.post(f"/knowledge-bases/{a['id']}/documents", headers=headers, json=material())
        assert retry.status_code == 202 and retry.json() == first.json()
        other = await client.post(f"/knowledge-bases/{b['id']}/documents", headers=headers, json=material())
        assert other.status_code == 409
    assert (await service.list_documents("alice", b["id"]))["documents"] == []
    assert (await service.get_scope("alice", b["id"]))["epoch"] == 0


async def test_same_replace_key_cannot_replay_another_document(repository, settings):
    service, worker = build(repository, settings)
    kb = await service.create_knowledge_base("alice", "A")
    for index, text in enumerate(["first document", "second document"]):
        await service.ingest_document("alice", kb["id"], material(text, index), idempotency_key=f"seed-{index}")
        await worker.process_one()
    docs = (await service.list_documents("alice", kb["id"]))["documents"]
    payload = material("replacement", 2)
    first = await service.ingest_document("alice", kb["id"], payload,
                                          document_id=docs[0]["id"], idempotency_key="replace-once")
    with pytest.raises(Conflict):
        await service.ingest_document("alice", kb["id"], payload,
                                      document_id=docs[1]["id"], idempotency_key="replace-once")
    await worker.process_one()
    assert await service.ingest_document("alice", kb["id"], payload,
                                         document_id=docs[0]["id"], idempotency_key="replace-once") == first
    after = (await service.list_documents("alice", kb["id"]))["documents"]
    unchanged = next(doc for doc in after if doc["id"] == docs[1]["id"])
    assert unchanged["version_id"] == docs[1]["version_id"]


@pytest.mark.parametrize("changed_target", ["knowledge_base", "snapshot"])
async def test_web_receipt_binds_kb_and_snapshot_and_replays_after_expiry(repository, settings, changed_target):
    service, worker = build(repository, settings)
    a = await service.create_knowledge_base("alice", "A")
    b = await service.create_knowledge_base("alice", "B")
    now = datetime.now(timezone.utc)
    payload = {"session_id": "session", "url": "https://example.org/reference", "title": "Reference",
               "text": "captured body", "content_hash": hashlib.sha256(b"captured body").hexdigest(),
               "fetched_at": now.isoformat(), "expires_at": (now + timedelta(days=7)).isoformat()}
    first_snapshot = await service.create_web_snapshot("alice", payload)
    second_snapshot = await service.create_web_snapshot("alice", payload)
    first = await service.import_web_snapshot("alice", a["id"], first_snapshot["id"],
                                              expected_revision=0, idempotency_key="web-once")
    with pytest.raises(Conflict):
        await service.import_web_snapshot(
            "alice", b["id"] if changed_target == "knowledge_base" else a["id"],
            second_snapshot["id"] if changed_target == "snapshot" else first_snapshot["id"],
            expected_revision=0, idempotency_key="web-once",
        )
    await worker.process_one()
    await repository.pool.execute("UPDATE sl_web_snapshots SET expires_at=$2 WHERE id=$1",
                                  uuid.UUID(first_snapshot["id"]), now - timedelta(seconds=1))
    assert await service.import_web_snapshot("alice", a["id"], first_snapshot["id"],
                                             expected_revision=0, idempotency_key="web-once") == first
    with pytest.raises(Expired):
        await service.import_web_snapshot("alice", a["id"], first_snapshot["id"],
                                          expected_revision=1, idempotency_key="new-web-intent")


@pytest.mark.parametrize("operation", ["correction", "delete_document"])
async def test_existing_resource_receipt_does_not_bypass_a_different_kb_path(repository, settings, operation):
    service, worker = build(repository, settings)
    a = await service.create_knowledge_base("alice", "A")
    b = await service.create_knowledge_base("alice", "B")
    await service.ingest_document("alice", a["id"], material(), idempotency_key="seed")
    await worker.process_one()
    if operation == "correction":
        entity_id = uuid.uuid4()
        await repository.pool.execute(
            "INSERT INTO sl_entities(id,knowledge_base_id,label,aliases) VALUES($1,$2,'Alpha','[]')",
            entity_id, uuid.UUID(a["id"]),
        )
        async def mutate(kb_id):
            return await service.create_correction("alice", kb_id,
                {"kind": "delete_entity", "entity_id": str(entity_id), "expected_revision": 1},
                idempotency_key="one-resource")
    else:
        doc = (await service.list_documents("alice", a["id"]))["documents"][0]
        async def mutate(kb_id):
            return await service.delete_document("alice", kb_id, doc["id"], 1, "one-resource")
    first = await mutate(a["id"])
    assert await mutate(a["id"]) == first
    with pytest.raises(Conflict):
        await mutate(b["id"])
