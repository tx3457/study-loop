"""Deleting a knowledge base or a document removes its content, not only its index."""

from __future__ import annotations

import base64
import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("asyncpg")

from graph_service.materials import MaterialStore
from graph_service.service import KnowledgeService
from graph_service.worker import WriteWorker

from fakes import FakeEngine


pytestmark = pytest.mark.asyncio


def material_payload(text: str, *, name: str = "notes.txt", revision: int = 0) -> dict:
    return {
        "name": name,
        "kind": "file",
        "content_base64": base64.b64encode(text.encode()).decode(),
        "parsed_blocks": [{"text": text, "metadata": {"page": 1}}],
        "expected_revision": revision,
    }


async def make_service(repository, settings):
    engine = FakeEngine()
    service = KnowledgeService(repository, MaterialStore(settings.materials_dir), engine)
    return service, WriteWorker(repository, service, engine), engine


async def ingest(service, worker, kb_id, text, *, revision, name="notes.txt", document_id=None):
    await service.ingest_document(
        "alice",
        kb_id,
        material_payload(text, name=name, revision=revision),
        idempotency_key=f"ingest-{uuid.uuid4()}",
        document_id=document_id,
    )
    await worker.process_one()


async def version_rows(repository, document_id):
    return await repository.fetch(
        "SELECT id,parsed_text,parsed_blocks,content_hash,material_path "
        "FROM sl_source_versions WHERE document_id=$1 ORDER BY created_at",
        uuid.UUID(document_id),
    )


def material_file(settings, kb_id, version_id) -> Path:
    return Path(settings.materials_dir) / kb_id / f"{version_id}.bin"


async def count(repository, table, kb_id) -> int:
    return await repository.pool.fetchval(
        f"SELECT count(*) FROM {table} WHERE knowledge_base_id=$1", uuid.UUID(kb_id)
    )


async def kb_status(repository, kb_id) -> str:
    return await repository.pool.fetchval(
        "SELECT status FROM sl_knowledge_bases WHERE id=$1", uuid.UUID(kb_id)
    )


async def test_deleted_knowledge_base_keeps_only_a_nameless_tombstone(
    repository, settings
) -> None:
    service, worker, engine = await make_service(repository, settings)
    kb = await service.create_knowledge_base(
        "alice", "Private notes", "diagnosis", idempotency_key="create-private"
    )
    kept = await service.create_knowledge_base("alice", "Kept", "")
    await ingest(service, worker, kb["id"], "secret alpha", revision=0)
    await ingest(service, worker, kept["id"], "kept beta", revision=0)
    doc = (await service.list_documents("alice", kb["id"]))["documents"][0]
    kept_doc = (await service.list_documents("alice", kept["id"]))["documents"][0]
    # Rows the delete path never reached before: corrections and graph identity.
    await repository.pool.execute(
        "INSERT INTO sl_corrections(id,knowledge_base_id,sequence,kind,payload) "
        "VALUES($1,$2,1,'delete_entity',$3::jsonb)",
        uuid.uuid4(), uuid.UUID(kb["id"]), json.dumps({"entity_label": "secret"}),
    )
    await repository.pool.execute(
        "INSERT INTO sl_entities(id,knowledge_base_id,label) VALUES($1,$2,'secret')",
        uuid.uuid4(), uuid.UUID(kb["id"]),
    )
    workspace = (await repository.get_kb("alice", kb["id"]))["workspace"]
    assert material_file(settings, kb["id"], doc["version_id"]).exists()

    job = await service.delete_knowledge_base("alice", kb["id"], 1, "delete-private")
    assert await worker.process_one() == job["job_id"]

    # The client polling the delete job still sees it finish.
    assert (await service.get_job("alice", job["job_id"]))["status"] == "succeeded"
    # Nothing is reachable any more, including the source text of old citations.
    for read in (
        service.get_knowledge_base("alice", kb["id"]),
        service.list_documents("alice", kb["id"]),
        service.get_source("alice", kb["id"], doc["version_id"]),
    ):
        with pytest.raises(KeyError):
            await read
    # Nor writable: every write answers 404, and an upload recreates no directory.
    for write in (
        service.ingest_document(
            "alice", kb["id"], material_payload("late upload", revision=2),
            idempotency_key="late-upload",
        ),
        service.delete_knowledge_base("alice", kb["id"], 2, "delete-again"),
        service.delete_document("alice", kb["id"], doc["id"], 2, "late-document-delete"),
        service.update_knowledge_base(
            "alice", kb["id"], {"name": "back", "expected_revision": 2},
            idempotency_key="late-rename",
        ),
    ):
        with pytest.raises(KeyError):
            await write
    listed = (await service.list_knowledge_bases("alice"))["knowledge_bases"]
    assert [item["id"] for item in listed] == [kept["id"]]
    # Nothing is left in the rows either.
    for table in ("sl_documents", "sl_source_versions", "sl_corrections", "sl_entities"):
        assert await count(repository, table, kb["id"]) == 0, table
    assert await repository.pool.fetchval(
        "SELECT array_agg(operation) FROM sl_jobs WHERE knowledge_base_id=$1",
        uuid.UUID(kb["id"]),
    ) == ["delete_knowledge_base"]
    tombstone = await repository.fetchrow(
        "SELECT name,description,status FROM sl_knowledge_bases WHERE id=$1",
        uuid.UUID(kb["id"]),
    )
    assert tombstone == {"name": "", "description": "", "status": "deleted"}
    # The create replay keeps only the id, so a late retry cannot make a new base.
    replay = await repository.pool.fetchval(
        "SELECT response FROM sl_idempotency WHERE key='create-private'"
    )
    assert json.loads(replay) == {"id": kb["id"]}
    # Nor on disk, nor in the engine.
    assert not (Path(settings.materials_dir) / kb["id"]).exists()
    assert engine.forgotten == [workspace]

    # Control: the other knowledge base is untouched.
    assert material_file(settings, kept["id"], kept_doc["version_id"]).exists()
    source = await service.get_source("alice", kept["id"], kept_doc["version_id"])
    assert source["text"] == "kept beta"


async def test_every_spelling_of_a_knowledge_base_id_shares_one_directory_and_lock(
    repository, settings
) -> None:
    """asyncpg accepts any UUID spelling, so each must reach the directory the purge removes."""
    service, worker, _ = await make_service(repository, settings)
    kb = await service.create_knowledge_base("alice", "KB", "")
    await ingest(service, worker, kb["id"].upper(), "shouted id", revision=0)
    doc = (await service.list_documents("alice", kb["id"]))["documents"][0]
    assert material_file(settings, kb["id"], doc["version_id"]).exists()
    assert await service.locks._lock(kb["id"].upper()) is await service.locks._lock(kb["id"])

    await service.delete_knowledge_base("alice", kb["id"].upper(), 1, "delete-shouted")
    await worker.process_one()

    assert list(Path(settings.materials_dir).iterdir()) == []


async def test_deleted_document_loses_its_text_and_files_in_every_version(
    repository, settings
) -> None:
    service, worker, _ = await make_service(repository, settings)
    kb = await service.create_knowledge_base("alice", "KB", "")
    await ingest(service, worker, kb["id"], "first draft", revision=0, name="a.txt")
    doc = (await service.list_documents("alice", kb["id"]))["documents"][0]
    await ingest(
        service, worker, kb["id"], "second draft", revision=1, name="a.txt",
        document_id=doc["id"],
    )
    await ingest(service, worker, kb["id"], "other notes", revision=2, name="b.txt")
    documents = (await service.list_documents("alice", kb["id"]))["documents"]
    other = next(item for item in documents if item["id"] != doc["id"])
    versions = [str(row["id"]) for row in await version_rows(repository, doc["id"])]
    assert len(versions) == 2
    assert all(material_file(settings, kb["id"], v).exists() for v in versions)

    await service.delete_document("alice", kb["id"], doc["id"], 3, "delete-a")
    await worker.process_one()

    for row in await version_rows(repository, doc["id"]):
        assert row["parsed_text"] == ""
        assert row["parsed_blocks"] in ([], "[]")
        assert row["content_hash"] == ""
        assert row["material_path"] == ""
    for version_id in versions:
        assert not material_file(settings, kb["id"], version_id).exists()
        # An old citation still resolves, as deleted and without the text.
        source = await service.get_source("alice", kb["id"], version_id)
        assert source["source_status"] == "deleted"
        assert (source["title"], source["text"], source["parsed_blocks"]) == ("a.txt", "", [])

    # Control: the other document keeps its text and file.
    assert material_file(settings, kb["id"], other["version_id"]).exists()
    assert (await service.get_source("alice", kb["id"], other["version_id"]))[
        "text"
    ] == "other notes"


async def test_a_deletion_the_sdk_refuses_is_completed_by_rebuilding_without_it(
    repository, settings
) -> None:
    """LightRAG refuses some deletions before writing anything, and refuses again on
    retry; the job removes the document by rebuilding instead of leaving the base dirty."""
    service, worker, engine = await make_service(repository, settings)
    kb = await service.create_knowledge_base("alice", "KB", "")
    await ingest(service, worker, kb["id"], "refused text", revision=0, name="a.txt")
    await ingest(service, worker, kb["id"], "kept text", revision=1, name="b.txt")
    refused, kept = (await service.list_documents("alice", kb["id"]))["documents"]
    workspace = (await repository.get_kb("alice", kb["id"]))["workspace"]
    engine.refuse_deletion.add(refused["version_id"])

    job = await service.delete_document("alice", kb["id"], refused["id"], 2, "delete-refused")
    await worker.process_one()

    assert (await service.get_job("alice", job["job_id"]))["status"] == "succeeded"
    assert engine.rebuilds == [workspace]
    assert set(engine.documents[workspace]) == {kept["version_id"]}
    assert (await service.get_knowledge_base("alice", kb["id"]))["status"] == "ready"


async def test_identity_corrected_delete_rebuilds_and_still_purges_the_file(
    repository, settings
) -> None:
    service, worker, engine = await make_service(repository, settings)
    kb = await service.create_knowledge_base("alice", "KB", "")
    await ingest(service, worker, kb["id"], "renamed source", revision=0)
    doc = (await service.list_documents("alice", kb["id"]))["documents"][0]
    await repository.pool.execute(
        "INSERT INTO sl_corrections(id,knowledge_base_id,sequence,kind,payload) "
        "VALUES($1,$2,1,'rename_entity',$3::jsonb)",
        uuid.uuid4(), uuid.UUID(kb["id"]),
        json.dumps({"engine_payload": {"entity_label": "A", "new_label": "B"}}),
    )

    await service.delete_document("alice", kb["id"], doc["id"], 1, "delete-renamed")
    await worker.process_one()

    workspace = (await repository.get_kb("alice", kb["id"]))["workspace"]
    assert engine.rebuilds == [workspace]
    assert engine.documents[workspace] == {}
    assert not material_file(settings, kb["id"], doc["version_id"]).exists()


async def test_the_sweep_finishes_deletions_left_by_a_crash_or_an_older_release(
    repository, settings
) -> None:
    """Before this fix a delete stopped at 'deleting' and kept every row and file.

    Both that state and a crash between the delete job and its purge look the same,
    and the periodic sweep and the startup migration finish them.
    """
    service, worker, engine = await make_service(repository, settings)
    legacy = await service.create_knowledge_base("alice", "Legacy", "old")
    await ingest(service, worker, legacy["id"], "legacy text", revision=0)
    legacy_doc = (await service.list_documents("alice", legacy["id"]))["documents"][0]
    await repository.pool.execute(
        "UPDATE sl_knowledge_bases SET status='deleting' WHERE id=$1", uuid.UUID(legacy["id"])
    )
    await repository.pool.execute(
        "UPDATE sl_documents SET status='deleted' WHERE knowledge_base_id=$1",
        uuid.UUID(legacy["id"]),
    )
    live = await service.create_knowledge_base("alice", "Live", "")
    await ingest(service, worker, live["id"], "old deleted text", revision=0)
    old_doc = (await service.list_documents("alice", live["id"]))["documents"][0]
    await repository.pool.execute(
        "UPDATE sl_documents SET status='deleted' WHERE id=$1", uuid.UUID(old_doc["id"])
    )
    # A workspace still serving a request is not torn down under it.
    legacy_workspace = await repository.pool.fetchval(
        "SELECT workspace FROM sl_knowledge_bases WHERE id=$1", uuid.UUID(legacy["id"])
    )
    engine.busy_workspaces.add(legacy_workspace)

    await repository.migrate()
    await worker.purge_deleted_data()

    assert (await version_rows(repository, old_doc["id"]))[0]["parsed_text"] == ""
    assert not material_file(settings, live["id"], old_doc["version_id"]).exists()
    # Files go first, by their recorded paths; the directory waits for the engine.
    assert not material_file(settings, legacy["id"], legacy_doc["version_id"]).exists()
    assert await count(repository, "sl_documents", legacy["id"]) == 0
    assert await kb_status(repository, legacy["id"]) == "deleting"

    engine.busy_workspaces.clear()
    await worker.purge_deleted_data()

    assert await kb_status(repository, legacy["id"]) == "deleted"
    assert not (Path(settings.materials_dir) / legacy["id"]).exists()
    assert engine.forgotten == [legacy_workspace]
    # Idempotent: another pass finds nothing to do.
    await worker.purge_deleted_data()
    assert engine.forgotten == [legacy_workspace]


async def test_material_purge_pages_past_files_that_cannot_be_removed(
    repository, settings, monkeypatch
) -> None:
    service, worker, _ = await make_service(repository, settings)
    kb = await service.create_knowledge_base("alice", "KB", "")
    for index in range(3):
        await ingest(service, worker, kb["id"], f"text {index}", revision=index, name=f"{index}.txt")
    documents = (await service.list_documents("alice", kb["id"]))["documents"]
    await repository.pool.execute(
        "UPDATE sl_documents SET status='deleted' WHERE knowledge_base_id=$1", uuid.UUID(kb["id"])
    )
    stuck = min(documents, key=lambda d: d["version_id"])["version_id"]
    original_delete = service.materials.delete

    def delete(path):
        if stuck in path:
            raise PermissionError(path)
        original_delete(path)

    monkeypatch.setattr(service.materials, "delete", delete)
    original_page = repository.deleted_materials
    monkeypatch.setattr(
        repository, "deleted_materials", lambda after=None: original_page(after, limit=1)
    )

    await worker.purge_deleted_data()

    for document in documents:
        assert material_file(settings, kb["id"], document["version_id"]).exists() is (
            document["version_id"] == stuck
        )


async def import_page(service, worker, kb_id, text, *, revision, snapshot_id=None):
    if snapshot_id is None:
        snapshot_id = await create_snapshot(service, text)
    await service.import_web_snapshot(
        "alice", kb_id, snapshot_id,
        expected_revision=revision, idempotency_key=f"import-{uuid.uuid4()}",
    )
    await worker.process_one()
    return snapshot_id


async def create_snapshot(service, text: str) -> str:
    now = datetime.now(timezone.utc)
    snapshot = await service.create_web_snapshot(
        "alice",
        {
            "session_id": "session-a",
            "url": "https://example.test/page",
            "title": "Page",
            "text": text,
            "content_hash": hashlib.sha256(text.encode()).hexdigest(),
            "fetched_at": now.isoformat(),
            "expires_at": (now + timedelta(days=1)).isoformat(),
        },
    )
    return snapshot["id"]


async def snapshot_body(repository, snapshot_id) -> str:
    return await repository.pool.fetchval(
        "SELECT body FROM sl_web_snapshots WHERE id=$1", uuid.UUID(snapshot_id)
    )


async def snapshot_url(repository, snapshot_id) -> str:
    return await repository.pool.fetchval(
        "SELECT url FROM sl_web_snapshots WHERE id=$1", uuid.UUID(snapshot_id)
    )


async def test_deleting_an_imported_page_clears_only_the_snapshot_it_came_from(
    repository, settings
) -> None:
    service, worker, _ = await make_service(repository, settings)
    first = await service.create_knowledge_base("alice", "First", "")
    second = await service.create_knowledge_base("alice", "Second", "")
    # The same page fetched twice and imported into two bases, plus one snapshot
    # imported into both.
    first_copy = await import_page(service, worker, first["id"], "same page", revision=0)
    second_copy = await import_page(service, worker, second["id"], "same page", revision=0)
    shared = await create_snapshot(service, "shared page")
    await import_page(service, worker, first["id"], "", revision=1, snapshot_id=shared)
    await import_page(service, worker, second["id"], "", revision=1, snapshot_id=shared)
    first_docs = (await service.list_documents("alice", first["id"]))["documents"]

    for position, document in enumerate(first_docs):
        await service.delete_document(
            "alice", first["id"], document["id"], 2 + position, f"delete-{position}"
        )
        await worker.process_one()

    assert await snapshot_body(repository, first_copy) == ""
    assert await snapshot_url(repository, first_copy) == ""
    # The other base's documents still stand on their snapshots.
    assert await snapshot_body(repository, second_copy) == "same page"
    assert await snapshot_body(repository, shared) == "shared page"
    # The tombstone keeps the title only.
    source = await service.get_source("alice", first["id"], first_docs[0]["version_id"])
    assert (source["title"], source["source_url"], source["source_fetched_at"]) == (
        "Page", None, None
    )

    await service.delete_knowledge_base("alice", second["id"], 2, "delete-second")
    await worker.process_one()
    assert await snapshot_body(repository, second_copy) == ""
    assert await snapshot_body(repository, shared) == ""


async def test_a_page_imported_twice_into_one_base_clears_both_snapshots(
    repository, settings
) -> None:
    """The second import reuses the first document, so it adds no document to link;
    the snapshot it came from must still be cleared when that document goes."""
    service, worker, _ = await make_service(repository, settings)
    kb = await service.create_knowledge_base("alice", "Web", "")
    first = await import_page(service, worker, kb["id"], "page body", revision=0)
    second = await import_page(service, worker, kb["id"], "page body", revision=1)
    [document] = (await service.list_documents("alice", kb["id"]))["documents"]

    await service.delete_document("alice", kb["id"], document["id"], 2, "delete-twice")
    await worker.process_one()

    assert await snapshot_body(repository, first) == ""
    assert await snapshot_body(repository, second) == ""


async def test_migration_links_and_clears_pages_deleted_by_an_older_release(
    repository, settings
) -> None:
    service, worker, _ = await make_service(repository, settings)
    kb = await service.create_knowledge_base("alice", "Web", "")
    page = await import_page(service, worker, kb["id"], "legacy page body", revision=0)
    # A second fetch of the same page, imported as a reuse of the same document.
    again = await import_page(service, worker, kb["id"], "legacy page body", revision=1)
    # Older releases recorded no snapshot link and cleared nothing.
    await repository.pool.execute("DELETE FROM sl_document_snapshots")
    await repository.pool.execute(
        "UPDATE sl_documents SET status='deleted' WHERE knowledge_base_id=$1",
        uuid.UUID(kb["id"]),
    )

    await repository.migrate()

    assert await snapshot_body(repository, page) == ""
    assert await snapshot_body(repository, again) == ""
    assert await repository.pool.fetchval(
        "SELECT source_url FROM sl_documents WHERE knowledge_base_id=$1", uuid.UUID(kb["id"])
    ) is None


async def test_a_deleted_documents_files_go_at_once_whatever_the_backlog_cursor(
    repository, settings
) -> None:
    service, worker, _ = await make_service(repository, settings)
    kb = await service.create_knowledge_base("alice", "KB", "")
    await ingest(service, worker, kb["id"], "prompt removal", revision=0)
    doc = (await service.list_documents("alice", kb["id"]))["documents"][0]
    # The shared backlog cursor is past every id, as it is mid-way through a backlog.
    worker._material_cursor = uuid.UUID(int=(1 << 128) - 1)

    await service.delete_document("alice", kb["id"], doc["id"], 1, "delete-now")
    await worker.process_one()

    assert not material_file(settings, kb["id"], doc["version_id"]).exists()


async def test_a_restart_does_not_tie_a_snapshot_to_a_document_that_never_used_it(
    repository, settings
) -> None:
    service, worker, _ = await make_service(repository, settings)
    kept = await service.create_knowledge_base("alice", "Kept", "")
    dropped = await service.create_knowledge_base("alice", "Dropped", "")
    await import_page(service, worker, kept["id"], "same page", revision=0)
    other_fetch = await import_page(service, worker, dropped["id"], "same page", revision=0)

    await repository.migrate()
    [document] = (await service.list_documents("alice", dropped["id"]))["documents"]
    await service.delete_document("alice", dropped["id"], document["id"], 1, "delete-dropped")
    await worker.process_one()

    assert await snapshot_body(repository, other_fetch) == ""
