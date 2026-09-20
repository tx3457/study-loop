"""Documents beyond the default page remain discoverable and mutable."""

import uuid

import httpx
import pytest

pytest.importorskip("asyncpg")

from graph_service.api import create_app
from test_mutation_scope import build, material


pytestmark = pytest.mark.asyncio


async def test_document_pages_report_owner_scoped_totals_and_keep_later_documents_mutable(repository, settings):
    service, worker = build(repository, settings)
    kb = await service.create_knowledge_base("alice", "Many documents")
    for index in range(101):
        payload = {**material(f"document {index}", index), "name": f"doc-{index:03}.txt"}
        await service.ingest_document("alice", kb["id"], payload, idempotency_key=f"page-seed-{index}")
        await worker.process_one()
    other = await service.create_knowledge_base("alice", "Separate KB")
    await service.ingest_document("alice", other["id"], material("separate"), idempotency_key="separate")
    await worker.process_one()
    # Timestamp ties need deterministic ID ordering to avoid page overlaps.
    await repository.pool.execute(
        "UPDATE sl_documents SET created_at='2026-09-20T00:00:00Z' WHERE knowledge_base_id=$1",
        uuid.UUID(kb["id"]),
    )
    app = create_app(settings, service=service, worker=worker, manage_lifespan=False)
    headers = {"Authorization": "Bearer test-token", "X-StudyLoop-Subject": "alice"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test", headers=headers) as client:
        first = (await client.get(f"/knowledge-bases/{kb['id']}/documents")).json()
        second = (await client.get(f"/knowledge-bases/{kb['id']}/documents?offset=50&limit=50")).json()
        last = (await client.get(f"/knowledge-bases/{kb['id']}/documents?offset=100&limit=50")).json()
        for offset, page, size in [(0, first, 50), (50, second, 50), (100, last, 1)]:
            assert (page["total"], page["limit"], page["offset"]) == (101, 50, offset)
            assert len(page["documents"]) == size
        ids = [doc["id"] for page in [first, second, last] for doc in page["documents"]]
        assert len(set(ids)) == 101 and ids == sorted(ids)
        late = last["documents"][0]
        await service.ingest_document("alice", kb["id"], material("new late version", 101),
                                      document_id=late["id"], idempotency_key="late-replace")
        await worker.process_one()
        replaced = (await client.get(f"/knowledge-bases/{kb['id']}/documents?offset=100")).json()
        assert replaced["documents"][0]["version_id"] != late["version_id"]
        await service.delete_document("alice", kb["id"], late["id"], 102, "late-delete")
        await worker.process_one()
        empty = (await client.get(f"/knowledge-bases/{kb['id']}/documents?offset=100")).json()
        assert empty == {"documents": [], "total": 100, "limit": 50, "offset": 100}
        assert (await client.get(f"/knowledge-bases/{kb['id']}/documents", headers={
            **headers, "X-StudyLoop-Subject": "bob",
        })).status_code == 404
