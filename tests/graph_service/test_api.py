from __future__ import annotations

import base64

import httpx
import pytest

pytest.importorskip("asyncpg")

from graph_service.api import create_app
from graph_service.materials import MaterialStore
from graph_service.service import KnowledgeService
from graph_service.worker import WriteWorker

from fakes import FakeEngine


pytestmark = pytest.mark.asyncio


async def test_internal_auth_and_document_contract(repository, settings) -> None:
    engine = FakeEngine()
    service = KnowledgeService(repository, MaterialStore(settings.materials_dir), engine)
    worker = WriteWorker(repository, service, engine)
    app = create_app(settings, service=service, worker=worker, manage_lifespan=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://service") as client:
        assert (await client.get("/health/live")).status_code == 200
        assert (await client.get("/knowledge-bases")).status_code == 401
        headers = {
            "Authorization": "Bearer test-token",
            "X-StudyLoop-Subject": "alice",
            "Idempotency-Key": "create-kb",
        }
        created = await client.post(
            "/knowledge-bases",
            headers=headers,
            json={"name": "KB", "description": "private"},
        )
        assert created.status_code == 201
        kb = created.json()
        payload = {
            "name": "notes.txt",
            "kind": "file",
            "content_base64": base64.b64encode(b"Alpha").decode(),
            "parsed_blocks": [{"text": "Alpha", "metadata": {}}],
            "expected_revision": 0,
        }
        queued = await client.post(
            f"/knowledge-bases/{kb['id']}/documents",
            headers={**headers, "Idempotency-Key": "api-upload"},
            json=payload,
        )
        assert queued.status_code == 202
        assert set(queued.json()) == {"job_id", "knowledge_base_id"}
        documents = await client.get(
            f"/knowledge-bases/{kb['id']}/documents", headers=headers
        )
        assert documents.status_code == 200
        assert set(documents.json()) == {"documents", "total", "limit", "offset"}
        assert documents.json()["total"] == 1
        assert documents.json()["limit"] == 50
        assert documents.json()["offset"] == 0


async def test_revision_conflict_has_stable_http_status(repository, settings) -> None:
    engine = FakeEngine()
    service = KnowledgeService(repository, MaterialStore(settings.materials_dir), engine)
    worker = WriteWorker(repository, service, engine)
    app = create_app(settings, service=service, worker=worker, manage_lifespan=False)
    headers = {
        "Authorization": "Bearer test-token",
        "X-StudyLoop-Subject": "alice",
        "Idempotency-Key": "create-kb",
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://service"
    ) as client:
        kb = (
            await client.post(
                "/knowledge-bases",
                headers=headers,
                json={"name": "KB", "description": ""},
            )
        ).json()
        response = await client.post(
            f"/knowledge-bases/{kb['id']}/documents",
            headers={**headers, "Idempotency-Key": "wrong-revision"},
            json={
                "name": "notes.txt",
                "kind": "file",
                "content_base64": base64.b64encode(b"Alpha").decode(),
                "parsed_blocks": [{"text": "Alpha", "metadata": {}}],
                "expected_revision": 9,
            },
        )
        assert response.status_code == 409
        assert response.json()["error_code"] == "conflict"
