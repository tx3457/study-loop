import asyncio
from dataclasses import dataclass

import httpx
import pytest

from services.knowledge_client import KnowledgeClient, KnowledgeServiceError


def test_request_binds_internal_identity_and_rejects_header_forgery():
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers["authorization"]
        seen["subject"] = request.headers["x-studyloop-subject"]
        return httpx.Response(200, json={"ok": True})

    client = KnowledgeClient(
        service_url="http://graph.internal:8080",
        service_token="shared-secret",
        transport=httpx.MockTransport(handler),
    )

    response = asyncio.run(client.request("GET", "/knowledge-bases", owner_id="alice"))
    assert response.json() == {"ok": True}
    assert seen == {
        "url": "http://graph.internal:8080/knowledge-bases",
        "authorization": "Bearer shared-secret",
        "subject": "alice",
    }

    with pytest.raises(ValueError):
        asyncio.run(
            client.request(
                "GET",
                "/knowledge-bases",
                owner_id="alice",
                headers={"X-StudyLoop-Subject": "mallory"},
            )
        )


def test_request_never_accepts_an_arbitrary_absolute_url():
    client = KnowledgeClient(
        service_url="http://graph.internal:8080",
        service_token="shared-secret",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )
    with pytest.raises(ValueError):
        asyncio.run(client.request("GET", "http://attacker.invalid/steal", owner_id="alice"))
    with pytest.raises(ValueError):
        asyncio.run(client.request("GET", "/../health/private", owner_id="alice"))


def test_network_timeout_becomes_safe_503_without_exception_text():
    secret = "postgresql://secret@db/internal"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(secret, request=request)

    client = KnowledgeClient(
        service_url="http://graph.internal:8080",
        service_token="shared-secret",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(KnowledgeServiceError) as caught:
        asyncio.run(client.request("GET", "/knowledge-bases", owner_id="alice"))
    assert caught.value.status_code == 503
    assert secret not in str(caught.value)


def test_default_read_timeout_covers_graph_query_budget(monkeypatch):
    monkeypatch.delenv("KNOWLEDGE_HTTP_TIMEOUT_SECONDS", raising=False)
    client = KnowledgeClient(
        service_url="http://graph.internal:8080",
        service_token="shared-secret",
    )
    assert client._timeout.read == 100


def test_http_timeout_is_positive_and_bounded(monkeypatch):
    for value in ("0", "301", "not-a-number"):
        monkeypatch.setenv("KNOWLEDGE_HTTP_TIMEOUT_SECONDS", value)
        with pytest.raises(ValueError):
            KnowledgeClient(
                service_url="http://graph.internal:8080",
                service_token="shared-secret",
            )


def test_validate_scope_fails_closed_for_dirty_and_changed_indexes():
    responses = iter(
        [
            {"knowledge_base_id": "kb-1", "revision": 3, "epoch": 7, "status": "dirty"},
            {"knowledge_base_id": "kb-1", "revision": 4, "epoch": 7, "status": "ready"},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(responses))

    client = KnowledgeClient(
        service_url="http://graph.internal:8080",
        service_token="shared-secret",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(KnowledgeServiceError) as dirty:
        asyncio.run(client.get_scope("kb-1", "alice"))
    assert dirty.value.status_code == 503

    with pytest.raises(KnowledgeServiceError) as changed:
        asyncio.run(client.validate_scope("kb-1", "alice", 3, 7))
    assert changed.value.status_code == 409


@dataclass(frozen=True)
class _FrozenPage:
    url: str
    title: str
    text: str
    content_hash: str
    fetched_at: str


def test_store_web_snapshot_accepts_frozen_page_shape_and_pins_session():
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.update(__import__("json").loads(request.content))
        return httpx.Response(201, json={"id": "snap-1", **seen})

    client = KnowledgeClient(
        service_url="http://graph.internal:8080",
        service_token="shared-secret",
        transport=httpx.MockTransport(handler),
    )
    page = _FrozenPage(
        url="https://example.test/a",
        title="A",
        text="bounded text",
        content_hash="a" * 64,
        fetched_at="2026-09-20T01:02:03Z",
    )
    result = asyncio.run(
        client.store_web_snapshot("alice", "session-1", page, "2026-09-27T01:02:03Z")
    )
    assert result["id"] == "snap-1"
    assert seen == {
        "session_id": "session-1",
        "url": page.url,
        "title": page.title,
        "text": page.text,
        "content_hash": page.content_hash,
        "fetched_at": page.fetched_at,
        "expires_at": "2026-09-27T01:02:03Z",
    }
