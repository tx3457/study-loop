import base64
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import knowledge
from services.auth import require_user_id
from services.mcp_client import mcp_registry
from services.tool_registry import EffectMode, Tool, ToolMetadata


def _app(owner_id: str = "alice") -> FastAPI:
    app = FastAPI()
    app.include_router(knowledge.router)
    app.dependency_overrides[require_user_id] = lambda: owner_id
    return app


def _response(method: str, path: str, status: int, payload: dict) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        request=httpx.Request(method, f"http://graph.internal{path}"),
    )


def test_disabled_capability_is_side_effect_free_and_does_not_block_legacy_service():
    client = TestClient(_app())
    with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "false"}, clear=False), patch.object(
        knowledge.knowledge_client, "request", AsyncMock()
    ) as request:
        response = client.get("/knowledge-bases/capabilities")
    assert response.status_code == 200
    assert response.json() == {
        "enabled": False,
        "available": False,
        "web_search_available": False,
    }
    request.assert_not_awaited()


def test_merge_accepts_one_source_into_a_distinct_target():
    payload = {"kind": "merge_entities", "entity_ids": ["entity-source"],
               "target_id": "entity-target", "expected_revision": 4}
    upstream = _response("POST", "/knowledge-bases/kb-1/corrections", 202,
                         {"job_id": "merge-job", "knowledge_base_id": "kb-1"})
    with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}), patch.object(
        knowledge.knowledge_client, "request", AsyncMock(return_value=upstream)
    ) as request:
        response = TestClient(_app()).post(
            "/knowledge-bases/kb-1/corrections", json=payload,
            headers={"Idempotency-Key": "merge-distinct-entities"},
        )
    assert response.status_code == 202
    assert request.await_args.kwargs["json"] == payload
    assert request.await_args.kwargs["owner_id"] == "alice"


def test_merge_rejects_empty_duplicate_or_self_sources_before_proxying():
    for sources in ([], ["target"], ["source", "source"], ["source", "target"]):
        with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}), patch.object(
            knowledge.knowledge_client, "request", AsyncMock()
        ) as request:
            response = TestClient(_app()).post(
                "/knowledge-bases/kb-1/corrections",
                json={"kind": "merge_entities", "entity_ids": sources,
                      "target_id": "target", "expected_revision": 0},
                headers={"Idempotency-Key": "invalid-merge-request"},
            )
        assert response.status_code == 422
        request.assert_not_awaited()


def test_capability_uses_registered_local_ddg_tool_when_graph_is_available():
    async def handler(**kwargs):
        return "[]"

    tool = Tool(
        name="mcp_ddg_search",
        description="test DDG",
        parameters_schema={"type": "object", "properties": {}},
        handler=handler,
        metadata=ToolMetadata(max_retries=0, effect_mode=EffectMode.READ_ONLY),
    )
    previous = mcp_registry.get(tool.name)
    mcp_registry.register(tool)
    upstream = _response(
        "GET",
        "/knowledge-bases/capabilities",
        200,
        {"available": True, "web_search_available": False},
    )
    try:
        with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}, clear=False), patch.object(
            knowledge.knowledge_client, "request", AsyncMock(return_value=upstream)
        ):
            response = TestClient(_app()).get("/knowledge-bases/capabilities")
    finally:
        mcp_registry.unregister(tool.name, expected_tool=tool)
        if previous is not None:
            mcp_registry.register(previous)
    assert response.json() == {
        "enabled": True,
        "available": True,
        "web_search_available": True,
    }


def test_create_forwards_only_resolved_owner_and_validated_idempotency_key():
    client = TestClient(_app("trusted-owner"))
    upstream = _response(
        "POST",
        "/knowledge-bases",
        201,
        {
            "id": "kb-1",
            "name": "ML",
            "description": "notes",
            "status": "ready",
            "revision": 0,
            "epoch": 0,
            "document_count": 0,
            "created_at": "2026-09-20T01:00:00Z",
        },
    )
    with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}, clear=False), patch.object(
        knowledge.knowledge_client, "request", AsyncMock(return_value=upstream)
    ) as request:
        response = client.post(
            "/knowledge-bases",
            headers={"Idempotency-Key": "create-kb-0001"},
            json={"name": "ML", "description": "notes", "owner_id": "mallory"},
        )
    assert response.status_code == 422
    request.assert_not_awaited()

    with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}, clear=False), patch.object(
        knowledge.knowledge_client, "request", AsyncMock(return_value=upstream)
    ) as request:
        response = client.post(
            "/knowledge-bases",
            headers={"Idempotency-Key": "create-kb-0001"},
            json={"name": "ML", "description": "notes"},
        )
    assert response.status_code == 201
    assert request.await_args.kwargs == {
        "owner_id": "trusted-owner",
        "json": {"name": "ML", "description": "notes"},
        "headers": {"Idempotency-Key": "create-kb-0001"},
    }


def test_upload_rejects_malicious_filename_and_parser_error_metadata():
    client = TestClient(_app())
    with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}, clear=False), patch.object(
        knowledge.knowledge_client, "request", AsyncMock()
    ) as request:
        bad_name = client.post(
            "/knowledge-bases/kb-1/documents/upload",
            headers={"Idempotency-Key": "upload-doc-0001"},
            data={"expected_revision": "0"},
            files={"file": ("../secret.txt", b"hello", "text/plain")},
        )
    assert bad_name.status_code == 422
    request.assert_not_awaited()

    parsed = [SimpleNamespace(page_content="provider secret", metadata={"error": "boom"})]
    with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}, clear=False), patch.object(
        knowledge, "parse_upload", AsyncMock(return_value=parsed)
    ), patch.object(knowledge.knowledge_client, "request", AsyncMock()) as request:
        failed_parse = client.post(
            "/knowledge-bases/kb-1/documents/upload",
            headers={"Idempotency-Key": "upload-doc-0002"},
            data={"expected_revision": "0"},
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
    assert failed_parse.status_code == 422
    assert "secret" not in failed_parse.text
    request.assert_not_awaited()


def test_upload_forwards_parsed_blocks_revision_and_same_body_on_replay():
    client = TestClient(_app("alice"))
    parsed = [SimpleNamespace(page_content="first block", metadata={"page": 1})]
    upstream = _response(
        "POST", "/knowledge-bases/kb-1/documents", 202, {"job_id": "job-1", "knowledge_base_id": "kb-1"}
    )
    request = AsyncMock(return_value=upstream)
    calls = []

    async def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return upstream

    request.side_effect = capture
    with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}, clear=False), patch.object(
        knowledge, "parse_upload", AsyncMock(return_value=parsed)
    ), patch.object(knowledge.knowledge_client, "request", request):
        for _ in range(2):
            response = client.post(
                "/knowledge-bases/kb-1/documents/upload",
                headers={"Idempotency-Key": "upload-doc-replay-1"},
                data={"expected_revision": "4"},
                files={"file": ("notes.txt", b"hello", "text/plain")},
            )
            assert response.status_code == 202

    assert calls[0] == calls[1]
    args, kwargs = calls[0]
    assert args == ("POST", "/knowledge-bases/kb-1/documents")
    assert kwargs["owner_id"] == "alice"
    assert kwargs["headers"] == {"Idempotency-Key": "upload-doc-replay-1"}
    assert kwargs["json"] == {
        "name": "notes.txt",
        "kind": "file",
        "content_base64": base64.b64encode(b"hello").decode("ascii"),
        "parsed_blocks": [{"text": "first block", "metadata": {"page": 1}}],
        "expected_revision": 4,
    }


def test_plain_text_upload_uses_the_real_parser_before_forwarding():
    client = TestClient(_app("alice"))
    upstream = _response(
        "POST",
        "/knowledge-bases/kb-1/documents",
        202,
        {"job_id": "job-real-parse", "knowledge_base_id": "kb-1"},
    )
    with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}, clear=False), patch.object(
        knowledge.knowledge_client, "request", AsyncMock(return_value=upstream)
    ) as request:
        response = client.post(
            "/knowledge-bases/kb-1/documents/upload",
            headers={"Idempotency-Key": "upload-real-parse-1"},
            data={"expected_revision": "0"},
            files={"file": ("notes.txt", "梯度下降\n".encode(), "text/plain")},
        )
    assert response.status_code == 202
    payload = request.await_args.kwargs["json"]
    assert payload["parsed_blocks"][0]["text"] == "梯度下降"


def test_import_legacy_copies_authorized_chunks_without_old_store_mutation():
    client = TestClient(_app("alice"))
    upstream = _response(
        "POST", "/knowledge-bases/kb-1/documents", 202, {"job_id": "job-2", "knowledge_base_id": "kb-1"}
    )
    collection = SimpleNamespace(name="old.md", metadata={"owner_id": "alice"})
    resolve = AsyncMock(return_value=collection)
    read_index = AsyncMock(
        return_value={
            "all_docs": ["chunk one", "chunk two"],
            "all_ids": ["c1", "c2"],
        }
    )
    with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}, clear=False), patch.object(
        knowledge, "_get_public_document_collection", resolve
    ), patch.object(
        knowledge, "_get_bm25_index", read_index
    ), patch.object(knowledge.knowledge_client, "request", AsyncMock(return_value=upstream)) as request:
        response = client.post(
            "/knowledge-bases/kb-1/documents/import-legacy",
            headers={"Idempotency-Key": "legacy-copy-001"},
            json={"legacy_document_id": "old.md", "expected_revision": 2},
        )
    assert response.status_code == 202
    resolve.assert_awaited_once_with("old.md", "alice")
    read_index.assert_awaited_once_with(collection, "old.md", "alice")
    payload = request.await_args.kwargs["json"]
    assert payload["kind"] == "legacy_copy"
    assert payload["legacy_document_id"] == "old.md"
    assert payload["parsed_blocks"] == [
        {"text": "chunk one", "metadata": {"legacy_chunk_id": "c1"}},
        {"text": "chunk two", "metadata": {"legacy_chunk_id": "c2"}},
    ]


def test_graph_limits_and_revision_are_forwarded_but_unbounded_values_are_rejected():
    client = TestClient(_app())
    upstream = _response(
        "GET", "/knowledge-bases/kb-1/graph", 200, {"nodes": [], "edges": [], "truncated": False, "revision": 2, "epoch": 3}
    )
    with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}, clear=False), patch.object(
        knowledge.knowledge_client, "request", AsyncMock(return_value=upstream)
    ) as request:
        ok = client.get("/knowledge-bases/kb-1/graph?node_limit=200&edge_limit=400&search=ml")
        bad = client.get("/knowledge-bases/kb-1/graph?node_limit=201&edge_limit=400")
    assert ok.status_code == 200
    assert bad.status_code == 422
    assert request.await_count == 1
    assert request.await_args.args[1] == "/knowledge-bases/kb-1/graph?node_limit=200&edge_limit=400&search=ml"


def test_graph_frontend_aliases_are_translated_to_internal_parameter_names():
    client = TestClient(_app())
    upstream = _response(
        "GET",
        "/knowledge-bases/kb-1/graph",
        200,
        {"nodes": [], "edges": [], "truncated": False, "revision": 2, "epoch": 3},
    )
    with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}, clear=False), patch.object(
        knowledge.knowledge_client, "request", AsyncMock(return_value=upstream)
    ) as request:
        response = client.get(
            "/knowledge-bases/kb-1/graph?limit_nodes=17&limit_edges=31&focus=entity-7"
        )
    assert response.status_code == 200
    assert request.await_args.args[1] == (
        "/knowledge-bases/kb-1/graph?node_limit=17&edge_limit=31&focus_id=entity-7"
    )


def test_delete_forwards_revision_in_the_internal_query_contract():
    client = TestClient(_app("alice"))
    upstream = _response(
        "DELETE",
        "/knowledge-bases/kb-1/documents/doc-1?expected_revision=6",
        202,
        {"job_id": "job-delete", "knowledge_base_id": "kb-1"},
    )
    with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}, clear=False), patch.object(
        knowledge.knowledge_client, "request", AsyncMock(return_value=upstream)
    ) as request:
        response = client.request(
            "DELETE",
            "/knowledge-bases/kb-1/documents/doc-1",
            headers={"Idempotency-Key": "delete-document-1"},
            json={"expected_revision": 6},
        )
    assert response.status_code == 202
    assert request.await_args.args == (
        "DELETE",
        "/knowledge-bases/kb-1/documents/doc-1?expected_revision=6",
    )
    assert request.await_args.kwargs["json"] is None


def test_knowledge_base_delete_accepts_frontend_json_and_rejects_string_revision():
    client = TestClient(_app("alice"))
    upstream = _response(
        "DELETE",
        "/knowledge-bases/kb-1?expected_revision=9",
        202,
        {"job_id": "job-delete-kb", "knowledge_base_id": "kb-1"},
    )
    with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}, clear=False), patch.object(
        knowledge.knowledge_client, "request", AsyncMock(return_value=upstream)
    ) as request:
        response = client.request(
            "DELETE",
            "/knowledge-bases/kb-1",
            headers={"Idempotency-Key": "delete-kb-json-1"},
            json={"expected_revision": 9},
        )
        invalid = client.request(
            "DELETE",
            "/knowledge-bases/kb-1",
            headers={"Idempotency-Key": "delete-kb-json-2"},
            json={"expected_revision": "9"},
        )
    assert response.status_code == 202
    assert invalid.status_code == 422
    assert request.await_count == 1
    assert request.await_args.args == (
        "DELETE",
        "/knowledge-bases/kb-1?expected_revision=9",
    )


def test_rebuild_forwards_identity_revision_and_idempotency_key():
    client = TestClient(_app("trusted-owner"))
    upstream = _response(
        "POST",
        "/knowledge-bases/kb-1/rebuild",
        202,
        {"job_id": "job-rebuild", "knowledge_base_id": "kb-1"},
    )
    with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}, clear=False), patch.object(
        knowledge.knowledge_client, "request", AsyncMock(return_value=upstream)
    ) as request:
        response = client.post(
            "/knowledge-bases/kb-1/rebuild",
            headers={"Idempotency-Key": "rebuild-kb-0001"},
            json={"expected_revision": 12},
        )
        invalid = client.post(
            "/knowledge-bases/kb-1/rebuild",
            headers={"Idempotency-Key": "rebuild-kb-0002"},
            json={"expected_revision": "12"},
        )
    assert response.status_code == 202
    assert invalid.status_code == 422
    assert request.await_count == 1
    assert request.await_args.args == ("POST", "/knowledge-bases/kb-1/rebuild")
    assert request.await_args.kwargs == {
        "owner_id": "trusted-owner",
        "json": {"expected_revision": 12},
        "headers": {"Idempotency-Key": "rebuild-kb-0001"},
    }
