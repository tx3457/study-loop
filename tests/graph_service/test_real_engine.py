from __future__ import annotations

import base64
import asyncio
import json
import uuid
from dataclasses import replace

import pytest

pytest.importorskip("asyncpg")
pytest.importorskip("lightrag")

from graph_service.engine import LightRAGEngine
from graph_service.materials import MaterialStore
from graph_service.service import KnowledgeService
from graph_service.worker import WriteWorker
from lightrag_contract.support import build_embedding_func, deterministic_llm

pytestmark = [pytest.mark.asyncio, pytest.mark.lightrag_live]


def material_payload(text: str) -> dict:
    return {
        "name": "real-sdk.txt",
        "kind": "file",
        "content_base64": base64.b64encode(text.encode()).decode(),
        "parsed_blocks": [{"text": text, "metadata": {"page": 1}}],
        "expected_revision": 0,
    }


async def test_production_embedding_callback_uses_supported_sdk_client_config(
    settings, monkeypatch
) -> None:
    requests: list[dict] = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        headers = await reader.readuntil(b"\r\n\r\n")
        content_length = 0
        for line in headers.decode("latin-1").split("\r\n"):
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":", 1)[1].strip())
        requests.append(json.loads((await reader.readexactly(content_length)).decode()))
        body = json.dumps(
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.125] * 64}
                ],
                "model": "local-embedding",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        ).encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    import lightrag.llm.openai as sdk_openai

    monkeypatch.setattr(sdk_openai, "EMBEDDING_USE_BASE64", False)
    engine = LightRAGEngine(
        replace(
            settings,
            provider="openai",
            llm_model="local-chat",
            embedding_model="local-embedding",
            embedding_dim=64,
            llm_api_key="local-test-key",
            embedding_api_key="local-test-key",
            llm_base_url=f"http://127.0.0.1:{port}/v1",
            embedding_base_url=f"http://127.0.0.1:{port}/v1",
        )
    )
    try:
        _, embedding = engine._provider_callbacks()
        vectors = await embedding(["production callback probe"])
        assert vectors.shape == (1, 64)
        assert requests[0]["model"] == "local-embedding"
        assert "dimensions" not in requests[0]
    finally:
        server.close()
        await server.wait_closed()


async def test_workspace_rebuild_clears_llm_cache_only_when_requested(
    repository, settings
) -> None:
    engine = LightRAGEngine(
        settings,
        llm_func=deterministic_llm,
        embedding_func=build_embedding_func(),
    )
    workspace = f"cache_rebuild_{uuid.uuid4().hex}"
    try:
        async with engine._use(workspace):
            pass
        await repository.pool.execute(
            "INSERT INTO lightrag_llm_cache(workspace,id,original_prompt,return_value,cache_type) "
            "VALUES($1,'sentinel','prompt','response','extract')",
            workspace,
        )
        await engine.rebuild(workspace, [], [], clear_llm_cache=False)
        assert await repository.pool.fetchval(
            "SELECT count(*) FROM lightrag_llm_cache WHERE workspace=$1 AND id='sentinel'",
            workspace,
        ) == 1
        await engine.rebuild(workspace, [], [], clear_llm_cache=True)
        assert await repository.pool.fetchval(
            "SELECT count(*) FROM lightrag_llm_cache WHERE workspace=$1 AND id='sentinel'",
            workspace,
        ) == 0
    finally:
        await engine.close()


async def test_service_indexes_and_resolves_evidence_with_real_lightrag(
    repository, settings
) -> None:
    engine = LightRAGEngine(
        settings,
        llm_func=deterministic_llm,
        embedding_func=build_embedding_func(),
    )
    service = KnowledgeService(repository, MaterialStore(settings.materials_dir), engine)
    worker = WriteWorker(repository, service, engine)
    try:
        kb = await service.create_knowledge_base("real-owner", "Real SDK", "")
        await service.ingest_document(
            "real-owner",
            kb["id"],
            material_payload(
                "ENTITY[RealService] ENTITY[Evidence] REL[RealService|Evidence] service_marker"
            ),
            idempotency_key="real-sdk-insert",
        )
        await worker.process_one()
        result = await service.query("real-owner", kb["id"], "service_marker", 1, 1)
        assert len(result["evidence"]) == 1
        assert result["evidence"][0]["text"].endswith("service_marker")
        graph = await service.graph("real-owner", kb["id"], "RealService", 200, 400)
        assert graph["nodes"]
        assert all("id" in node and "label" in node for node in graph["nodes"])
        assert result["evidence"][0]["source_version_id"] in {
            source_id for node in graph["nodes"] for source_id in node["source_version_ids"]
        }
    finally:
        await engine.close()


async def test_corrected_destructive_mutations_reconstruct_only_live_sources(
    repository, settings, caplog
) -> None:
    engine = LightRAGEngine(
        settings,
        llm_func=deterministic_llm,
        embedding_func=build_embedding_func(),
    )
    service = KnowledgeService(repository, MaterialStore(settings.materials_dir), engine)
    worker = WriteWorker(repository, service, engine)
    kb = await service.create_knowledge_base("reconstruct-owner", "Reconstruct", "")
    try:
        await service.ingest_document(
            "reconstruct-owner",
            kb["id"],
            material_payload(
                "ENTITY[Alpha] ENTITY[Anchor] REL[Alpha|Anchor] old_alpha_marker"
            ),
            idempotency_key="alpha-v1",
        )
        await worker.process_one()
        alpha_doc = (await service.list_documents("reconstruct-owner", kb["id"]))[
            "documents"
        ][0]
        await service.ingest_document(
            "reconstruct-owner",
            kb["id"],
            {
                **material_payload(
                    "ENTITY[Beta] ENTITY[Anchor] REL[Beta|Anchor] beta_marker"
                ),
                "expected_revision": 1,
            },
            idempotency_key="beta-v1",
        )
        await worker.process_one()
        documents = (await service.list_documents("reconstruct-owner", kb["id"]))[
            "documents"
        ]
        beta_doc = next(document for document in documents if document["id"] != alpha_doc["id"])
        graph = await service.graph("reconstruct-owner", kb["id"], None, 200, 400)
        ids = {node["label"]: node["id"] for node in graph["nodes"]}
        await service.create_correction(
            "reconstruct-owner",
            kb["id"],
            {
                "kind": "rename_entity",
                "entity_id": ids["Alpha"],
                "label": "Unified",
                "expected_revision": 2,
            },
            idempotency_key="rename-alpha",
        )
        await worker.process_one()
        await service.create_correction(
            "reconstruct-owner",
            kb["id"],
            {
                "kind": "merge_entities",
                "entity_ids": [ids["Beta"]],
                "target_id": ids["Alpha"],
                "expected_revision": 3,
            },
            idempotency_key="merge-beta",
        )
        await worker.process_one()
        await service.delete_document(
            "reconstruct-owner", kb["id"], beta_doc["id"], 4, "delete-beta"
        )
        await worker.process_one()
        await service.ingest_document(
            "reconstruct-owner",
            kb["id"],
            {
                **material_payload(
                    "ENTITY[Alpha] ENTITY[NewAnchor] REL[Alpha|NewAnchor] new_alpha_marker"
                ),
                "expected_revision": 5,
            },
            idempotency_key="alpha-v2",
            document_id=alpha_doc["id"],
        )
        await worker.process_one()
        current = (await service.list_documents("reconstruct-owner", kb["id"]))[
            "documents"
        ][0]
        current_chunk = f"{current['version_id']}-chunk-000"
        workspace = (await repository.get_kb("reconstruct-owner", kb["id"]))["workspace"]
        graph_source = await repository.pool.fetchval(
            "SELECT properties->>'source_id' FROM lightrag_graph_nodes "
            "WHERE workspace=$1 AND id='Unified'",
            workspace,
        )
        vector_sources = await repository.pool.fetchval(
            "SELECT chunk_ids FROM lightrag_vdb_entity_deterministic_contract_v1_64d "
            "WHERE workspace=$1 AND entity_name='Unified'",
            workspace,
        )
        assert graph_source == current_chunk
        assert vector_sources == [current_chunk]
    finally:
        await engine.close()

    reopened = LightRAGEngine(
        settings,
        llm_func=deterministic_llm,
        embedding_func=build_embedding_func(),
    )
    reopened_service = KnowledgeService(
        repository, MaterialStore(settings.materials_dir), reopened
    )
    reopened_worker = WriteWorker(repository, reopened_service, reopened)
    try:
        caplog.clear()
        result = await reopened_service.query(
            "reconstruct-owner", kb["id"], "new_alpha_marker", 6, 6
        )
        assert {item["source_version_id"] for item in result["evidence"]} == {
            current["version_id"]
        }
        assert not any(
            "missing" in record.getMessage().lower()
            and "chunk" in record.getMessage().lower()
            for record in caplog.records
        )
        delete_job = await reopened_service.delete_knowledge_base(
            "reconstruct-owner", kb["id"], 6, "delete-corrected-kb"
        )
        assert await reopened_worker.process_one() == delete_job["job_id"]
        for table in (
            "lightrag_graph_nodes",
            "lightrag_graph_edges",
            "lightrag_doc_chunks",
            "lightrag_doc_status",
            "lightrag_vdb_entity_deterministic_contract_v1_64d",
            "lightrag_vdb_relation_deterministic_contract_v1_64d",
            "lightrag_vdb_chunks_deterministic_contract_v1_64d",
            "lightrag_llm_cache",
        ):
            assert await repository.pool.fetchval(
                f'SELECT count(*) FROM "{table}" WHERE workspace=$1', workspace
            ) == 0
    finally:
        await reopened.close()
