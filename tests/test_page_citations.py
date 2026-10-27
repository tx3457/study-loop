"""PDF page numbers travel from parsing through retrieval to Autonomous citations."""

import asyncio
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import chromadb
from fastapi.testclient import TestClient
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))

import routers.documents as documents_router
import services.vectorstore as vectorstore
from main import app
from services.autonomous_snapshot import (
    AutonomousSession, StepRecord, session_from_payload, session_to_payload,
)
from services.citations import EvidenceChunk, collect_search_evidence, resolve_citations
from services.parser import pdf_page_number
from services.tools import _search_document
from services.vectorstore import DEFAULT_DOCUMENT_OWNER


def _embedding_response(texts):
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=[float(i), 1.0]) for i, _ in enumerate(texts)]
    )


def _stop_chroma_client(client) -> None:
    stop = getattr(getattr(client, "_system", None), "stop", None)
    if callable(stop):
        stop()


def test_pdf_page_number_is_one_based_and_pdf_only():
    assert pdf_page_number({"file_type": "pdf", "page": 0}) == 1
    assert pdf_page_number({"file_type": "PDF", "page": 11}) == 12
    assert pdf_page_number({"file_type": "md", "page": 0}) is None
    assert pdf_page_number({"file_type": "pdf"}) is None
    assert pdf_page_number({"file_type": "pdf", "page": True}) is None
    assert pdf_page_number({"file_type": "pdf", "page": -1}) is None


class TestUploadPassesPageNumbers(unittest.TestCase):
    def test_pdf_chunks_are_indexed_with_their_page(self):
        pages = [
            Document(page_content="第一页讲梯度下降。", metadata={"file_type": "pdf", "page": 0}),
            Document(page_content="   ", metadata={"file_type": "pdf", "page": 1}),
            Document(page_content="第三页讲反向传播。", metadata={"file_type": "pdf", "page": 2}),
        ]
        index = AsyncMock(return_value=2)

        with (
            patch.object(documents_router, "parse_upload", AsyncMock(return_value=pages)),
            patch.object(documents_router, "deal_document", index),
        ):
            response = TestClient(app, raise_server_exceptions=False).post(
                "/documents/upload",
                files={"file": ("讲义.pdf", b"%PDF-placeholder", "application/pdf")},
            )

        self.assertEqual(response.status_code, 200)
        index.assert_awaited_once_with(
            "讲义.pdf",
            "讲义.pdf",
            ["第一页讲梯度下降。", "第三页讲反向传播。"],
            owner_id=DEFAULT_DOCUMENT_OWNER,
            page_numbers=[1, 3],
        )


class TestRetrievalReturnsPageNumbers(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        vectorstore.start_vectorstore_io()
        vectorstore.clear_bm25_cache()

    async def asyncTearDown(self):
        vectorstore.clear_bm25_cache()
        await vectorstore.shutdown_vectorstore_io()

    async def test_hybrid_results_carry_stored_page_numbers(self):
        chunks = ["梯度下降沿负梯度方向更新参数", "反向传播用链式法则计算梯度"]
        with ExitStack() as stack:
            temp_dir = stack.enter_context(tempfile.TemporaryDirectory())
            client = chromadb.PersistentClient(path=str(Path(temp_dir) / "chroma"))
            stack.callback(_stop_chroma_client, client)
            stack.enter_context(patch.object(vectorstore, "chromadb_client", client))
            stack.enter_context(patch.object(
                vectorstore, "_embed", AsyncMock(side_effect=_embedding_response)
            ))

            await vectorstore.deal_document(
                "讲义.pdf", "讲义.pdf", chunks,
                owner_id=DEFAULT_DOCUMENT_OWNER, page_numbers=[4, None],
            )
            result = await vectorstore.hybrid_query_document(
                "讲义.pdf", "链式法则", n_results=2, enable_rerank=False,
                owner_id=DEFAULT_DOCUMENT_OWNER,
            )

        by_text = dict(zip(result["documents"][0], result["page_numbers"][0]))
        self.assertEqual(set(by_text), set(chunks))
        self.assertEqual(by_text[chunks[0]], 4)
        self.assertIsNone(by_text[chunks[1]])

    async def test_misaligned_page_numbers_are_rejected_before_any_io(self):
        embed = AsyncMock()
        with patch.object(vectorstore, "_embed", embed):
            with self.assertRaises(ValueError):
                await vectorstore.deal_document(
                    "讲义.pdf", "讲义.pdf", ["a", "b"],
                    owner_id=DEFAULT_DOCUMENT_OWNER, page_numbers=[1],
                )
        embed.assert_not_awaited()

    async def test_multi_query_merge_keeps_each_chunks_page(self):
        returns = {
            "原q": {"documents": [["共享", "仅q"]], "ids": [["c", "x1"]],
                   "page_numbers": [[7, None]]},
            "变体": {"documents": [["共享", "仅变体"]], "ids": [["c", "x2"]],
                   "page_numbers": [[7, 2]]},
        }

        async def _hybrid(document_id, query, n_results=5, *, owner_id):
            return returns[query]

        async def _mq(query, n=None):
            return [query, "变体"]

        with patch.object(vectorstore, "multiquery_enabled", return_value=True), \
             patch.object(vectorstore, "hyde_enabled", return_value=False), \
             patch.object(vectorstore, "multi_query_rewrite", _mq), \
             patch.object(vectorstore, "hybrid_query_document", _hybrid):
            out = await vectorstore.retrieve_with_rewrite(
                "doc", "原q", n_results=3, owner_id=DEFAULT_DOCUMENT_OWNER
            )

        self.assertEqual(
            dict(zip(out["ids"][0], out["page_numbers"][0])),
            {"c": 7, "x1": None, "x2": 2},
        )


def _search(retrieval):
    with patch("services.tools.retrieve_with_rewrite", AsyncMock(return_value=retrieval)):
        return json.loads(asyncio.run(_search_document("doc.pdf", "q", DEFAULT_DOCUMENT_OWNER)))


def test_search_payload_carries_page_numbers_aligned_with_chunk_ids():
    payload = _search({
        "documents": [["a", "b", "c", "d"]],
        "ids": [["doc_chunk_1", "doc_chunk_2", "doc_chunk_3", "doc_chunk_4"]],
        "page_numbers": [[2, None, 5, 9]],
    })
    assert payload["chunk_ids"] == ["doc_chunk_1", "doc_chunk_2", "doc_chunk_3"]
    assert payload["page_numbers"] == [2, None, 5]


def test_search_payload_omits_page_numbers_when_no_chunk_has_a_page():
    payload = _search({
        "documents": [["a"]], "ids": [["doc_chunk_1"]], "page_numbers": [[None]],
    })
    assert "page_numbers" not in payload


def test_search_rejects_page_numbers_misaligned_with_ids():
    try:
        _search({"documents": [["a", "b"]], "ids": [["x", "y"]], "page_numbers": [[1]]})
    except ValueError:
        return
    raise AssertionError("misaligned page numbers must not reach the model")


def test_citation_resolves_the_page_of_observed_evidence():
    registry = {}
    accepted = collect_search_evidence(
        json.dumps({
            "document_id": "doc.pdf",
            "chunks": ["first", "second"],
            "chunk_ids": ["doc_chunk_1", "doc_chunk_2"],
            "page_numbers": [3, None],
        }),
        registry,
    )
    resolution = resolve_citations(["doc_chunk_1", "doc_chunk_2"], registry)

    assert accepted == 2
    assert [c.page_number for c in resolution.citations] == [3, None]


def test_malformed_page_numbers_fail_closed():
    for page_numbers in ([True], [0], ["3"], [1, 2], "3"):
        registry = {}
        accepted = collect_search_evidence(
            json.dumps({
                "document_id": "doc.pdf",
                "chunks": ["first"],
                "chunk_ids": ["doc_chunk_1"],
                "page_numbers": page_numbers,
            }),
            registry,
        )
        assert accepted == 0, page_numbers
        assert registry == {}


def _paused_session(evidence_registry):
    return AutonomousSession(
        conversation_id="conv_1", user_id="learner", document_id="doc.pdf",
        messages=[
            {"role": "system", "content": "Grounded tutor"},
            {"role": "user", "content": "Explain"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "ask_1", "type": "function", "function": {
                    "name": "ask_user", "arguments": '{"question":"Which part?"}',
                },
            }]},
        ],
        plan=[], steps=[StepRecord(round_index=0, tool_name="ask_user")],
        tools_called=[], rounds_used=1, evidence_registry=evidence_registry,
        grounding_required=True, pending_ask_call_id="ask_1", registry_sha256="a" * 64,
    )


class TestPausedSessionsKeepPageNumbers(unittest.TestCase):
    def test_page_number_survives_pause_and_resume(self):
        evidence = {
            "doc_chunk_1": EvidenceChunk("doc_chunk_1", "doc.pdf", "first", 1, page_number=3),
            "doc_chunk_2": EvidenceChunk("doc_chunk_2", "doc.pdf", "second", 2),
        }
        restored = session_from_payload("conv_1", session_to_payload(_paused_session(evidence)))
        self.assertEqual(restored.evidence_registry, evidence)

    def test_evidence_without_a_page_keeps_the_pre_page_snapshot_shape(self):
        # Older builds reject unknown evidence keys, so a rollback must still be
        # able to resume sessions that cite no PDF page.
        evidence = {"doc_chunk_1": EvidenceChunk("doc_chunk_1", "doc.pdf", "first", 1)}
        payload = session_to_payload(_paused_session(evidence))

        self.assertEqual(
            set(payload["evidence_registry"]["doc_chunk_1"]),
            {"chunk_id", "document_id", "text", "rank"},
        )
        restored = session_from_payload("conv_1", payload)
        self.assertIsNone(restored.evidence_registry["doc_chunk_1"].page_number)
