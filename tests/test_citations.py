"""Chunk-ID citation protocol contracts."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.citations import collect_search_evidence, resolve_citations
from services.tools import _search_document


def test_collect_and_resolve_only_server_observed_chunk_ids():
    registry = {}
    accepted = collect_search_evidence(
        json.dumps(
            {
                "document_id": "doc",
                "chunks": ["first evidence", "second evidence"],
                "chunk_ids": ["doc_chunk_7", "doc_chunk_12"],
            }
        ),
        registry,
    )

    resolution = resolve_citations(
        ["doc_chunk_12", "invented_chunk_99", "doc_chunk_12"],
        registry,
    )
    assert accepted == 2
    assert [citation.chunk_id for citation in resolution.citations] == ["doc_chunk_12"]
    assert resolution.citations[0].snippet == "second evidence"
    assert resolution.invalid_ids == ["invented_chunk_99"]


def test_misaligned_tool_payload_fails_closed_without_partial_registry_update():
    registry = {}
    accepted = collect_search_evidence(
        json.dumps(
            {
                "document_id": "doc",
                "chunks": ["one", "two"],
                "chunk_ids": ["doc_chunk_1"],
            }
        ),
        registry,
    )
    assert accepted == 0
    assert registry == {}


def test_non_json_tool_result_is_not_treated_as_evidence():
    registry = {}
    assert collect_search_evidence("tool failed", registry) == 0
    assert registry == {}


def test_search_document_returns_aligned_server_chunk_ids():
    retrieval = {
        "documents": [["a", "b", "c", "d"]],
        "ids": [["doc_chunk_1", "doc_chunk_2", "doc_chunk_3", "doc_chunk_4"]],
    }
    with patch(
        "services.tools.retrieve_with_rewrite",
        AsyncMock(return_value=retrieval),
    ):
        payload = json.loads(asyncio.run(_search_document("doc", "question")))

    assert payload == {
        "document_id": "doc",
        "chunks": ["a", "b", "c"],
        "chunk_ids": ["doc_chunk_1", "doc_chunk_2", "doc_chunk_3"],
    }
