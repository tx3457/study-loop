"""Weighted RRF: the 1:1 default must reproduce the previous plain RRF exactly."""

import asyncio
import random
from unittest.mock import AsyncMock, patch

import pytest

from services import vectorstore


def _previous_rrf(vec_ids, vec_docs, bm25_ids, bm25_docs, limit):
    # The fusion as it was before weights existed, kept verbatim as the oracle.
    K = 60
    rrf_scores, id_to_doc = {}, {}
    for rank, doc_id in enumerate(vec_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (K + rank + 1)
        id_to_doc[doc_id] = vec_docs[rank]
    for rank, doc_id in enumerate(bm25_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (K + rank + 1)
        id_to_doc[doc_id] = bm25_docs[rank]
    top = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:limit]
    return top, [id_to_doc[i] for i in top]


def test_equal_weights_reproduce_plain_rrf_on_random_rankings():
    rng = random.Random(7)
    for _ in range(500):
        pool = [f"c{i}" for i in range(rng.randint(1, 30))]
        vec = rng.sample(pool, rng.randint(0, len(pool)))
        bm25 = rng.sample(pool, rng.randint(0, len(pool)))
        limit = rng.randint(1, 40)
        expected = _previous_rrf(vec, [f"t{i}" for i in vec], bm25, [f"t{i}" for i in bm25], limit)
        actual = vectorstore.weighted_rrf(
            [(1.0, vec, [f"t{i}" for i in vec]), (1.0, bm25, [f"t{i}" for i in bm25])], limit
        )
        assert actual == expected


def test_weights_shift_the_ranking_and_zero_removes_a_retriever():
    vec, bm25 = ["a", "b", "c"], ["c", "b", "a"]
    docs = lambda ids: [i.upper() for i in ids]  # noqa: E731
    equal = vectorstore.weighted_rrf([(1, vec, docs(vec)), (1, bm25, docs(bm25))], 3)
    dense_led = vectorstore.weighted_rrf([(3, vec, docs(vec)), (1, bm25, docs(bm25))], 3)
    dense_only = vectorstore.weighted_rrf([(1, vec, docs(vec)), (0, bm25, docs(bm25))], 3)
    # a and c are 1st and 3rd once each (1/61 + 1/63) and edge out b's 2/62; their
    # exact tie keeps first-seen order.
    assert equal[0] == ["a", "c", "b"]
    assert dense_led == (["a", "b", "c"], ["A", "B", "C"])
    assert dense_only[0] == vec
    bm25_led = vectorstore.weighted_rrf([(1, vec, docs(vec)), (3, bm25, docs(bm25))], 3)
    assert bm25_led[0] == ["c", "b", "a"]


@pytest.mark.parametrize(
    ("dense", "bm25", "expected"),
    [
        (None, None, (1.0, 1.0)),
        ("3", "1", (3.0, 1.0)),
        ("1", "0", (1.0, 0.0)),
        ("-2", "1", (1.0, 1.0)),  # negative is invalid, not a reversal
        ("nan", "1", (1.0, 1.0)),
        ("abc", "0.5", (1.0, 0.5)),
        ("0", "0", (1.0, 1.0)),  # both off would return nothing
    ],
)
def test_weights_are_validated_from_the_environment(monkeypatch, dense, bm25, expected):
    for name, value in (("HYBRID_DENSE_WEIGHT", dense), ("HYBRID_BM25_WEIGHT", bm25)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert vectorstore.hybrid_fusion_weights() == expected


def test_zero_bm25_weight_skips_bm25_scoring(monkeypatch):
    monkeypatch.setenv("HYBRID_DENSE_WEIGHT", "1")
    monkeypatch.setenv("HYBRID_BM25_WEIGHT", "0")

    class Embedding:
        data = [type("Row", (), {"embedding": [0.1, 0.2]})()]

    async def run_io(function, *args, operation_name, **kwargs):
        return function(*args, **kwargs)

    collection = type("Collection", (), {
        "query": lambda self, **_: {"ids": [["v1", "v2"]], "documents": [["V1", "V2"]]},
    })()
    rank = AsyncMock()
    with patch.object(vectorstore, "_get_public_document_collection", AsyncMock(return_value=collection)), \
            patch.object(vectorstore, "_embed", AsyncMock(return_value=Embedding())), \
            patch.object(vectorstore, "_run_chroma_io", side_effect=run_io), \
            patch.object(vectorstore, "_get_bm25_index", AsyncMock(return_value={
                "all_docs": ["B1"], "all_ids": ["b1"], "bm25": object()})), \
            patch.object(vectorstore, "rank_bm25", rank):
        result = asyncio.run(vectorstore.hybrid_query_document(
            "doc", "query", n_results=2, enable_rerank=False, owner_id="owner"
        ))
    rank.assert_not_called()
    assert result["ids"] == [["v1", "v2"]]
