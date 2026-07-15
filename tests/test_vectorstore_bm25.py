"""Pure BM25 helper contracts; importing this file has no service side effects."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from services import bm25 as bm25_service
from services.tokenization import tokenize_for_bm25


class _RecordingBm25:
    def __init__(self, scores):
        self.scores = scores
        self.queries = []

    def get_scores(self, tokens):
        self.queries.append(tokens)
        return self.scores


def test_index_uses_shared_document_tokenizer():
    captured = {}

    class _CapturingIndex:
        def __init__(self, corpus):
            captured["corpus"] = corpus

    with patch.object(bm25_service, "BM25Okapi", _CapturingIndex):
        index = bm25_service.build_bm25_index(["Neural-network RAG", "机器学习"])

    assert index is not None
    assert captured["corpus"] == [
        tokenize_for_bm25("Neural-network RAG"),
        tokenize_for_bm25("机器学习"),
    ]


def test_all_empty_document_tokens_do_not_build_invalid_index():
    assert bm25_service.build_bm25_index(["，。", "!?"]) is None


def test_rank_uses_shared_query_tokenizer_and_stable_tie_break():
    bm25 = _RecordingBm25([0.1, 0.9])
    result = bm25_service.rank_bm25(bm25, "Neural-Network", 2)

    assert bm25.queries == [["neural", "network"]]
    assert result == [(1, 0.9), (0, 0.1)]


def test_empty_query_does_not_return_arbitrary_bm25_hits():
    bm25 = _RecordingBm25([1.0])
    assert bm25_service.rank_bm25(bm25, "，。!?", 1) == []
    assert bm25.queries == []


def test_equal_scores_keep_corpus_position_order():
    bm25 = _RecordingBm25([0.5, 0.5, 0.4])
    assert bm25_service.rank_bm25(bm25, "RAG", 3) == [
        (0, 0.5),
        (1, 0.5),
        (2, 0.4),
    ]
