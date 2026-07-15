"""Pure BM25 construction and ranking helpers."""

from __future__ import annotations

from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from services.tokenization import tokenize_for_bm25


def build_bm25_index(documents: Sequence[str]) -> BM25Okapi | None:
    """Build an index, or return ``None`` when every document is token-empty."""

    tokenized_documents = [tokenize_for_bm25(document) for document in documents]
    if not any(tokenized_documents):
        return None
    return BM25Okapi(tokenized_documents)


def rank_bm25(
    index: BM25Okapi | None,
    query: str,
    limit: int,
) -> list[tuple[int, float]]:
    """Return ``(corpus_position, score)`` sorted by score then position."""

    if index is None or limit <= 0:
        return []
    query_tokens = tokenize_for_bm25(query)
    if not query_tokens:
        return []
    scores = index.get_scores(query_tokens)
    return sorted(
        ((position, float(score)) for position, score in enumerate(scores)),
        key=lambda item: (-item[1], item[0]),
    )[:limit]
