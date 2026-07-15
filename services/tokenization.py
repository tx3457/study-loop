"""Deterministic, dependency-free tokenization for BM25 retrieval."""

from __future__ import annotations

import re
import unicodedata


_TOKEN_RE = re.compile(
    r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+"
)
BM25_TOKENIZER_ID = "nfkc_casefold_ascii_words_cjk_unigram_bigram_v1"


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize Latin text by word and CJK text by unigram plus bigram.

    NFKC normalization keeps full-width forms comparable with their ASCII
    equivalents.  CJK bigrams add local word-order signal without introducing
    an external segmentation dependency, while unigrams preserve recall for
    short queries and uncommon terms.
    """

    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []

    for match in _TOKEN_RE.finditer(normalized):
        segment = match.group(0)
        if segment.isascii():
            tokens.append(segment)
            continue

        chars = list(segment)
        tokens.extend(chars)
        tokens.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))

    return tokens
