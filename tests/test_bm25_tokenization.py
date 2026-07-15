"""BM25 language-aware tokenization regression tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.tokenization import BM25_TOKENIZER_ID, tokenize_for_bm25


def test_tokenizer_has_stable_version_identifier():
    assert BM25_TOKENIZER_ID == "nfkc_casefold_ascii_words_cjk_unigram_bigram_v1"


def test_tokenize_english_as_words_not_characters():
    assert tokenize_for_bm25("BM25 ranks neural networks, not neural-network!") == [
        "bm25",
        "ranks",
        "neural",
        "networks",
        "not",
        "neural",
        "network",
    ]


def test_tokenize_cjk_with_unigrams_and_bigrams():
    assert tokenize_for_bm25("机器学习") == [
        "机",
        "器",
        "学",
        "习",
        "机器",
        "器学",
        "学习",
    ]


def test_tokenize_mixed_text_with_nfkc_and_casefold():
    assert tokenize_for_bm25("ＬＬＭ驱动 RAG-2026") == [
        "llm",
        "驱",
        "动",
        "驱动",
        "rag",
        "2026",
    ]


def test_tokenize_ignores_punctuation_and_whitespace():
    assert tokenize_for_bm25(" \t，。!?\n") == []
