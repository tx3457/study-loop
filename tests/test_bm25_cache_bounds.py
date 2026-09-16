"""BM25 索引缓存的内存上界。

这个缓存持有每个文档的全部 chunk 正文和 BM25 索引，原先是一个无上限的 dict，
只在该文档更新或删除时才失效——文档越传越多，进程内存就一路涨。

限额按「缓存正文总字符数」而不是条目数：一个大文档抵得上几百个小文档，
按条目数根本封不住内存。
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import services.vectorstore as vectorstore


def _entry(chars: int) -> dict:
    return {"bm25": None, "all_docs": ["x" * chars], "all_ids": ["i"],
            "tokenizer_id": "t", "cached_chars": chars}


class TestBm25CacheBounds(unittest.TestCase):
    def setUp(self):
        vectorstore.clear_bm25_cache()
        self.addCleanup(vectorstore.clear_bm25_cache)
        self._budget = patch.object(vectorstore, "_BM25_CACHE_MAX_CHARS", 1000)
        self._budget.start()
        self.addCleanup(self._budget.stop)

    def test_cache_stops_growing_at_the_budget(self):
        for i in range(20):
            vectorstore._bm25_cache_store(f"doc-{i}", _entry(300))
        self.assertLessEqual(vectorstore._bm25_cached_chars, 1000)
        # 300 字符一条、预算 1000 → 最多留 3 条，而不是全部 20 条。
        self.assertLessEqual(len(vectorstore._bm25_cache), 4)

    def test_eviction_is_least_recently_used(self):
        for key in ("a", "b", "c"):
            vectorstore._bm25_cache_store(key, _entry(300))
        # 重新读取 a，使其成为最近使用；接下来该淘汰的是 b。
        self.assertIsNotNone(vectorstore._bm25_cache_take("a"))
        vectorstore._bm25_cache_store("d", _entry(300))

        self.assertIn("a", vectorstore._bm25_cache)
        self.assertIn("d", vectorstore._bm25_cache)
        self.assertNotIn("b", vectorstore._bm25_cache)

    def test_char_count_does_not_drift_across_replace_and_drop(self):
        """计数漂移会让限额慢慢失效，而且不会有任何报错。"""
        vectorstore._bm25_cache_store("doc", _entry(400))
        vectorstore._bm25_cache_store("doc", _entry(100))   # 同键覆盖
        self.assertEqual(vectorstore._bm25_cached_chars, 100)

        vectorstore._bm25_cache_drop("doc")
        self.assertEqual(vectorstore._bm25_cached_chars, 0)
        self.assertEqual(len(vectorstore._bm25_cache), 0)

        vectorstore._bm25_cache_drop("never-cached")       # 不存在的键不应改变计数
        self.assertEqual(vectorstore._bm25_cached_chars, 0)

    def test_a_single_oversized_document_is_still_cached_alone(self):
        """单个文档超过整个预算时，保留它一条比反复重建索引更合理；
        内存上界因此是「预算 + 一个文档」，仍然有界。"""
        vectorstore._bm25_cache_store("huge", _entry(5000))
        self.assertIn("huge", vectorstore._bm25_cache)
        self.assertEqual(len(vectorstore._bm25_cache), 1)

        vectorstore._bm25_cache_store("small", _entry(100))
        # 新条目进来后，超额的旧条目被淘汰，不会两条并存。
        self.assertEqual(len(vectorstore._bm25_cache), 1)
        self.assertIn("small", vectorstore._bm25_cache)
        self.assertEqual(vectorstore._bm25_cached_chars, 100)

    def test_clear_resets_the_counter(self):
        vectorstore._bm25_cache_store("doc", _entry(500))
        vectorstore.clear_bm25_cache()
        self.assertEqual(vectorstore._bm25_cached_chars, 0)
        self.assertEqual(len(vectorstore._bm25_cache), 0)


if __name__ == "__main__":
    unittest.main()
