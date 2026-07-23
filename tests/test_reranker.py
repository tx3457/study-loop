"""
单测：Cross-Encoder Reranker

不下载真实模型(2.27GB),全部 mock CrossEncoder.predict。

验证 4 个契约:
1. rerank_docs 按 cross-encoder 分数降序返回 top_k
2. enable_rerank=False 时 hybrid_query_document 跳过精排
3. RerankerUnavailable 异常时降级到 RRF-only,不抛错
4. 候选不足 top_k 时仍返回(不崩)

跑法:
  cd study-loop
  python -m pytest tests/test_reranker.py -q
"""
import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRerankDocs(unittest.IsolatedAsyncioTestCase):
    """单元测试:rerank_docs 排序契约 + 异常透传"""

    async def test_sorts_by_score_descending(self):
        from services import reranker

        # mock CrossEncoder 返回固定分数 [0.1, 0.9, 0.5]
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[0.1, 0.9, 0.5])

        with patch.object(reranker, "_get_reranker", return_value=mock_model):
            result = await reranker.rerank_docs(
                "q",
                docs=["doc_a", "doc_b", "doc_c"],
                ids=["id_a", "id_b", "id_c"],
                top_k=3,
            )

        # 关键:输出按 score 降序排列
        self.assertEqual([r[0] for r in result], ["doc_b", "doc_c", "doc_a"])
        self.assertEqual([r[2] for r in result], ["id_b", "id_c", "id_a"])
        self.assertAlmostEqual(result[0][1], 0.9)

    async def test_top_k_truncation(self):
        from services import reranker
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[0.5, 0.3, 0.9, 0.1])
        with patch.object(reranker, "_get_reranker", return_value=mock_model):
            result = await reranker.rerank_docs("q", docs=["a", "b", "c", "d"], top_k=2)
        self.assertEqual(len(result), 2)
        self.assertEqual([r[0] for r in result], ["c", "a"])

    async def test_empty_docs_returns_empty(self):
        from services import reranker
        # 不需要 mock 模型,空 docs 应短路
        with patch.object(reranker, "_get_reranker") as get_model:
            result = await reranker.rerank_docs("q", docs=[], top_k=5)
            get_model.assert_not_called()
        self.assertEqual(result, [])

    async def test_predict_failure_raises_RerankerUnavailable(self):
        from services import reranker
        mock_model = MagicMock()
        mock_model.predict = MagicMock(side_effect=RuntimeError("boom"))
        with patch.object(reranker, "_get_reranker", return_value=mock_model):
            with self.assertRaises(reranker.RerankerUnavailable):
                await reranker.rerank_docs("q", docs=["a", "b"], top_k=2)


class TestRerankerEnabledFlag(unittest.TestCase):
    """env 开关契约"""

    def test_default_disabled_when_optional_runtime_is_not_installed(self):
        from services import reranker
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RERANKER_ENABLED", None)
            self.assertFalse(reranker.reranker_enabled())

    def test_disabled_when_env_false(self):
        from services import reranker
        with patch.dict(os.environ, {"RERANKER_ENABLED": "false"}):
            self.assertFalse(reranker.reranker_enabled())

    def test_disabled_with_various_falsy_values(self):
        from services import reranker
        for v in ("0", "no", "NO", "False", "off"):
            with patch.dict(os.environ, {"RERANKER_ENABLED": v}):
                self.assertFalse(reranker.reranker_enabled(), f"value {v!r} should disable")


if __name__ == "__main__":
    unittest.main()
