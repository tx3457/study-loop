"""
单测：HyDE + Multi-query 改写

不消耗真实 LLM API,全 mock client.chat.completions.create。

验证 5 个契约:
1. HyDE 把 query 扩展为假设答案文本
2. HyDE LLM 失败 → fallback 返回原 query(不抛错)
3. Multi-query 拆出 N 个变体并去掉编号符号
4. Multi-query 列表首项始终是原 query
5. RRF 合并多路检索结果按 rrf_score 降序

跑法:
  python -m pytest tests/test_query_rewriter.py -q
"""
import asyncio
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


def _mock_chat_client(returned_text: str | None = None, raise_exc: Exception | None = None):
    """构造 mock AsyncOpenAI client,chat.completions.create 返回固定 message.content"""
    client = MagicMock()
    if raise_exc is not None:
        client.chat.completions.create = AsyncMock(side_effect=raise_exc)
    else:
        client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=returned_text))]
        ))
    return client


class TestHyDE(unittest.IsolatedAsyncioTestCase):
    """HyDE:假设性答案扩展"""

    async def test_returns_hypothesis_text(self):
        from services.query_rewriter import hyde_rewrite
        fake_doc = "Transformer 是一种基于自注意力机制的神经网络架构,广泛用于 NLP 任务。"
        client = _mock_chat_client(returned_text=fake_doc)
        result = await hyde_rewrite("什么是 Transformer", client=client)
        self.assertEqual(result, fake_doc)
        client.chat.completions.create.assert_awaited_once()

    async def test_fallback_to_original_on_llm_error(self):
        from services.query_rewriter import hyde_rewrite
        client = _mock_chat_client(raise_exc=RuntimeError("network down"))
        original = "什么是 Transformer"
        result = await hyde_rewrite(original, client=client)
        self.assertEqual(result, original)

    async def test_fallback_on_empty_response(self):
        from services.query_rewriter import hyde_rewrite
        client = _mock_chat_client(returned_text="   ")
        original = "x"
        result = await hyde_rewrite(original, client=client)
        self.assertEqual(result, original)

    async def test_empty_query_short_circuits(self):
        from services.query_rewriter import hyde_rewrite
        # 空 query 不应发起 LLM call
        client = _mock_chat_client(returned_text="should not be called")
        result = await hyde_rewrite("", client=client)
        self.assertEqual(result, "")
        client.chat.completions.create.assert_not_called()


class TestMultiQuery(unittest.IsolatedAsyncioTestCase):
    """Multi-query:N 个变体生成 + 前缀清理 + 原 query 首项"""

    async def test_generates_n_variants_with_original_first(self):
        from services.query_rewriter import multi_query_rewrite
        llm_output = "Transformer 模型是什么\n注意力机制原理\nself-attention 工作流程"
        client = _mock_chat_client(returned_text=llm_output)
        result = await multi_query_rewrite("什么是 Transformer", n=3, client=client)
        # 首项必须是原 query
        self.assertEqual(result[0], "什么是 Transformer")
        self.assertEqual(len(result), 4)
        self.assertIn("注意力机制原理", result)

    async def test_strips_numbered_prefixes(self):
        from services.query_rewriter import multi_query_rewrite
        llm_output = "1. 变体 A\n2) 变体 B\n- 变体 C"
        client = _mock_chat_client(returned_text=llm_output)
        result = await multi_query_rewrite("orig", n=3, client=client)
        # 编号/符号前缀应被清掉
        self.assertEqual(set(result[1:]), {"变体 A", "变体 B", "变体 C"})

    async def test_dedupes_against_original(self):
        from services.query_rewriter import multi_query_rewrite
        # LLM 偷懒原样返回 → 应被过滤
        llm_output = "orig\norig\n真正的变体"
        client = _mock_chat_client(returned_text=llm_output)
        result = await multi_query_rewrite("orig", n=3, client=client)
        # 原 query 只在首位出现一次
        self.assertEqual(result.count("orig"), 1)
        self.assertIn("真正的变体", result)

    async def test_fallback_on_llm_error(self):
        from services.query_rewriter import multi_query_rewrite
        client = _mock_chat_client(raise_exc=TimeoutError())
        result = await multi_query_rewrite("orig", n=3, client=client)
        self.assertEqual(result, ["orig"])


class TestRRFMerge(unittest.TestCase):
    """RRF 合并多路检索结果"""

    def test_higher_rank_in_more_lists_wins(self):
        from services.query_rewriter import rrf_merge_ranked_lists
        # doc_x 出现在两路第 1 位,doc_y 只在一路第 1 位 → doc_x 总分应更高
        list1 = [("x", "x_text"), ("y", "y_text"), ("z", "z_text")]
        list2 = [("x", "x_text"), ("w", "w_text"), ("v", "v_text")]
        result = rrf_merge_ranked_lists([list1, list2], top_k=3)
        self.assertEqual(result[0][0], "x")
        # x 被合并出现在 (id, text, score) 元组
        self.assertEqual(result[0][1], "x_text")
        self.assertGreater(result[0][2], result[1][2])

    def test_top_k_truncation(self):
        from services.query_rewriter import rrf_merge_ranked_lists
        list1 = [("a", "a"), ("b", "b"), ("c", "c"), ("d", "d")]
        result = rrf_merge_ranked_lists([list1], top_k=2)
        self.assertEqual(len(result), 2)
        self.assertEqual([r[0] for r in result], ["a", "b"])

    def test_empty_lists_returns_empty(self):
        from services.query_rewriter import rrf_merge_ranked_lists
        self.assertEqual(rrf_merge_ranked_lists([], top_k=5), [])
        self.assertEqual(rrf_merge_ranked_lists([[]], top_k=5), [])


if __name__ == "__main__":
    unittest.main()
