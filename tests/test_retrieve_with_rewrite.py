"""
retrieve_with_rewrite 单测

验证生产检索入口的改写编排逻辑,全部 mock 掉 LLM(query_rewriter)与 ChromaDB(hybrid_query_document),
只测纯编排:单 query 直连 / 多 query RRF 合并 / 去重 / 全失败兜底 / 双开关关闭零行为变化。

跑:
  python -m pytest tests/test_retrieve_with_rewrite.py -v
"""
import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import services.vectorstore as vs


def _hybrid_stub(returns_by_query: dict):
    """造一个假的 hybrid_query_document:按 query 返回预设 {documents,ids}。"""
    async def _fn(document_id, query, n_results=5):
        return returns_by_query[query]
    return _fn


class TestRetrieveWithRewrite(unittest.IsolatedAsyncioTestCase):

    async def test_both_off_single_hybrid(self):
        """HyDE/Multi-query 全关 → 直接 hybrid(原 query),零行为变化。"""
        calls = []

        async def _hybrid(document_id, query, n_results=5):
            calls.append(query)
            return {"documents": [["d1"]], "ids": [["id1"]]}

        with patch.object(vs, "multiquery_enabled", return_value=False), \
             patch.object(vs, "hyde_enabled", return_value=False), \
             patch.object(vs, "hybrid_query_document", _hybrid):
            out = await vs.retrieve_with_rewrite("doc", "原始query")

        self.assertEqual(calls, ["原始query"])          # 只检索一次,用原 query
        self.assertEqual(out["documents"][0], ["d1"])

    async def test_multiquery_rrf_merge(self):
        """Multi-query 开 → N 变体各 hybrid → RRF 合并(出现在多个列表里的 doc 排前)。"""
        async def _mq(query, n=None):
            return [query, "变体A", "变体B"]

        returns = {
            "原q":   {"documents": [["共享", "仅q"]],   "ids": [["c", "x1"]]},
            "变体A": {"documents": [["共享", "仅A"]],   "ids": [["c", "x2"]]},
            "变体B": {"documents": [["仅B", "共享"]],   "ids": [["x3", "c"]]},
        }

        with patch.object(vs, "multiquery_enabled", return_value=True), \
             patch.object(vs, "hyde_enabled", return_value=False), \
             patch.object(vs, "multi_query_rewrite", _mq), \
             patch.object(vs, "hybrid_query_document", _hybrid_stub(returns)):
            out = await vs.retrieve_with_rewrite("doc", "原q", n_results=3)

        ids = out["ids"][0]
        self.assertEqual(len(ids), 3)               # top_k=n_results=3,只回 3 条
        self.assertEqual(ids[0], "c")               # "共享" 命中全部 3 路,RRF 分最高,排第一
        self.assertIn("x3", ids)                    # x3 在自己列表 rank0(1/61)> x1/x2 的 rank1(1/62)
        self.assertEqual(len({"x1", "x2"} & set(ids)), 1)  # x1/x2 并列,top_k 截断只进其一

    async def test_dedup_when_hyde_collapses(self):
        """HyDE 失败时对每个变体都 fallback 回原文 → 出现重复 query → 必须去重为 1 次检索。"""
        calls = []

        async def _mq(query, n=None):
            return [query, "变体A", "变体B"]

        async def _hyde_fail(q):
            return "同一句"                            # 全部塌缩成同一句

        async def _hybrid(document_id, query, n_results=5):
            calls.append(query)
            return {"documents": [["d"]], "ids": [["i"]]}

        with patch.object(vs, "multiquery_enabled", return_value=True), \
             patch.object(vs, "hyde_enabled", return_value=True), \
             patch.object(vs, "multi_query_rewrite", _mq), \
             patch.object(vs, "hyde_rewrite", _hyde_fail), \
             patch.object(vs, "hybrid_query_document", _hybrid):
            out = await vs.retrieve_with_rewrite("doc", "原q")

        self.assertEqual(calls, ["同一句"])            # 去重后只检索一次,而非 3 次
        self.assertEqual(out["documents"][0], ["d"])

    async def test_all_subqueries_fail_fallback(self):
        """多 query 各路检索全抛异常 → 兜底退回单 query hybrid(原 query),绝不返回空。"""
        call_log = []

        async def _mq(query, n=None):
            return [query, "变体A"]

        async def _hybrid(document_id, query, n_results=5):
            call_log.append(query)
            # 前两次(多路)抛错,兜底那次(原 query)成功
            if len(call_log) <= 2:
                raise RuntimeError("retrieval boom")
            return {"documents": [["兜底块"]], "ids": [["fb"]]}

        with patch.object(vs, "multiquery_enabled", return_value=True), \
             patch.object(vs, "hyde_enabled", return_value=False), \
             patch.object(vs, "multi_query_rewrite", _mq), \
             patch.object(vs, "hybrid_query_document", _hybrid):
            out = await vs.retrieve_with_rewrite("doc", "原q")

        self.assertEqual(out["documents"][0], ["兜底块"])   # 没有返回空 chunks
        self.assertEqual(call_log[-1], "原q")               # 最后一次是用原 query 兜底


if __name__ == "__main__":
    unittest.main(verbosity=2)
