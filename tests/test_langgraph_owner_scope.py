"""
回归测试：LangGraph 工作流里的 critic 检索证据时必须带上当前用户

2026-09 文档按属主隔离后，search_document 把 user_id 列为必填参数，并由注册表
对照可信上下文核对（owner_argument）。orchestrator / tutor_graph 的 critic 预取和
critic_agent 的二次检索当时没跟着改：调用在注册表校验阶段就被判为参数无效，
critic 一直拿到空证据，日志里也看不出来。

这里不 mock dispatch_tool，让调用真正经过注册表的必填字段和属主校验，
只替换最底层的检索函数。

跑法:
  python -m pytest tests/test_langgraph_owner_scope.py -q
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# retrieve_with_rewrite 的返回结构（与 Chroma query 结果同形）
_RETRIEVAL = {"documents": [["证据片段 A", "证据片段 B"]], "ids": [["c1", "c2"]]}


def _patch_retrieval():
    return patch("services.tools.retrieve_with_rewrite", AsyncMock(return_value=_RETRIEVAL))


class TestCriticEvidenceSearch(unittest.IsolatedAsyncioTestCase):
    """三处 critic 检索都要通过注册表的属主校验，并把证据交给 critic"""

    async def test_orchestrator_prefetch_passes_registry_owner_check(self):
        from agents import orchestrator as orch

        critic = AsyncMock(return_value={"critique": {}})
        with _patch_retrieval() as retrieve, patch.object(orch.critic_agent, "ainvoke", critic):
            await orch._critic_adapter({
                "quiz": {"questions": []},
                "user_id": "alice",
                "document_id": "doc1",
                "description": "二分查找",
            })
        retrieve.assert_awaited_once_with("doc1", "二分查找", owner_id="alice")
        self.assertEqual(critic.await_args.args[0]["chunks"], ["证据片段 A", "证据片段 B"])

    async def test_tutor_prefetch_passes_registry_owner_check(self):
        from agents import tutor_graph as tg

        critic = AsyncMock(return_value={"critique": {}})
        with _patch_retrieval() as retrieve, patch.object(tg.critic_agent, "ainvoke", critic):
            await tg._critic_node({
                "quiz": {"questions": []},
                "user_id": "alice",
                "document_id": "doc1",
                "description": "二分查找",
            })
        retrieve.assert_awaited_once_with("doc1", "二分查找", owner_id="alice")
        self.assertEqual(critic.await_args.args[0]["chunks"], ["证据片段 A", "证据片段 B"])

    async def test_reinforce_search_passes_registry_owner_check(self):
        from agents import critic_agent as ca

        with _patch_retrieval() as retrieve:
            result = await ca._reinforce_evidence({
                "user_id": "alice",
                "document_id": "doc1",
                "weak_points": ["二分查找"],
            })
        retrieve.assert_awaited_once_with("doc1", "二分查找", owner_id="alice")
        self.assertTrue(result["triggered_search"])
        self.assertEqual(result["reinforce_chunks"], ["证据片段 A", "证据片段 B"])

    async def test_no_user_means_no_search(self):
        """拿不到可信 user_id 时不检索（失败关闭），critic 按无证据评估"""
        from agents import critic_agent as ca
        from agents import orchestrator as orch

        critic = AsyncMock(return_value={"critique": {}})
        with _patch_retrieval() as retrieve, patch.object(orch.critic_agent, "ainvoke", critic):
            await orch._critic_adapter({
                "quiz": {"questions": []},
                "document_id": "doc1",
                "description": "二分查找",
            })
            reinforce = await ca._reinforce_evidence({
                "document_id": "doc1",
                "weak_points": ["二分查找"],
            })
        retrieve.assert_not_awaited()
        self.assertEqual(critic.await_args.args[0]["chunks"], [])
        self.assertFalse(reinforce["triggered_search"])


if __name__ == "__main__":
    unittest.main()
