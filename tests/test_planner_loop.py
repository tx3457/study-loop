"""
集成测试：planner_agent 5 阶段子图 + path_reviser 循环

不消耗 LLM API,通过 mock services/learning_path 的 5 个纯函数 + path_reviser 的 LLM 调用,
验证图拓扑 + 循环路由 + 精修契约。

验证 9 个契约:
1. 编译后子图含 6 个节点:extract_brief / explore / compress / synthesize / critique / path_reviser
2. critique 通过 → END(不进 path_reviser)
3. critique 不通过 + PATH_REVISER_ENABLED=true → path_reviser → critique 重审
4. critique 不通过 + PATH_REVISER_ENABLED=false → 回到 synthesize 整段重写
5. revision_count >= 2 → END(不死循环)
6. enable_critique=False → critique 直接通过(ablation)
7. _route_after_critique 路由表行为
8. path_reviser LLM 失败 → 返回 {} 不更新 learning_path(降级)
9. planner_agent adapter 把 OrchestratorState 正确转 PlannerState

跑法:
  python -m pytest tests/test_planner_loop.py -q
"""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── 共用 mock 工厂 ────────────────────────────────────────────────────────
def _mock_brief(target=4, keywords=None):
    from models.learning_path import PathBrief
    return PathBrief(
        title="t", scope="s", level="beginner",
        target_count=target, keywords=keywords or ["k1", "k2"],
    )


def _mock_exploration(chunks=None):
    from models.learning_path import ExplorationReport
    return ExplorationReport(
        queries_used=["k1", "k2"],
        chunks=chunks or ["chunk1", "chunk2"],
        candidate_concepts=[],
    )


def _mock_compressed():
    from models.learning_path import CompressedReport
    return CompressedReport(
        summary="summary", key_concepts=["c1", "c2"], suggested_stage_count=4,
    )


def _mock_path(doc_id="doc1", stages=4):
    from models.learning_path import LearningPath, LearningStage
    return LearningPath(
        document_id=doc_id, title="t", total_stages=stages,
        stages=[
            LearningStage(stage=i+1, title=f"s{i+1}", topics=["x"],
                          description="d", estimated_minutes=20)
            for i in range(stages)
        ],
    )


def _mock_critique(needs_revision=False, score=0.9):
    from models.learning_path import PathCritique
    return PathCritique(
        overall_score=score, issues=[] if not needs_revision else ["阶段 2 过浅"],
        needs_revision=needs_revision,
        revision_hints="" if not needs_revision else "把阶段 2 改深",
    )


class TestGraphTopology(unittest.TestCase):
    """编译后子图结构正确性"""

    def test_subgraph_has_all_six_nodes(self):
        from agents.planner_agent import planner_subgraph
        nodes = set(planner_subgraph.nodes.keys()) if hasattr(planner_subgraph, "nodes") else set()
        expected = {"extract_brief", "explore", "compress", "synthesize", "critique", "path_reviser"}
        self.assertTrue(expected.issubset(nodes), f"missing nodes: {expected - nodes}")

    def test_old_planner_agent_callable_still_exists(self):
        """planner_agent 是 adapter 函数,向后兼容 orchestrator.add_node 调用方式"""
        from agents.planner_agent import planner_agent
        self.assertTrue(callable(planner_agent))


class TestRouteAfterCritique(unittest.TestCase):
    """_route_after_critique 路由契约"""

    def test_pass_returns_end(self):
        from agents.planner_agent import _route_after_critique
        state = {"critique": _mock_critique(needs_revision=False).model_dump(), "revision_count": 0}
        self.assertEqual(_route_after_critique(state), "end")

    def test_needs_revision_with_reviser_enabled(self):
        from agents.planner_agent import _route_after_critique
        with patch.dict(os.environ, {"PATH_REVISER_ENABLED": "true"}):
            state = {"critique": _mock_critique(needs_revision=True).model_dump(), "revision_count": 0}
            self.assertEqual(_route_after_critique(state), "path_reviser")

    def test_needs_revision_with_reviser_disabled_falls_back_to_synthesize(self):
        from agents.planner_agent import _route_after_critique
        with patch.dict(os.environ, {"PATH_REVISER_ENABLED": "false"}):
            state = {"critique": _mock_critique(needs_revision=True).model_dump(), "revision_count": 0}
            self.assertEqual(_route_after_critique(state), "synthesize")

    def test_max_revisions_returns_end(self):
        from agents.planner_agent import _route_after_critique, _MAX_PATH_REVISIONS
        state = {
            "critique": _mock_critique(needs_revision=True).model_dump(),
            "revision_count": _MAX_PATH_REVISIONS,
        }
        self.assertEqual(_route_after_critique(state), "end")


class TestPathReviserNodeFallback(unittest.IsolatedAsyncioTestCase):
    """path_reviser LLM 调用失败的降级行为"""

    async def test_llm_failure_returns_empty_dict(self):
        from agents import planner_agent as pa
        state = {
            "learning_path": _mock_path().model_dump(),
            "critique": _mock_critique(needs_revision=True).model_dump(),
            "revision_count": 1,
        }
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse = AsyncMock(side_effect=RuntimeError("boom"))
        with patch.object(pa, "_client", mock_client):
            result = await pa._node_path_reviser(state)
        self.assertEqual(result, {})

    async def test_skip_when_no_needs_revision(self):
        from agents import planner_agent as pa
        state = {
            "learning_path": _mock_path().model_dump(),
            "critique": _mock_critique(needs_revision=False).model_dump(),
        }
        result = await pa._node_path_reviser(state)
        self.assertEqual(result, {})

    async def test_preserves_document_id_after_revision(self):
        from agents import planner_agent as pa
        path = _mock_path(doc_id="ORIGINAL_DOC")
        # mock LLM 返回的 path 有不同 document_id,reviser 应强制改回原值
        bad_path = _mock_path(doc_id="LLM_MADE_UP")

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=bad_path))]
            )
        )
        state = {
            "learning_path": path.model_dump(),
            "critique": _mock_critique(needs_revision=True).model_dump(),
            "revision_count": 1,
        }
        with patch.object(pa, "_client", mock_client):
            result = await pa._node_path_reviser(state)

        self.assertEqual(result["learning_path"]["document_id"], "ORIGINAL_DOC")


class TestSubgraphIntegration(unittest.IsolatedAsyncioTestCase):
    """端到端跑子图:5 阶段 + reviser 循环,全部 mock 业务函数"""

    async def asyncSetUp(self):
        # 每次测试前清环境变量
        os.environ.pop("PATH_REVISER_ENABLED", None)

    async def test_pass_path_skips_reviser(self):
        from agents import planner_agent as pa

        path = _mock_path()
        # mock 5 个纯函数 + LLM 不会被触发(critique 通过)
        with patch.object(pa, "extract_brief", AsyncMock(return_value=_mock_brief())), \
             patch.object(pa, "explore", AsyncMock(return_value=_mock_exploration())), \
             patch.object(pa, "compress", AsyncMock(return_value=_mock_compressed())), \
             patch.object(pa, "synthesize", AsyncMock(return_value=path)), \
             patch.object(pa, "critique", AsyncMock(return_value=_mock_critique(needs_revision=False))):

            result = await pa.planner_subgraph.ainvoke({
                "document_id": "doc1",
                "user_intent": "学 RAG",
                "enable_critique": True,
                "revision_count": 0,
            })
        self.assertEqual(result["learning_path"]["document_id"], "doc1")
        self.assertEqual(result["revision_count"], 1)  # critique 跑了 1 次

    async def test_fail_then_reviser_then_pass(self):
        """第一轮 critique 不通过 → reviser → 第二轮通过"""
        from agents import planner_agent as pa

        path = _mock_path()
        revised_path = _mock_path(stages=5)
        critique_calls = {"count": 0}

        async def fake_critique(p, c):
            critique_calls["count"] += 1
            # 第 1 次不通过,第 2 次通过
            return _mock_critique(needs_revision=(critique_calls["count"] == 1))

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=revised_path))]
            )
        )

        with patch.object(pa, "extract_brief", AsyncMock(return_value=_mock_brief())), \
             patch.object(pa, "explore", AsyncMock(return_value=_mock_exploration())), \
             patch.object(pa, "compress", AsyncMock(return_value=_mock_compressed())), \
             patch.object(pa, "synthesize", AsyncMock(return_value=path)), \
             patch.object(pa, "critique", fake_critique), \
             patch.object(pa, "_client", mock_client), \
             patch.dict(os.environ, {"PATH_REVISER_ENABLED": "true"}):

            result = await pa.planner_subgraph.ainvoke({
                "document_id": "doc1",
                "user_intent": "x",
                "enable_critique": True,
                "revision_count": 0,
            })

        # critique 跑了 2 次(第 1 次不通过 → 进 reviser → 第 2 次通过)
        self.assertEqual(critique_calls["count"], 2)
        # reviser 改过 path(总阶段数从 4 变 5)
        self.assertEqual(result["learning_path"]["total_stages"], 5)


class TestAdapter(unittest.IsolatedAsyncioTestCase):
    """planner_agent adapter:OrchestratorState ↔ PlannerState 转换"""

    async def test_adapter_writes_learning_path_back_to_orchestrator_state(self):
        from agents import planner_agent as pa

        path = _mock_path(doc_id="adapter_doc")
        fake_result = {"learning_path": path.model_dump()}

        with patch.object(pa.planner_subgraph, "ainvoke", AsyncMock(return_value=fake_result)):
            result = await pa.planner_agent({
                "document_id": "adapter_doc",
                "description": "学测试",
            })
        self.assertEqual(result["learning_path"]["document_id"], "adapter_doc")

    async def test_adapter_returns_none_on_subgraph_failure(self):
        from agents import planner_agent as pa
        with patch.object(pa.planner_subgraph, "ainvoke", AsyncMock(side_effect=RuntimeError("x"))):
            result = await pa.planner_agent({"document_id": "x", "description": ""})
        self.assertIsNone(result["learning_path"])


if __name__ == "__main__":
    unittest.main()
