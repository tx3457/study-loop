"""
集成测试：reviewer↔reviser 循环子图

不消耗 LLM API,验证图拓扑正确 + 循环路由契约:
1. reviser 节点出现在编译后 graph 的节点列表
2. _should_revise 路由表 critic 触发时 REVISER_ENABLED=true → 'reviser'
3. _should_revise REVISER_ENABLED=false → 'quiz_agent' 整轮重出
4. _should_revise critic 通过时 → 'output_guard'
5. _should_revise revision_count 到上限 → 'output_guard'(不死循环)
6. reviser_agent 节点函数:LLM 失败时 fallback 返回原 quiz

跑法:
  python -m pytest tests/test_reviser_loop.py -q
"""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGraphTopology(unittest.TestCase):
    """编译后图结构正确性"""

    def test_reviser_node_registered(self):
        from agents.orchestrator import orchestrator
        # CompiledStateGraph 暴露 .nodes / .builder.nodes
        node_names = set(orchestrator.nodes.keys()) if hasattr(orchestrator, "nodes") else set()
        self.assertIn("reviser", node_names, f"got nodes: {node_names}")
        self.assertIn("critic_adapter", node_names)
        self.assertIn("quiz_agent", node_names)

    def test_graph_compiles_without_errors(self):
        # 仅 import + 编译路径不抛错(编译时 LangGraph 会做拓扑校验)
        from agents.orchestrator import orchestrator, _builder
        self.assertIsNotNone(orchestrator)
        # builder 里 reviser → critic_adapter 边应存在
        # 不同 langgraph 版本边的内部结构有差异,只做 contains 校验
        self.assertIn("reviser", _builder.nodes)


class TestShouldReviseRouting(unittest.TestCase):
    """_should_revise 路由表契约"""

    def _state_with_critique(self, overall, has_high=False, revision_count=0):
        suggestions = [{"target": "x", "severity": "high", "action": "fix"}] if has_high else []
        return {
            "critique_history": [{
                "overall_score": overall,
                "suggestions": suggestions,
            }],
            "revision_count": revision_count,
        }

    def test_critic_pass_returns_output_guard(self):
        from agents.orchestrator import _should_revise
        state = self._state_with_critique(overall=0.85)
        self.assertEqual(_should_revise(state), "output_guard")

    def test_critic_disabled_returns_output_guard(self):
        from agents.orchestrator import _should_revise
        with patch.dict(os.environ, {"CRITIC_ENABLED": "false"}):
            state = self._state_with_critique(overall=0.3, has_high=True)
            self.assertEqual(_should_revise(state), "output_guard")

    def test_max_revision_reached_returns_output_guard(self):
        from agents.orchestrator import _should_revise
        state = self._state_with_critique(overall=0.3, revision_count=2)
        self.assertEqual(_should_revise(state), "output_guard")

    def test_low_score_with_reviser_enabled_routes_to_reviser(self):
        from agents.orchestrator import _should_revise
        with patch.dict(os.environ, {"REVISER_ENABLED": "true", "CRITIC_ENABLED": "true"}):
            state = self._state_with_critique(overall=0.55)
            self.assertEqual(_should_revise(state), "reviser")

    def test_low_score_with_reviser_disabled_routes_to_quiz_agent(self):
        from agents.orchestrator import _should_revise
        with patch.dict(os.environ, {"REVISER_ENABLED": "false", "CRITIC_ENABLED": "true"}):
            state = self._state_with_critique(overall=0.55)
            self.assertEqual(_should_revise(state), "quiz_agent")

    def test_high_severity_triggers_revise_even_if_overall_ok(self):
        from agents.orchestrator import _should_revise
        with patch.dict(os.environ, {"REVISER_ENABLED": "true", "CRITIC_ENABLED": "true"}):
            state = self._state_with_critique(overall=0.85, has_high=True)
            self.assertEqual(_should_revise(state), "reviser")


class TestReviserNodeFallback(unittest.IsolatedAsyncioTestCase):
    """reviser_agent 节点函数的失败降级"""

    async def test_returns_original_quiz_on_llm_error(self):
        from agents import reviser_agent as ra
        original_quiz = {"questions": [{"question": "Q", "answer": "A", "type": "choice"}]}
        state = {
            "quiz": original_quiz,
            "critique_history": [{
                "overall_score": 0.5,
                "suggestions": [{"target": "relevance", "severity": "high", "action": "fix"}],
            }],
            "reflected_message": "改改改",
            "revision_count": 1,
            "type": "choice",
        }

        # mock LLM 抛错 → reviser 应返回原 quiz
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse = AsyncMock(side_effect=RuntimeError("boom"))
        with patch.object(ra, "_client", mock_client):
            result = await ra.reviser_agent(state)
        self.assertEqual(result["quiz"], original_quiz)

    async def test_skip_when_no_critique(self):
        from agents import reviser_agent as ra
        result = await ra.reviser_agent({"quiz": {"questions": []}, "critique_history": []})
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
