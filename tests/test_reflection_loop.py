"""
单测：Reflection 错误回灌

验证三件事:
1. _format_reflected_message 把 CritiqueReport 格式化为人类可读的反思文本
2. _critic_adapter 把格式化后的 reflected_message 写入返回 dict
3. quiz_agent.generate 节点把 state["reflected_message"] 传给 LLM 调用

跑法:
  cd study-loop
  python -m pytest tests/test_reflection_loop.py -q
或
  python -m unittest test.test_reflection_loop
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("CRITIC_ENABLED", "true")


class TestFormatReflectedMessage(unittest.TestCase):
    """单元测试 _format_reflected_message 的输出契约"""

    def setUp(self):
        from agents.orchestrator import _format_reflected_message
        self.fn = _format_reflected_message

    def test_empty_critique_returns_empty(self):
        self.assertEqual(self.fn({}), "")
        self.assertEqual(self.fn(None), "")

    def test_low_score_renders_dimensions(self):
        critique = {
            "overall_score": 0.55,
            "difficulty": {"score": 0.4, "reasoning": "题目过于简单"},
            "relevance": {"score": 0.5, "reasoning": "选项 B 在原文中不存在"},
            "coverage": {"score": 0.9, "reasoning": "已覆盖薄弱点"},
            "suggestions": [
                {"target": "relevance", "severity": "high", "action": "严格基于原文"},
                {"target": "difficulty", "severity": "medium", "action": "提升至 0.7"},
            ],
        }
        msg = self.fn(critique)
        # 关键 assertion:文本里有具体错误信息,而不仅是数字
        self.assertIn("0.55", msg)
        self.assertIn("题目过于简单", msg)
        self.assertIn("选项 B 在原文中不存在", msg)
        self.assertIn("严格基于原文", msg)
        self.assertIn("提升至 0.7", msg)
        # 高分维度(coverage=0.9)不应出现在低分列表
        self.assertNotIn("已覆盖薄弱点", msg)

    def test_suggestions_sorted_by_severity(self):
        critique = {
            "overall_score": 0.5,
            "suggestions": [
                {"target": "x", "severity": "low", "action": "低优先级"},
                {"target": "y", "severity": "high", "action": "高优先级"},
                {"target": "z", "severity": "medium", "action": "中优先级"},
            ],
        }
        msg = self.fn(critique)
        # high 出现位置应早于 medium 早于 low
        pos_high = msg.find("高优先级")
        pos_med = msg.find("中优先级")
        pos_low = msg.find("低优先级")
        self.assertGreater(pos_med, pos_high)
        self.assertGreater(pos_low, pos_med)


class TestCriticAdapterReturnsReflectedMessage(unittest.IsolatedAsyncioTestCase):
    """验证 _critic_adapter 把 reflected_message 写进返回 dict"""

    async def test_critic_adapter_attaches_reflected_message(self):
        from agents import orchestrator as orch

        fake_critique = {
            "overall_score": 0.5,
            "difficulty": {"score": 0.4, "reasoning": "太简单"},
            "relevance": {"score": 0.5, "reasoning": "未引用原文"},
            "coverage": {"score": 0.5, "reasoning": "未覆盖薄弱点"},
            "suggestions": [
                {"target": "relevance", "severity": "high", "action": "严格基于原文"},
            ],
        }

        # mock critic_agent.ainvoke 直接返回 critique
        mock_invoke = AsyncMock(return_value={"critique": fake_critique})
        # mock dispatch_tool 不触发真实检索
        mock_dispatch = AsyncMock(return_value='{"chunks": []}')

        with patch.object(orch.critic_agent, "ainvoke", mock_invoke), \
             patch.object(orch, "dispatch_tool", mock_dispatch):
            state = {
                "quiz": {"questions": []},
                "user_id": "u1",
                "document_id": "doc1",
                "description": "test",
                "difficulty_score": 0.7,
                "weak_points": ["x"],
                "critique_history": [],
                "revision_count": 0,
            }
            result = await orch._critic_adapter(state)

        # 核心 assertion:reflected_message 非空且包含具体错误
        self.assertIn("reflected_message", result)
        self.assertTrue(len(result["reflected_message"]) > 0)
        self.assertIn("严格基于原文", result["reflected_message"])
        self.assertIn("0.50", result["reflected_message"])
        self.assertEqual(result["revision_count"], 1)
        self.assertEqual(len(result["critique_history"]), 1)


class TestGenerateNodeReadsReflectedMessage(unittest.IsolatedAsyncioTestCase):
    """验证 quiz_agent.generate 节点把 state["reflected_message"] 传到 LLM 调用"""

    async def test_generate_passes_reflected_message_to_rag(self):
        from agents import quiz_agent as qa

        captured_kwargs = {}

        async def fake_generate(*args, **kwargs):
            captured_kwargs.update(kwargs)
            # 返回最小可用 quiz
            from models.quiz import QuizResponse, Question
            return QuizResponse(questions=[Question(
                question="Q?", options=["A", "B", "C", "D"],
                answer="A", explanation="exp", source="src", type="choice",
            )])

        reflected_text = "上一轮被拒,严格基于原文,提升难度至 0.7"
        state = {
            "chunks": ["chunk1", "chunk2"],
            "count": 3,
            "difficulty": "medium",
            "type": "choice",
            "difficulty_score": 0.7,
            "weak_points": ["x"],
            "reflected_message": reflected_text,
            "revision_count": 1,
        }

        with patch.object(qa, "generate_question_from_chunks", fake_generate):
            result = await qa.generate(state)

        self.assertIn("quiz", result)
        # 关键:reflected_message 必须经 generate 节点透传到 rag.generate_question_from_chunks
        self.assertEqual(captured_kwargs.get("reflected_message"), reflected_text)


if __name__ == "__main__":
    unittest.main()
