"""
Assistant Worker 单测（非 interrupt 部分）

覆盖：
  1) assistant_agent ReAct：mock run_tool_round 返回 finalize → 写 final_answer + assistant_done
  2) assistant_agent 业务工具一轮 → 再 finalize：tools_called 记录、messages 持久化
  3) 无 tool_calls（隐式 finalize）→ 用 content 作 final_answer
  4) MAX_ASSIST_ROUNDS 截断 → assistant_done + 兜底 final_answer
  5) _assist_next 路由：未收尾 → assistant；assistant_done → finish
  6) teaching_supervisor(mode="assist") Command 路由
  7) 全图闭环：assist finalize 路径（mock run_tool_round）跑到 finish

ask_user→interrupt→resume 续跑测试见 test_tutor_assist_interrupt.py（需 Python 3.11+）。

运行：python -m pytest tests/test_assistant_agent.py -q
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import agents.assistant_agent as aa
import agents.tutor_graph as tg
from agents.assistant_agent import assistant_agent
from agents.supervisor import _assist_next, teaching_supervisor
from services.tool_loop import ToolCallOutcome, ToolRoundResult
from services.tool_registry import SideEffectAmbiguousError


def _round_finalize(final_answer="这是答案", reason="已能回答"):
    """造一轮 run_tool_round 结果：LLM 调 finalize。"""
    oc = ToolCallOutcome(call_id="c1", name="finalize",
                         arguments={"final_answer": final_answer, "reason": reason}, kind="control")
    return ToolRoundResult(assistant_message=object(), has_tool_calls=True, outcomes=[oc])


def _round_tool(name="search_document", result='{"chunks": ["x"]}'):
    """造一轮：LLM 调业务工具（已 dispatch）。"""
    oc = ToolCallOutcome(call_id="t1", name=name, arguments={"query": "q"},
                         kind="dispatched", result=result)
    return ToolRoundResult(assistant_message=object(), has_tool_calls=True, outcomes=[oc])


def _round_text(content="纯文字回答"):
    """造一轮：LLM 无 tool_calls，直接给文字（隐式 finalize）。"""
    return ToolRoundResult(assistant_message=object(), has_tool_calls=False, content=content)


def _round_ask_user(question="请给文档 ID"):
    oc = ToolCallOutcome(call_id="a1", name="ask_user",
                         arguments={"question": question}, kind="control")
    return ToolRoundResult(assistant_message=object(), has_tool_calls=True, outcomes=[oc])


# ═══════════════════════════════════════════════════════════════════════════
# 1) assistant_agent ReAct（mock run_tool_round）
# ═══════════════════════════════════════════════════════════════════════════
class TestAssistantReAct(unittest.IsolatedAsyncioTestCase):

    async def test_finalize_first_round(self):
        with patch.object(aa, "run_tool_round", AsyncMock(return_value=_round_finalize("RAG 是检索增强生成"))):
            out = await assistant_agent({"goal": "什么是 RAG", "user_id": "u1"})
        self.assertEqual(out["final_answer"], "RAG 是检索增强生成")
        self.assertTrue(out["assistant_done"])
        self.assertIn("messages", out)

    async def test_tool_then_finalize(self):
        rounds = [_round_tool("search_document"), _round_finalize("基于文档的答案")]
        with patch.object(aa, "run_tool_round", AsyncMock(side_effect=rounds)):
            out = await assistant_agent({"goal": "讲讲最短路", "user_id": "u1", "document_id": "doc1"})
        self.assertEqual(out["final_answer"], "基于文档的答案")
        self.assertTrue(out["assistant_done"])
        self.assertIn("search_document", out["tools_called"])

    async def test_rejected_control_outcome_does_not_finalize(self):
        rejected = ToolRoundResult(
            assistant_message=object(),
            has_tool_calls=True,
            outcomes=[ToolCallOutcome(
                call_id="c1",
                name="finalize",
                arguments={"final_answer": "不应采用"},
                kind="blocked",
                blocked_reason="mixed_control_batch_rejected",
            )],
        )
        run_round = AsyncMock(side_effect=[rejected, _round_finalize("重新决策后的答案")])
        with patch.object(aa, "run_tool_round", run_round):
            out = await assistant_agent({"goal": "什么是 RAG", "user_id": "u1"})

        self.assertEqual(out["final_answer"], "重新决策后的答案")
        self.assertEqual(run_round.await_count, 2)

    async def test_interrupt_capable_assistant_exposes_only_replay_safe_tools(self):
        captured = {}

        async def capture_round(*args, **kwargs):
            captured.update(kwargs)
            return _round_finalize("安全工具集")

        with patch.object(aa, "run_tool_round", side_effect=capture_round):
            out = await assistant_agent({"goal": "复习 RAG", "user_id": "u1"})

        tool_names = {
            item["function"]["name"]
            for item in captured["tools"]
        }
        self.assertEqual(out["final_answer"], "安全工具集")
        self.assertIn("search_document", tool_names)
        self.assertNotIn("update_learning_profile", tool_names)
        self.assertNotIn("get_user_profile", tool_names)
        self.assertNotIn("update_learning_profile", aa._ASSIST_SYSTEM)
        self.assertEqual(
            captured["business_tool_allowlist"],
            aa.replay_safe_tool_names(),
        )

    async def test_implicit_finalize_no_tool_calls(self):
        with patch.object(aa, "run_tool_round", AsyncMock(return_value=_round_text("直接回答"))):
            out = await assistant_agent({"goal": "什么是向量检索", "user_id": "u1"})
        self.assertEqual(out["final_answer"], "直接回答")
        self.assertTrue(out["assistant_done"])

    async def test_max_rounds_truncates(self):
        # 持续返回业务工具轮（永不 finalize）→ 达上限 → 兜底收尾
        with patch.object(aa, "run_tool_round", AsyncMock(return_value=_round_tool())):
            out = await assistant_agent({"goal": "无限循环测试", "user_id": "u1", "document_id": "doc1"})
        self.assertTrue(out["assistant_done"])
        self.assertTrue(out["final_answer"])

    async def test_llm_failure_returns_done(self):
        with patch.object(aa, "run_tool_round", AsyncMock(side_effect=RuntimeError("LLM down"))):
            out = await assistant_agent({"goal": "x", "user_id": "u1"})
        self.assertTrue(out["assistant_done"])
        self.assertIn("失败", out["final_answer"])

    async def test_ambiguous_side_effect_is_never_downgraded_to_done(self):
        with patch.object(
            aa,
            "run_tool_round",
            AsyncMock(side_effect=SideEffectAmbiguousError("unsafe_tool")),
        ):
            with self.assertRaises(SideEffectAmbiguousError):
                await assistant_agent({"goal": "x", "user_id": "u1"})

    async def test_truncation_does_not_swallow_ambiguous_side_effect(self):
        rounds = [
            _round_tool()
            for _ in range(aa.MAX_ASSIST_ROUNDS)
        ] + [SideEffectAmbiguousError("unsafe_tool")]
        with patch.object(aa, "run_tool_round", AsyncMock(side_effect=rounds)):
            with self.assertRaises(SideEffectAmbiguousError):
                await assistant_agent({"goal": "x", "user_id": "u1"})


# ═══════════════════════════════════════════════════════════════════════════
# 2) _assist_next 路由 + supervisor assist 模式
# ═══════════════════════════════════════════════════════════════════════════
class TestAssistRouting(unittest.IsolatedAsyncioTestCase):

    def test_not_done_routes_assistant(self):
        d = _assist_next({"goal": "x"})
        self.assertEqual(d.next_agent, "assistant")

    def test_done_routes_finish(self):
        d = _assist_next({"goal": "x", "assistant_done": True})
        self.assertEqual(d.next_agent, "finish")
        self.assertTrue(d.done)

    async def test_supervisor_assist_routes_assistant(self):
        cmd = await teaching_supervisor({"mode": "assist", "goal": "什么是 RAG", "history": []})
        self.assertEqual(cmd.goto, "assistant")

    async def test_supervisor_assist_done_finishes(self):
        cmd = await teaching_supervisor({"mode": "assist", "goal": "x", "assistant_done": True})
        self.assertEqual(cmd.goto, "output_guard")
        self.assertTrue(cmd.update["done"])


# ═══════════════════════════════════════════════════════════════════════════
# 3) 全图闭环：assist finalize 路径跑到 finish（mock run_tool_round）
# ═══════════════════════════════════════════════════════════════════════════
class TestAssistClosedLoop(unittest.IsolatedAsyncioTestCase):

    async def test_assist_finalize_runs_to_finish(self):
        with patch.object(tg, "check_injection", AsyncMock(return_value=(False, ""))), \
             patch.object(aa, "run_tool_round", AsyncMock(return_value=_round_finalize("最终答案"))):
            result = await tg.tutor_graph.ainvoke({
                "user_id": "u1", "document_id": None, "goal": "什么是 RAG",
                "description": "什么是 RAG", "mode": "assist",
                "turn": 0, "handoff_count": 0, "history": [], "messages": [], "tools_called": [],
            })
        self.assertTrue(result.get("done"))
        self.assertEqual(result.get("final_answer"), "最终答案")
        self.assertEqual(result.get("terminate_reason"), "agent_finish")


if __name__ == "__main__":
    unittest.main(verbosity=2)
