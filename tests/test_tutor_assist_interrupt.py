"""
Assistant ask_user interrupt + resume 集成测试（Phase 4 里程碑二）

用真 AsyncSqliteSaver(临时 db) + mock run_tool_round，验证 assistant 的 HITL：
  1) assist 模式跑到 assistant 内部 ask_user → interrupt 暂停 → 返回值含 __interrupt__，payload 带 question
  2) Command(resume=user_reply) 续跑 → assistant 节点重放 → finalize → done + final_answer

⚠️ 运行环境：必须 Python 3.11+（langgraph interrupt 依赖 contextvar 跨 asyncio.create_task
   传播，3.10 async interrupt 会 RuntimeError）。生产 Docker 用 python:3.11-slim。

interrupt 重放语义说明（关键，决定 mock side_effect 顺序）：
  langgraph 节点从头重跑，已解决的 interrupt 按序返回缓存值。assistant 节点首跑：
  round0 → ask_user → interrupt 暂停。resume 时节点重放：round0 再次跑 run_tool_round
  （故 mock 第二次仍须给 ask_user）→ interrupt 返回缓存 user_reply → round1 → finalize。
  因此 mock side_effect = [ask_user, ask_user(replay), finalize]。

跑（用 3.11+ 解释器，需装 langgraph / langgraph-checkpoint-sqlite / aiosqlite / openai）：
  python -m pytest test/test_tutor_assist_interrupt.py -q
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import agents.assistant_agent as aa
import agents.tutor_graph as tg
from agents.tutor_graph import compile_tutor_graph
from langgraph.types import Command
from services.tool_loop import ToolCallOutcome, ToolRoundResult


def _round_ask_user(question="请告诉我文档 ID"):
    oc = ToolCallOutcome(call_id="a1", name="ask_user",
                         arguments={"question": question}, kind="control")
    return ToolRoundResult(assistant_message=object(), has_tool_calls=True, outcomes=[oc])


def _round_finalize(final_answer="根据你的文档，最终答案是 X", reason="信息已足"):
    oc = ToolCallOutcome(call_id="c1", name="finalize",
                         arguments={"final_answer": final_answer, "reason": reason}, kind="control")
    return ToolRoundResult(assistant_message=object(), has_tool_calls=True, outcomes=[oc])


@unittest.skipIf(sys.version_info < (3, 11), "LangGraph async interrupt requires Python 3.11+")
class TestAssistInterrupt(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="assist_intr_")
        self.db_path = str(Path(self.tmpdir) / "tutor.db")

    async def asyncTearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _init_state(self):
        return {
            "user_id": "u1", "document_id": None, "goal": "讲讲这份文档",
            "description": "讲讲这份文档", "mode": "assist",
            "turn": 0, "handoff_count": 0, "history": [], "messages": [], "tools_called": [],
        }

    async def test_ask_user_interrupt_then_resume_finalize(self):
        from services.checkpoint import open_sqlite_checkpointer
        config = {"configurable": {"thread_id": "t_assist"}}

        # side_effect：首跑 ask_user → 暂停；resume 重放 ask_user → finalize
        rounds = [_round_ask_user("请告诉我文档 ID"), _round_ask_user("请告诉我文档 ID"), _round_finalize()]
        with patch.object(tg, "check_injection", AsyncMock(return_value=(False, ""))), \
             patch.object(aa, "run_tool_round", AsyncMock(side_effect=rounds)):

            async with open_sqlite_checkpointer(self.db_path) as cp:
                graph = compile_tutor_graph(cp)
                # ① 跑到 assistant 内部 ask_user interrupt 暂停
                result = await graph.ainvoke(self._init_state(), config=config)
                self.assertIn("__interrupt__", result, "应跑到 ask_user interrupt 暂停")
                payload = result["__interrupt__"][0].value
                self.assertIn("question", payload)
                self.assertEqual(payload.get("kind"), "ask_user")

                # ② Command(resume=user_reply) 续跑 → 重放 → finalize
                result2 = await graph.ainvoke(Command(resume="文档 ID 是 doc1"), config=config)
                self.assertTrue(result2.get("done"))
                self.assertEqual(result2.get("final_answer"), "根据你的文档，最终答案是 X")

    async def test_crash_recovery_reopen_checkpointer(self):
        """崩溃恢复：assistant ask_user 暂停后，新 checkpointer 同 thread_id 取回中断点并续跑。"""
        from services.checkpoint import open_sqlite_checkpointer
        config = {"configurable": {"thread_id": "t_assist_crash"}}

        rounds = [_round_ask_user(), _round_ask_user(), _round_finalize()]
        with patch.object(tg, "check_injection", AsyncMock(return_value=(False, ""))), \
             patch.object(aa, "run_tool_round", AsyncMock(side_effect=rounds)):

            # 第一段进程：跑到 interrupt 暂停后"崩溃"
            async with open_sqlite_checkpointer(self.db_path) as cp:
                graph = compile_tutor_graph(cp)
                result = await graph.ainvoke(self._init_state(), config=config)
                self.assertIn("__interrupt__", result)

            # 第二段进程：新 checkpointer 同 thread_id 取回中断点并 resume
            async with open_sqlite_checkpointer(self.db_path) as cp:
                graph = compile_tutor_graph(cp)
                snapshot = await graph.aget_state(config)
                self.assertTrue(snapshot.values, "崩溃后状态应被 checkpointer 恢复")
                self.assertIn("assistant", snapshot.next,
                              f"应仍停在 assistant 中断点，实际 next={snapshot.next}")
                result2 = await graph.ainvoke(Command(resume="文档 ID 是 doc1"), config=config)
                self.assertTrue(result2.get("done"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
