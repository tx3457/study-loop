"""
Tutor Graph interrupt + checkpointer 集成测试（Phase 2 里程碑 B）

用真 AsyncSqliteSaver(临时 db) + mock worker，验证 HITL durable interrupt：
  1) ainvoke 跑到 wait_for_answers interrupt 暂停 → 返回值含 __interrupt__，payload 带 quiz
  2) Command(resume=[...]) 续跑 → 从 grader 继续 → 回填 last_report
  3) 崩溃恢复：interrupt 暂停后，用同 path 新建 checkpointer + 重新 compile + 同 thread_id
     resume，验证状态从中断点恢复（不丢 state）

⚠️ 运行环境：必须 Python 3.11+（langgraph interrupt 依赖 contextvar 跨 asyncio.create_task
   传播，langgraph._internal._runnable.ASYNCIO_ACCEPTS_CONTEXT = sys.version_info >= (3,11)）。
   生产 Docker 用 python:3.11-slim，故按生产运行时验证。本机 3.10 dev env 无法跑此用例。

跑（用 3.11+ 解释器，需装 langgraph==1.1.3 / langgraph-checkpoint-sqlite / aiosqlite）：
  python -m pytest test/test_tutor_interrupt.py -q
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import agents.tutor_graph as tg
import agents.diagnostic_worker as dw
from agents.tutor_graph import compile_tutor_graph
from langgraph.types import Command


_QUIZ = {"questions": [
    {"question": "最短路算法？", "options": ["A", "B"], "answer": "A",
     "explanation": "e", "source": "s", "type": "choice"},
]}


def _report(score: float) -> dict:
    return {"session_id": "s", "total": 1, "correct": int(score), "score": score,
            "grades": [{"index": 0, "question": "q", "user_answer": "A", "correct_answer": "A",
                        "is_correct": score >= 1.0, "knowledge_gap": None if score >= 1.0 else "最短路"}]}


def _patches():
    """mock 全部 worker：diagnostic / quiz / critic(高分过审) / grader_worker 内部。"""
    return [
        # 入口 guard 放行（不调真 injection LLM）
        patch.object(tg, "check_injection", AsyncMock(return_value=(False, ""))),
        # diagnostic：写画像难度
        patch.object(dw.adapt_reader, "ainvoke",
                     AsyncMock(return_value={"difficulty_score": 0.5, "weak_points": ["最短路"]})),
        patch.object(dw, "build_returning_context", AsyncMock(return_value={"is_returning": False})),
        patch.object(dw, "build_profile_card", AsyncMock(return_value="")),
        patch.object(dw, "append_decision", AsyncMock(return_value=None)),
        # quiz：出题
        patch.object(tg.quiz_agent, "ainvoke", AsyncMock(return_value={"quiz": _QUIZ})),
        # critic：高分过审（_critic_node 内部调 critic_agent + dispatch_tool）
        patch.object(tg.critic_agent, "ainvoke",
                     AsyncMock(return_value={"critique": {"overall_score": 0.9, "suggestions": []}})),
        patch.object(tg, "dispatch_tool", AsyncMock(return_value='{"chunks": []}')),
    ]


@unittest.skipIf(sys.version_info < (3, 11), "LangGraph async interrupt requires Python 3.11+")
class TestTutorInterrupt(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tutor_intr_")
        self.db_path = str(Path(self.tmpdir) / "tutor.db")
        self._ps = _patches()
        for p in self._ps:
            p.start()

    async def asyncTearDown(self):
        for p in self._ps:
            p.stop()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _init_state(self):
        return {
            "user_id": "u1", "document_id": "doc1", "goal": "图论", "description": "图论",
            "mode": "guided", "turn": 0, "handoff_count": 0, "history": [],
        }

    async def _open(self):
        from services.checkpoint import open_sqlite_checkpointer
        return open_sqlite_checkpointer(self.db_path)

    async def test_runs_to_interrupt_then_resumes(self):
        from services.checkpoint import open_sqlite_checkpointer
        config = {"configurable": {"thread_id": "t_resume"}}

        # mock grader_worker 的内部依赖（批改 + 画像写回 + mastery）
        import agents.grader_worker as gw
        with patch.dict("os.environ", {"SUPERVISOR_MODE": "rule"}), \
             patch.object(gw.grader_agent, "ainvoke", AsyncMock(return_value={"grading_report": _report(1.0)})), \
             patch.object(gw.adapt_writer, "ainvoke", AsyncMock(return_value={})), \
             patch.object(gw, "get_mastery", AsyncMock(return_value=0.9)):

            async with open_sqlite_checkpointer(self.db_path) as cp:
                graph = compile_tutor_graph(cp)
                # ① 跑到 wait_for_answers interrupt 暂停
                result = await graph.ainvoke(self._init_state(), config=config)
                self.assertIn("__interrupt__", result, "应跑到 wait_for_answers interrupt 暂停")
                payload = result["__interrupt__"][0].value
                self.assertIn("quiz", payload)
                self.assertEqual(payload["quiz"]["questions"][0]["answer"], "A")

                # ② Command(resume=...) 续跑 → grader → supervisor → 下一轮 interrupt 或 finish
                result2 = await graph.ainvoke(Command(resume=["A"]), config=config)
                # grader 跑过 → last_report 回填
                self.assertIsNotNone(result2.get("last_report"))
                self.assertEqual(result2["last_report"]["score"], 1.0)

    async def test_crash_recovery_reopen_checkpointer(self):
        """崩溃恢复：interrupt 暂停后，新进程(新 checkpointer + 重新 compile)同 thread_id 续跑。"""
        from services.checkpoint import open_sqlite_checkpointer
        config = {"configurable": {"thread_id": "t_crash"}}
        import agents.grader_worker as gw

        with patch.dict("os.environ", {"SUPERVISOR_MODE": "rule"}), \
             patch.object(gw.grader_agent, "ainvoke", AsyncMock(return_value={"grading_report": _report(1.0)})), \
             patch.object(gw.adapt_writer, "ainvoke", AsyncMock(return_value={})), \
             patch.object(gw, "get_mastery", AsyncMock(return_value=0.9)):

            # 第一段进程：跑到 interrupt 暂停后"崩溃"（退出 async with）
            async with open_sqlite_checkpointer(self.db_path) as cp:
                graph = compile_tutor_graph(cp)
                result = await graph.ainvoke(self._init_state(), config=config)
                self.assertIn("__interrupt__", result)

            # 第二段进程：全新 checkpointer + 重新 compile，同 thread_id 取回中断点
            async with open_sqlite_checkpointer(self.db_path) as cp:
                graph = compile_tutor_graph(cp)
                snapshot = await graph.aget_state(config)
                # 关键：状态被持久化恢复，且仍停在 wait_for_answers 中断点
                self.assertTrue(snapshot.values, "崩溃后状态应被 checkpointer 恢复")
                self.assertIn("wait_for_answers", snapshot.next,
                              f"应仍停在 wait_for_answers 中断点，实际 next={snapshot.next}")

                # 同 thread_id resume → 从中断点续跑成功
                result2 = await graph.ainvoke(Command(resume=["A"]), config=config)
                self.assertIsNotNone(result2.get("last_report"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
