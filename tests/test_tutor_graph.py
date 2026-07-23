"""
Tutor Graph 闭环逻辑单测（不含 interrupt）

覆盖：
  1) supervisor rule-mode 确定性路由：cold→diagnostic→quiz→critic→(reviser↔critic)→await_answers
     →[模拟 answers]→grader→supervisor→下一轮/finish 全链路 next_agent 正确
  2) critic 节点（mock critic_agent）：低分置 critic_passed=False、高分置 True，累积 critique_history
  3) grader_worker（mock grader_agent/adapt_writer/get_mastery）：回填 history(AdaptiveTurn dict)
     + last_report + 重置质量门标记
  4) 新一轮 quiz 时 supervisor 重置质量门状态（critique_history/revision_count/critic_passed/quiz_served）

全程 mock LLM / worker，纯路由 & 回填逻辑验证。跑：
  python -m pytest tests/test_tutor_graph.py -q
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import agents.tutor_graph as tg
from agents.supervisor import teaching_supervisor


def _report(score: float, gaps: list[str] | None = None) -> dict:
    gaps = gaps or []
    grades = [
        {"index": i, "question": f"q{i}", "user_answer": "x", "correct_answer": "y",
         "is_correct": i >= len(gaps), "knowledge_gap": (gaps[i] if i < len(gaps) else None)}
        for i in range(2)
    ]
    correct = sum(1 for g in grades if g["is_correct"])
    return {"session_id": "s", "total": 2, "correct": correct, "score": score, "grades": grades}


async def _rule_decide(state: dict):
    """rule 模式跑 teaching_supervisor，返回 Command（不依赖 LLM）。"""
    with patch.dict(os.environ, {"SUPERVISOR_MODE": "rule"}):
        return await teaching_supervisor(state)


# ═══════════════════════════════════════════════════════════════════════════
# 1) supervisor rule-mode 全链路确定性路由
# ═══════════════════════════════════════════════════════════════════════════
class TestClosedLoopRouting(unittest.IsolatedAsyncioTestCase):

    async def test_cold_start_to_diagnostic(self):
        cmd = await _rule_decide({"goal": "图论", "history": []})
        self.assertEqual(cmd.goto, "diagnostic")

    async def test_diagnosed_to_quiz_and_resets_gate(self):
        # 已诊断、还没出题 → quiz，且重置质量门标记
        cmd = await _rule_decide({"goal": "图论", "history": [{"agent": "diagnostic"}], "last_report": None})
        self.assertEqual(cmd.goto, "quiz")
        self.assertEqual(cmd.update["critique_history"], [])
        self.assertEqual(cmd.update["revision_count"], 0)
        self.assertFalse(cmd.update["critic_passed"])
        self.assertFalse(cmd.update["quiz_served"])

    async def test_quiz_to_critic(self):
        # 出完题（有 quiz、无 critique）→ critic 质量门
        cmd = await _rule_decide({
            "goal": "图论", "history": [{"agent": "diagnostic"}],
            "quiz": {"questions": [{"question": "q", "answer": "a"}]},
        })
        self.assertEqual(cmd.goto, "critic")

    async def test_critic_low_to_reviser(self):
        cmd = await _rule_decide({
            "goal": "图论", "history": [{"agent": "diagnostic"}],
            "quiz": {"questions": [{"question": "q", "answer": "a"}]},
            "critique_history": [{"overall_score": 0.4, "suggestions": []}],
            "revision_count": 1, "critic_passed": False,
        })
        self.assertEqual(cmd.goto, "reviser")

    async def test_critic_pass_to_wait_for_answers(self):
        cmd = await _rule_decide({
            "goal": "图论", "history": [{"agent": "diagnostic"}],
            "quiz": {"questions": [{"question": "q", "answer": "a"}]},
            "critique_history": [{"overall_score": 0.95, "suggestions": []}],
            "revision_count": 1, "critic_passed": True,
        })
        self.assertEqual(cmd.goto, "wait_for_answers")
        self.assertEqual(cmd.update["next_agent"], "await_answers")

    async def test_answers_submitted_to_grader(self):
        # quiz 已下发并恢复出 answers、还没批改 → grader
        cmd = await _rule_decide({
            "goal": "图论", "history": [{"agent": "diagnostic"}],
            "quiz": {"questions": [{"question": "q", "answer": "a"}]},
            "quiz_served": True, "answers": ["A", "B"],
        })
        self.assertEqual(cmd.goto, "grader")

    async def test_graded_high_score_advances_next_round(self):
        # 批改完（answers 已清空、quiz_served_graded=True）→ 进入下一轮出题（高分升难度）
        cmd = await _rule_decide({
            "goal": "图论", "history": [{"agent": "grader"}],
            "last_report": _report(0.9), "answers": [], "quiz_served_graded": True,
            "quiz": None, "quiz_served": False,
        })
        self.assertEqual(cmd.goto, "quiz")
        self.assertEqual(cmd.update["last_action"], "advance")

    async def test_mastery_reached_finishes(self):
        from services.adaptive_loop import MASTERY_TARGET
        cmd = await _rule_decide({
            "goal": "图论", "turn": 2, "history": [{"agent": "grader"}],
            "last_report": _report(MASTERY_TARGET + 0.05),
        })
        self.assertEqual(cmd.goto, "output_guard")
        self.assertEqual(cmd.update["terminate_reason"], "mastery_reached")


# ═══════════════════════════════════════════════════════════════════════════
# 2) critic 节点：mock critic_agent，验证 critic_passed + critique 累积
# ═══════════════════════════════════════════════════════════════════════════
class TestCriticNode(unittest.IsolatedAsyncioTestCase):

    async def _run_critic(self, critique: dict):
        fake = AsyncMock(return_value={"critique": critique})
        with patch.object(tg.critic_agent, "ainvoke", fake), \
             patch.object(tg, "dispatch_tool", AsyncMock(return_value='{"chunks": []}')):
            return await tg._critic_node({
                "quiz": {"questions": [{"question": "q", "answer": "a"}]},
                "document_id": "doc1", "description": "图论",
                "critique_history": [], "revision_count": 0,
            })

    async def test_high_score_passes(self):
        out = await self._run_critic({"overall_score": 0.9, "suggestions": []})
        self.assertTrue(out["critic_passed"])
        self.assertEqual(len(out["critique_history"]), 1)
        self.assertEqual(out["revision_count"], 1)

    async def test_low_score_fails(self):
        out = await self._run_critic({"overall_score": 0.4, "suggestions": []})
        self.assertFalse(out["critic_passed"])

    async def test_high_severity_fails(self):
        out = await self._run_critic({
            "overall_score": 0.9,
            "suggestions": [{"severity": "high", "target": "relevance", "action": "x"}],
        })
        self.assertFalse(out["critic_passed"])


# ═══════════════════════════════════════════════════════════════════════════
# 3) grader_worker：mock grader_agent/adapt_writer/get_mastery，验证回填
# ═══════════════════════════════════════════════════════════════════════════
class TestGraderWorker(unittest.IsolatedAsyncioTestCase):

    async def test_backfills_history_and_resets_gate(self):
        import agents.grader_worker as gw
        report = _report(0.5, ["最短路"])
        with patch.object(gw.grader_agent, "ainvoke", AsyncMock(return_value={"grading_report": report})), \
             patch.object(gw.adapt_writer, "ainvoke", AsyncMock(return_value={})), \
             patch.object(gw, "get_mastery", AsyncMock(return_value=0.62)):
            out = await gw.grader_worker({
                "user_id": "u1", "document_id": "doc1", "session_id": "s",
                "turn": 3, "last_action": "remediate", "description": "图论",
                "difficulty_score": 0.4, "history": [{"agent": "diagnostic"}],
            })

        # last_report + grading_report 回填
        self.assertEqual(out["last_report"]["score"], 0.5)
        # history 追加一条 AdaptiveTurn dict
        self.assertEqual(len(out["history"]), 2)
        turn_rec = out["history"][-1]
        self.assertEqual(turn_rec["agent"], "grader")
        self.assertEqual(turn_rec["score"], 0.5)
        self.assertEqual(turn_rec["mastery_after"], 0.62)
        self.assertEqual(turn_rec["knowledge_gaps"], ["最短路"])
        # 质量门标记重置
        self.assertTrue(out["quiz_served_graded"])
        self.assertFalse(out["critic_passed"])
        self.assertFalse(out["quiz_served"])
        self.assertEqual(out["critique_history"], [])
        self.assertEqual(out["answers"], [])

    async def test_empty_report_no_crash(self):
        import agents.grader_worker as gw
        with patch.object(gw.grader_agent, "ainvoke", AsyncMock(return_value={})):
            out = await gw.grader_worker({"user_id": "u1", "document_id": "doc1"})
        self.assertEqual(out, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
