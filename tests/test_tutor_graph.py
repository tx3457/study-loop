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
from models.grader import AIFeedback
from models.quiz import Question
from models.session import QuizSession
from services import grader as grader_service
from services.session import sessions
from services.tutor_sessions import (
    ensure_tutor_session,
    restore_completed_tutor_session,
    tutor_session_id,
)


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


class TestTutorQuizSessionIdentity(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_sessions = dict(sessions)
        sessions.clear()
        grader_service._grade_locks.clear()

    def tearDown(self):
        sessions.clear()
        sessions.update(self.original_sessions)
        grader_service._grade_locks.clear()

    @staticmethod
    def _quiz(prompt: str, answer: str) -> dict:
        return {
            "questions": [
                {
                    "question": prompt,
                    "options": ["A", "B"],
                    "answer": answer,
                    "explanation": "explanation",
                    "source": "notes.md",
                    "type": "choice",
                }
            ]
        }

    async def test_each_tutor_round_gets_an_immutable_quiz_session(self):
        first_state = {
            "thread_id": "thread-1",
            "user_id": "user-1",
            "document_id": "notes.md",
            "turn": 1,
            "quiz": self._quiz("first question", "A"),
        }
        first_id = tg._ensure_session(first_state)
        self.assertEqual(tg._ensure_session(first_state), first_id)

        first_session = sessions[first_id]
        first_session.user_answers = ["A"]
        first_session.status = "completed"
        first_report = await grader_service.grade_session(first_id)

        second_state = {
            **first_state,
            "turn": 2,
            "session_id": first_id,
            "quiz": self._quiz("second question", "B"),
        }
        second_id = tg._ensure_session(second_state)

        self.assertNotEqual(second_id, first_id)
        self.assertEqual(sessions[second_id].questions[0].question, "second question")
        self.assertIsNone(sessions[second_id].grading_report)
        sessions[second_id].user_answers = ["B"]
        sessions[second_id].status = "completed"
        second_report = await grader_service.grade_session(second_id)

        self.assertEqual(first_report.grades[0].question, "first question")
        self.assertEqual(second_report.grades[0].question, "second question")
        self.assertEqual(second_report.score, 1.0)

    async def test_state_type_is_authoritative_when_provider_omits_type(self):
        state = {
            "thread_id": "thread-short-answer",
            "user_id": "user-1",
            "document_id": "notes.md",
            "turn": 1,
            "type": "short_answer",
            "quiz": {
                "questions": [
                    {
                        "question": "Explain RAG",
                        "answer": "Retrieve evidence before generation",
                        "explanation": "Ground the answer in retrieved evidence",
                        "source": "notes.md",
                    }
                ]
            },
        }
        session = ensure_tutor_session(
            state,
            completed_answers=["First retrieve support, then answer from it"],
        )
        self.assertEqual(session.questions[0].type, "short_answer")

        with patch.object(
            grader_service,
            "_llm_grade",
            AsyncMock(
                return_value=AIFeedback(
                    is_correct=True,
                    feedback="语义正确",
                    knowledge_gap="",
                )
            ),
        ):
            report = await grader_service.grade_session(session.session_id)
        self.assertTrue(report.grades[0].is_correct)

    def test_invalid_answers_do_not_poison_session_and_checkpoint_can_restore(self):
        state = {
            "thread_id": "thread-recovery",
            "user_id": "owner",
            "document_id": "notes.md",
            "turn": 1,
            "quiz": self._quiz("question", "A"),
        }
        active = ensure_tutor_session(state)

        with self.assertRaises(ValueError):
            ensure_tutor_session(state, completed_answers=["   "])
        self.assertEqual(active.status, "active")
        self.assertEqual(active.user_answers, [])

        completed = ensure_tutor_session(state, completed_answers=[" A "])
        self.assertEqual(completed.user_answers, ["A"])
        checkpoint_state = {
            **state,
            "session_id": completed.session_id,
            "answers": ["A"],
        }
        sessions.clear()

        restored = restore_completed_tutor_session(checkpoint_state)
        self.assertEqual(restored.session_id, completed.session_id)
        self.assertEqual(restored.status, "completed")
        self.assertEqual(restored.user_answers, ["A"])

    def test_public_quiz_view_does_not_expose_answer_or_explanation(self):
        from services.tutor_sessions import tutor_quiz_view

        state = {
            "thread_id": "thread-public-view",
            "user_id": "owner",
            "document_id": "notes.md",
            "turn": 1,
            "quiz": self._quiz("question", "A"),
        }
        public_question = tutor_quiz_view(
            ensure_tutor_session(state).questions
        )["questions"][0]
        self.assertEqual(public_question["question"], "question")
        self.assertNotIn("answer", public_question)
        self.assertNotIn("explanation", public_question)


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

    def setUp(self):
        self.original_sessions = dict(sessions)
        sessions.clear()
        sessions["s"] = QuizSession(
            session_id="s",
            document_id="doc1",
            user_id="u1",
            questions=[
                Question(
                    question=f"q{index}",
                    options=["x", "y"],
                    answer="y",
                    explanation="explanation",
                    source="notes.md",
                    type="choice",
                )
                for index in range(2)
            ],
            user_answers=["x", "y"],
            status="completed",
        )

    def tearDown(self):
        sessions.clear()
        sessions.update(self.original_sessions)

    async def test_backfills_history_and_resets_gate(self):
        import agents.grader_worker as gw
        report = _report(0.5, ["最短路"])
        with patch.object(gw.grader_agent, "ainvoke", AsyncMock(return_value={"grading_report": report})), \
             patch.object(gw.adapt_writer, "ainvoke", AsyncMock(return_value={})), \
             patch.object(gw, "get_mastery", AsyncMock(return_value=0.62)), \
             patch.object(gw, "consolidate_session_extras", AsyncMock()), \
             patch.object(gw, "update_after_session", AsyncMock()):
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

    async def test_uses_session_owner_and_replay_does_not_repeat_extras(self):
        import agents.grader_worker as gw

        report = _report(0.5, ["最短路"])
        writer = AsyncMock()

        async def mark_profile_written(_state):
            sessions["s"].profile_written = True
            return {}

        writer.side_effect = mark_profile_written
        mastery = AsyncMock(return_value=0.62)
        consolidate = AsyncMock()
        update_srs = AsyncMock()
        state = {
            "user_id": "spoofed-user",
            "document_id": "spoofed-doc",
            "session_id": "s",
            "type": "short_answer",
            "weak_points": ["foreign-point"],
            "history": [],
        }

        with patch.object(
            gw.grader_agent,
            "ainvoke",
            AsyncMock(return_value={"grading_report": report}),
        ), patch.object(gw.adapt_writer, "ainvoke", writer), patch.object(
            gw, "get_mastery", mastery
        ), patch.object(
            gw, "consolidate_session_extras", consolidate
        ), patch.object(gw, "update_after_session", update_srs):
            await gw.grader_worker(state)
            await gw.grader_worker(state)

        writer_state = writer.await_args.args[0]
        self.assertEqual(writer_state["user_id"], "u1")
        self.assertEqual(writer_state["document_id"], "doc1")
        self.assertEqual(writer_state["type"], "choice")
        mastery.assert_awaited_with("u1", "doc1")
        consolidate.assert_awaited_once()
        self.assertEqual(consolidate.await_args.args[0], "u1")
        self.assertEqual(consolidate.await_args.args[2], "doc1")
        self.assertEqual(consolidate.await_args.kwargs["question_type"], "choice")
        self.assertEqual(consolidate.await_args.kwargs["session_id"], "s")
        update_srs.assert_awaited_once_with(
            "u1",
            "doc1",
            reviewed_points=[],
            wrong_gaps=["最短路"],
            session_id="s",
        )

    async def test_restores_post_answer_checkpoint_before_grading(self):
        import agents.grader_worker as gw

        sessions.clear()
        state = {
            "thread_id": "crash-window",
            "user_id": "owner",
            "document_id": "notes.md",
            "turn": 1,
            "quiz": TestTutorQuizSessionIdentity._quiz("question", "A"),
            "answers": ["A"],
            "history": [],
        }
        state["session_id"] = tutor_session_id(state)
        active = ensure_tutor_session(state)
        self.assertEqual(active.status, "active")
        report = {
            "session_id": state["session_id"],
            "total": 1,
            "correct": 1,
            "score": 1.0,
            "grades": [
                {
                    "index": 0,
                    "question": "question",
                    "user_answer": "A",
                    "correct_answer": "A",
                    "is_correct": True,
                    "knowledge_gap": None,
                }
            ],
        }

        async def grade_after_restore(_state):
            restored = sessions[state["session_id"]]
            self.assertEqual(restored.status, "completed")
            self.assertEqual(restored.user_answers, ["A"])
            return {"grading_report": report}

        with patch.object(
            gw.grader_agent,
            "ainvoke",
            AsyncMock(side_effect=grade_after_restore),
        ), patch.object(gw.adapt_writer, "ainvoke", AsyncMock()), patch.object(
            gw, "get_mastery", AsyncMock(return_value=1.0)
        ), patch.object(
            gw, "consolidate_session_extras", AsyncMock()
        ), patch.object(gw, "update_after_session", AsyncMock()):
            result = await gw.grader_worker(state)

        self.assertEqual(result["grading_report"]["session_id"], state["session_id"])

    async def test_empty_report_no_crash(self):
        import agents.grader_worker as gw
        with patch.object(gw.grader_agent, "ainvoke", AsyncMock(return_value={})):
            out = await gw.grader_worker({"user_id": "u1", "document_id": "doc1"})
        self.assertEqual(out, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
