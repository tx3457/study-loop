"""
Tutor oneshot 模式单测

覆盖：
  1) _oneshot_next 纯规则单跳：三 action（quiz/grade/plan）各单跳到对应 worker，
     产物存在后 → finish（不进循环、不 interrupt、不 reviser）
  2) teaching_supervisor(mode="oneshot")：单跳路由 Command.goto 正确 + 产物回来 → output_guard
  3) oneshot quiz 过一次 critic（ONESHOT_QUIZ_CRITIC=true）后 finish；关闭则出题即 finish
  4) 全图闭环（mock worker）：oneshot 三 action 各跑到 finish，不进 wait_for_answers/reviser 循环

全程 mock LLM / worker，纯路由 & 单调推进逻辑验证（不依赖 interrupt，3.10 可跑）。跑：
  python -m pytest tests/test_tutor_oneshot.py -q
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import agents.tutor_graph as tg
from agents.supervisor import _oneshot_next, teaching_supervisor
from models.quiz import Question
from models.session import QuizSession
from services.session import sessions


_QUIZ = {"questions": [
    {"question": "最短路算法？", "options": ["A", "B"], "answer": "A",
     "explanation": "e", "source": "s", "type": "choice"},
]}
_REPORT = {"session_id": "s", "total": 1, "correct": 1, "score": 1.0,
           "grades": [{"index": 0, "question": "q", "user_answer": "A",
                       "correct_answer": "A", "is_correct": True, "knowledge_gap": None}]}
_PATH = {"document_id": "doc1", "title": "图论路径", "total_stages": 2, "stages": []}


async def _oneshot_decide(state: dict):
    """以 mode=oneshot 跑 teaching_supervisor，返回 Command（纯规则，不调 LLM）。"""
    st = {**state, "mode": "oneshot"}
    return await teaching_supervisor(st)


# ═══════════════════════════════════════════════════════════════════════════
# 1) _oneshot_next 纯规则单跳
# ═══════════════════════════════════════════════════════════════════════════
class TestOneshotNext(unittest.TestCase):

    def test_quiz_first_hop_to_quiz(self):
        d = _oneshot_next({"action": "quiz", "description": "图论"})
        self.assertEqual(d.next_agent, "quiz")
        self.assertFalse(d.done)

    def test_quiz_after_quiz_goes_critic_when_enabled(self):
        with patch.dict(os.environ, {"ONESHOT_QUIZ_CRITIC": "true"}):
            d = _oneshot_next({"action": "quiz", "quiz": _QUIZ})
        self.assertEqual(d.next_agent, "critic")

    def test_quiz_after_critic_finishes(self):
        with patch.dict(os.environ, {"ONESHOT_QUIZ_CRITIC": "true"}):
            d = _oneshot_next({"action": "quiz", "quiz": _QUIZ,
                               "critique_history": [{"overall_score": 0.9}]})
        self.assertEqual(d.next_agent, "finish")
        self.assertTrue(d.done)

    def test_quiz_no_critic_finishes_directly(self):
        with patch.dict(os.environ, {"ONESHOT_QUIZ_CRITIC": "false"}):
            d = _oneshot_next({"action": "quiz", "quiz": _QUIZ})
        self.assertEqual(d.next_agent, "finish")
        self.assertTrue(d.done)

    def test_grade_first_hop_to_grader(self):
        d = _oneshot_next({"action": "grade", "session_id": "s"})
        self.assertEqual(d.next_agent, "grader")
        self.assertFalse(d.done)

    def test_grade_after_report_finishes(self):
        d = _oneshot_next({"action": "grade", "grading_report": _REPORT})
        self.assertEqual(d.next_agent, "finish")
        self.assertTrue(d.done)

    def test_plan_first_hop_to_planner(self):
        d = _oneshot_next({"action": "plan", "description": "图论"})
        self.assertEqual(d.next_agent, "planner")
        self.assertEqual(d.action, "switch_to_plan")

    def test_plan_after_path_finishes(self):
        d = _oneshot_next({"action": "plan", "learning_path": _PATH})
        self.assertEqual(d.next_agent, "finish")
        self.assertTrue(d.done)

    def test_default_action_is_quiz(self):
        d = _oneshot_next({"description": "图论"})
        self.assertEqual(d.next_agent, "quiz")


# ═══════════════════════════════════════════════════════════════════════════
# 2) teaching_supervisor(mode="oneshot") Command 路由
# ═══════════════════════════════════════════════════════════════════════════
class TestOneshotSupervisorRouting(unittest.IsolatedAsyncioTestCase):

    async def test_quiz_routes_to_quiz_node(self):
        cmd = await _oneshot_decide({"action": "quiz", "description": "图论"})
        self.assertEqual(cmd.goto, "quiz")
        self.assertEqual(cmd.update["handoff_count"], 1)
        self.assertEqual(cmd.update["turn"], 1)

    async def test_grade_routes_to_grader_node(self):
        cmd = await _oneshot_decide({"action": "grade", "session_id": "s"})
        self.assertEqual(cmd.goto, "grader")

    async def test_plan_routes_to_planner_node(self):
        cmd = await _oneshot_decide({"action": "plan", "description": "图论"})
        self.assertEqual(cmd.goto, "planner")

    async def test_quiz_done_routes_output_guard(self):
        with patch.dict(os.environ, {"ONESHOT_QUIZ_CRITIC": "false"}):
            cmd = await _oneshot_decide({"action": "quiz", "quiz": _QUIZ})
        self.assertEqual(cmd.goto, "output_guard")
        self.assertTrue(cmd.update["done"])
        self.assertEqual(cmd.update["terminate_reason"], "agent_finish")

    async def test_grade_done_routes_output_guard_not_mastery(self):
        # oneshot grade 完成 → agent_finish（不被 mastery 兜底抢先，即便满分）
        cmd = await _oneshot_decide({"action": "grade", "grading_report": _REPORT,
                                     "last_report": _REPORT, "turn": 1})
        self.assertEqual(cmd.goto, "output_guard")
        self.assertEqual(cmd.update["terminate_reason"], "agent_finish")

    async def test_quiz_does_not_reset_gate(self):
        # oneshot 单跳 quiz 不重置 critique_history/quiz_served（避免 quiz↔critic 死循环）
        cmd = await _oneshot_decide({"action": "quiz", "description": "图论"})
        self.assertNotIn("critique_history", cmd.update)
        self.assertNotIn("quiz_served", cmd.update)


# ═══════════════════════════════════════════════════════════════════════════
# 3) 全图闭环：oneshot 三 action 各跑到 finish（mock worker）
# ═══════════════════════════════════════════════════════════════════════════
class TestOneshotClosedLoop(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.original_sessions = dict(sessions)
        sessions.clear()
        sessions["s"] = QuizSession(
            session_id="s",
            document_id="doc1",
            user_id="u1",
            questions=[Question.model_validate(_QUIZ["questions"][0])],
            user_answers=["A"],
            status="completed",
        )

    def tearDown(self):
        sessions.clear()
        sessions.update(self.original_sessions)

    def _base_state(self, action: str, **extra):
        return {
            "action": action, "user_id": "u1", "document_id": "doc1",
            "description": "图论", "goal": "图论", "count": 3,
            "difficulty": "medium", "type": "choice",
            "mode": "oneshot", "turn": 0, "handoff_count": 0, "history": [],
            **extra,
        }

    async def test_quiz_oneshot_runs_to_finish(self):
        with patch.object(tg, "check_injection", AsyncMock(return_value=(False, ""))), \
             patch.object(tg.quiz_agent, "ainvoke", AsyncMock(return_value={"quiz": _QUIZ})), \
             patch.object(tg.critic_agent, "ainvoke",
                          AsyncMock(return_value={"critique": {"overall_score": 0.9, "suggestions": []}})), \
             patch.object(tg, "dispatch_tool", AsyncMock(return_value='{"chunks": []}')):
            result = await tg.tutor_graph.ainvoke(self._base_state("quiz"))

        self.assertTrue(result.get("done"))
        self.assertEqual(result["quiz"]["questions"][0]["answer"], "A")
        self.assertEqual(result.get("terminate_reason"), "agent_finish")
        # 没进 wait_for_answers（quiz_served 不应被置）/ 没批改
        self.assertFalse(result.get("quiz_served"))
        self.assertIsNone(result.get("grading_report"))

    async def test_quiz_oneshot_no_critic_runs_to_finish(self):
        with patch.dict(os.environ, {"ONESHOT_QUIZ_CRITIC": "false"}), \
             patch.object(tg, "check_injection", AsyncMock(return_value=(False, ""))), \
             patch.object(tg.quiz_agent, "ainvoke", AsyncMock(return_value={"quiz": _QUIZ})), \
             patch.object(tg.critic_agent, "ainvoke", AsyncMock(side_effect=AssertionError("critic 不应被调"))):
            result = await tg.tutor_graph.ainvoke(self._base_state("quiz"))
        self.assertTrue(result.get("done"))
        self.assertEqual(result.get("terminate_reason"), "agent_finish")

    async def test_plan_oneshot_runs_to_finish(self):
        import agents.planner_agent as pa
        with patch.object(tg, "check_injection", AsyncMock(return_value=(False, ""))), \
             patch.object(pa.planner_subgraph, "ainvoke",
                          AsyncMock(return_value={"learning_path": _PATH})):
            result = await tg.tutor_graph.ainvoke(self._base_state("plan"))

        self.assertTrue(result.get("done"))
        self.assertEqual(result["learning_path"]["title"], "图论路径")
        self.assertIsNone(result.get("quiz"))
        self.assertEqual(result.get("terminate_reason"), "agent_finish")

    async def test_grade_oneshot_runs_to_finish(self):
        import agents.grader_worker as gw
        with patch.object(tg, "check_injection", AsyncMock(return_value=(False, ""))), \
             patch.object(gw.grader_agent, "ainvoke", AsyncMock(return_value={"grading_report": _REPORT})), \
             patch.object(gw.adapt_writer, "ainvoke", AsyncMock(return_value={})), \
             patch.object(gw, "get_mastery", AsyncMock(return_value=0.9)), \
             patch.object(gw, "consolidate_session_extras", AsyncMock()), \
             patch.object(gw, "update_after_session", AsyncMock()):
            result = await tg.tutor_graph.ainvoke(self._base_state("grade", session_id="s"))

        self.assertTrue(result.get("done"))
        self.assertIsNotNone(result.get("grading_report") or result.get("last_report"))
        self.assertEqual(result.get("terminate_reason"), "agent_finish")
        self.assertIsNone(result.get("quiz"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
