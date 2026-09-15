"""
tutor_node 讲解节点单测（supervisor MAS 的 teach 轮）

覆盖：
  1) 产出：写 lesson / allow_teach=False / 追加一条 teach 轨迹
  2) 入参映射：description→topic（空则回退 goal）、user_id→owner_id、weak_points 透传
  3) last_report：dict → GradingReport 转换后传给 generate_lesson
  4) last_report 脏数据 → 降级为 None 继续讲，不抛错
  5) grader_worker 批改完一轮后复位 allow_teach=True（否则整个会话只能讲一次）

全程 mock generate_lesson（不调 LLM / 不检索），3.10 可跑。跑：
  python -m pytest tests/test_tutor_lesson.py -q
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import agents.tutor_graph as tg
from models.grader import GradingReport

_REPORT = {
    "session_id": "s", "total": 2, "correct": 1, "score": 0.5,
    "grades": [
        {"index": 0, "question": "最短路？", "user_answer": "B", "correct_answer": "A",
         "is_correct": False, "knowledge_gap": "Dijkstra"},
        {"index": 1, "question": "复杂度？", "user_answer": "A", "correct_answer": "A",
         "is_correct": True, "knowledge_gap": None},
    ],
}


def _state(**over) -> dict:
    base = {
        "user_id": "u1", "document_id": "doc1", "goal": "图论",
        "description": "最短路", "weak_points": ["Dijkstra", "松弛"],
        "turn": 2, "difficulty_score": 0.4, "history": [],
    }
    base.update(over)
    return base


class TestTutorNodeOutput(unittest.IsolatedAsyncioTestCase):
    """讲解节点的产出契约"""

    async def test_writes_lesson_and_blocks_consecutive_teach(self):
        with patch.object(tg, "generate_lesson", AsyncMock(return_value="讲解正文")):
            out = await tg.tutor_node(_state())
        self.assertEqual(out["lesson"], "讲解正文")
        self.assertFalse(out["allow_teach"], "讲完必须置 allow_teach=False，避免连续只讲不练")

    async def test_appends_teach_turn_to_history(self):
        with patch.object(tg, "generate_lesson", AsyncMock(return_value="讲解正文")):
            out = await tg.tutor_node(_state(history=[{"turn": 1, "agent": "quiz"}]))
        self.assertEqual(len(out["history"]), 2, "应保留旧轨迹并追加一条")
        entry = out["history"][-1]
        self.assertEqual(entry["agent"], "tutor")
        self.assertEqual(entry["action"], "teach")
        self.assertEqual(entry["topic"], "最短路")
        # 讲解轮没有得分：score/mastery_after 必须留空，否则会被当成一次"考过了"
        self.assertIsNone(entry["score"])
        self.assertIsNone(entry["mastery_after"])


class TestTutorNodeArgumentMapping(unittest.IsolatedAsyncioTestCase):
    """TutorState → generate_lesson 的入参映射"""

    async def _call(self, state: dict):
        mock = AsyncMock(return_value="L")
        with patch.object(tg, "generate_lesson", mock):
            await tg.tutor_node(state)
        return mock.await_args.kwargs

    async def test_maps_state_fields(self):
        kwargs = await self._call(_state())
        self.assertEqual(kwargs["document_id"], "doc1")
        self.assertEqual(kwargs["topic"], "最短路")
        self.assertEqual(kwargs["owner_id"], "u1", "owner_id 必须取 user_id（讲解检索要按 owner 隔离）")
        self.assertEqual(kwargs["weak_points"], ["Dijkstra", "松弛"])

    async def test_topic_falls_back_to_goal(self):
        kwargs = await self._call(_state(description=""))
        self.assertEqual(kwargs["topic"], "图论")

    async def test_last_report_dict_converted_to_model(self):
        kwargs = await self._call(_state(last_report=_REPORT))
        self.assertIsInstance(
            kwargs["last_report"], GradingReport,
            "generate_lesson 要 GradingReport，传 dict 会在它内部读 .grades 时炸",
        )
        self.assertEqual(kwargs["last_report"].score, 0.5)

    async def test_no_last_report_passes_none(self):
        kwargs = await self._call(_state())
        self.assertIsNone(kwargs["last_report"])

    async def test_malformed_last_report_degrades_to_none(self):
        # 脏 checkpoint / 半写入的 report 不该让整轮讲解挂掉
        kwargs = await self._call(_state(last_report={"score": "not-a-number"}))
        self.assertIsNone(kwargs["last_report"])


class TestAllowTeachReset(unittest.IsolatedAsyncioTestCase):
    """grader_worker 批改完一轮后复位讲解禁令"""

    async def test_grader_worker_reenables_teach(self):
        import agents.grader_worker as gw
        from models.quiz import Question
        from models.session import QuizSession
        from services.session import sessions

        # grader_worker 会校验批改报告对应的 session 真实存在且属主一致
        # （tutor replay safety），所以这里登记一个再跑。
        sessions["s"] = QuizSession(
            session_id="s", document_id="doc1", user_id="u1",
            questions=[Question(question="q", options=["A"], answer="A",
                                explanation="e", source="s", type="choice")],
            user_answers=["A"], status="completed",
        )
        self.addCleanup(sessions.pop, "s", None)

        with patch.object(gw.grader_agent, "ainvoke", AsyncMock(return_value={"grading_report": _REPORT})), \
             patch.object(gw.adapt_writer, "ainvoke", AsyncMock(return_value={})), \
             patch.object(gw, "get_mastery", AsyncMock(return_value=0.6)), \
             patch.object(gw, "consolidate_session_extras", AsyncMock(return_value=None)), \
             patch.object(gw, "update_after_session", AsyncMock(return_value=None)):
            out = await gw.grader_worker(_state(allow_teach=False, session_id="s", quiz=None))
        self.assertTrue(
            out["allow_teach"],
            "一轮讲→练→批走完后必须解禁；否则 allow_teach 置 False 后无人复位，整个会话只能讲一次",
        )


if __name__ == "__main__":
    unittest.main()
