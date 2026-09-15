"""
讲解出口单测：tutor_node 产出的 lesson 必须能被前端拿到，且不跨轮回显

背景：agents/state.py 声明了 lesson 字段，但在补完 tutor 节点之前，
全仓库既没有人写它、也没有人读它——只补节点不接出口，讲解文本写进 state 后就消失了。

覆盖：
  1) TutorTurnResponse 带 lesson 字段
  2) start 跑到 interrupt（讲解 + 题目同一轮下发）→ 响应带 lesson
  3) start 直接收尾（只讲不出题）→ 响应带 lesson
  4) submit 跑到下一轮 interrupt → 响应带 lesson
  5) submit 跑到 finish → 响应带 lesson
  6) grader_worker 批改后清空 lesson → 下一轮不会把上一轮讲解再回显一遍

mock 掉 checkpointer / 图，不依赖 interrupt 机制，3.10 可跑。跑：
  python -m pytest tests/test_tutor_lesson_response.py -q
"""
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import routers.tutor as rt
from routers.tutor import TutorStartRequest, TutorSubmitRequest, TutorTurnResponse
from services.tutor_sessions import tutor_session_id

_LESSON = "Dijkstra 的核心是每次取出当前最短的未确定点……"
_QUIZ = {"questions": [{"question": "q", "options": ["A"], "answer": "A",
                        "explanation": "e", "source": "s", "type": "choice"}]}


class _Interrupt:
    """langgraph Interrupt 的最小替身（routers 读 .value）。"""
    def __init__(self, value):
        self.value = value


class _Snapshot:
    def __init__(self, values, nxt):
        self.values = values
        self.next = nxt


class _FakeGraph:
    def __init__(self, result, snapshot=None):
        self._result = result
        self._snapshot = snapshot

    async def ainvoke(self, *a, **kw):
        return self._result

    async def aget_state(self, config):
        return self._snapshot


@asynccontextmanager
async def _fake_checkpointer(_path):
    yield object()


def _patch_graph(result, snapshot=None):
    """把 router 用到的 checkpointer / 图工厂 / mastery 全换成替身。"""
    return [
        patch.object(rt, "open_sqlite_checkpointer", _fake_checkpointer),
        patch.object(rt, "compile_tutor_graph", lambda cp: _FakeGraph(result, snapshot)),
        patch.object(rt, "supervisor_enabled", lambda: True),
        patch.object(rt, "get_mastery", AsyncMock(return_value=0.6)),
        patch.object(rt, "build_returning_context", AsyncMock(return_value={"is_returning": False})),
    ]


class _GraphCase(unittest.IsolatedAsyncioTestCase):
    async def _run(self, coro_factory, result, snapshot=None):
        patches = _patch_graph(result, snapshot)
        for p in patches:
            p.start()
        try:
            return await coro_factory()
        finally:
            for p in patches:
                p.stop()


class TestLessonReachesClient(_GraphCase):
    """lesson 必须出现在 HTTP 响应里"""

    def test_response_model_has_lesson_field(self):
        self.assertIn("lesson", TutorTurnResponse.model_fields)

    async def test_start_interrupt_branch_returns_lesson(self):
        result = {"lesson": _LESSON, "supervisor_reason": "先讲再练",
                  "__interrupt__": [_Interrupt({"quiz": _QUIZ, "turn": 1})]}
        resp = await self._run(
            lambda: rt.tutor_start(TutorStartRequest(user_id="u1", document_id="doc1", goal="图论")),
            result,
        )
        self.assertTrue(resp.awaiting_answers)
        self.assertEqual(resp.lesson, _LESSON, "讲解和题目同一轮下发，响应必须两样都带")
        self.assertEqual(resp.quiz, _QUIZ)

    async def test_start_finish_branch_returns_lesson(self):
        result = {"lesson": _LESSON, "done": True, "turn": 1, "terminate_reason": "agent_finish"}
        resp = await self._run(
            lambda: rt.tutor_start(TutorStartRequest(user_id="u1", document_id="doc1", goal="图论")),
            result,
        )
        self.assertTrue(resp.done)
        self.assertEqual(resp.lesson, _LESSON, "只讲不出题就收尾时，讲解是这一轮唯一的产物")

    # 待作答令牌由 state 确定性推导，测试按同一规则算出来，
    # 才能过 submit 的 409 过期围栏。
    _PENDING = {"thread_id": "t1", "user_id": "u1", "document_id": "doc1",
                "turn": 2, "quiz": _QUIZ}

    def _submit_req(self):
        return TutorSubmitRequest(
            thread_id="t1",
            quiz_session_id=tutor_session_id(self._PENDING),
            answers=["A"],
        )

    async def test_submit_interrupt_branch_returns_lesson(self):
        result = {"lesson": _LESSON, "user_id": "u1", "document_id": "doc1",
                  "last_report": {"score": 0.5},
                  "__interrupt__": [_Interrupt({"quiz": _QUIZ, "turn": 2})]}
        resp = await self._run(
            lambda: rt.tutor_submit(self._submit_req()),
            result, _Snapshot(dict(self._PENDING), ("wait_for_answers",)),
        )
        self.assertEqual(resp.lesson, _LESSON)

    async def test_submit_finish_branch_returns_lesson(self):
        result = {"lesson": _LESSON, "user_id": "u1", "document_id": "doc1", "done": True}
        resp = await self._run(
            lambda: rt.tutor_submit(self._submit_req()),
            result, _Snapshot(dict(self._PENDING), ("wait_for_answers",)),
        )
        self.assertEqual(resp.lesson, _LESSON)

    async def test_no_lesson_stays_none(self):
        result = {"supervisor_reason": "直接出题",
                  "__interrupt__": [_Interrupt({"quiz": _QUIZ, "turn": 1})]}
        resp = await self._run(
            lambda: rt.tutor_start(TutorStartRequest(user_id="u1", document_id="doc1", goal="图论")),
            result,
        )
        self.assertIsNone(resp.lesson, "没派 tutor 的轮次不该凭空出现讲解")


class TestLessonDoesNotLeakAcrossTurns(unittest.IsolatedAsyncioTestCase):
    """grader_worker 清空 lesson，避免后续每轮都回显同一段讲解"""

    async def test_grader_worker_clears_lesson(self):
        import agents.grader_worker as gw
        report = {"session_id": "s", "total": 1, "correct": 1, "score": 1.0,
                  "grades": [{"index": 0, "question": "q", "user_answer": "A",
                              "correct_answer": "A", "is_correct": True, "knowledge_gap": None}]}
        from models.quiz import Question
        from models.session import QuizSession
        from services.session import sessions

        # grader_worker 会校验报告对应的 session 存在且属主一致（replay safety）
        sessions["s"] = QuizSession(
            session_id="s", document_id="doc1", user_id="u1",
            questions=[Question(question="q", options=["A"], answer="A",
                                explanation="e", source="s", type="choice")],
            user_answers=["A"], status="completed",
        )
        self.addCleanup(sessions.pop, "s", None)

        state = {"user_id": "u1", "document_id": "doc1", "session_id": "s",
                 "lesson": _LESSON, "history": [], "quiz": None}
        with patch.object(gw.grader_agent, "ainvoke", AsyncMock(return_value={"grading_report": report})), \
             patch.object(gw.adapt_writer, "ainvoke", AsyncMock(return_value={})), \
             patch.object(gw, "get_mastery", AsyncMock(return_value=0.9)), \
             patch.object(gw, "consolidate_session_extras", AsyncMock(return_value=None)), \
             patch.object(gw, "update_after_session", AsyncMock(return_value=None)):
            out = await gw.grader_worker(state)
        self.assertIn("lesson", out, "必须显式写回 None；不写的话 state 里的旧讲解会一直留着")
        self.assertIsNone(out["lesson"])


if __name__ == "__main__":
    unittest.main()
