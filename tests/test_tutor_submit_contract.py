"""Tutor submit turn-token and crash-recovery routing contracts."""

import asyncio
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

import routers.tutor as tutor_router
from routers.tutor import TutorSubmitRequest
from services.tutor_sessions import tutor_session_id


def _pending_state() -> dict:
    return {
        "thread_id": "thread-1",
        "user_id": "user-1",
        "document_id": "notes.md",
        "turn": 1,
        "quiz": {
            "questions": [
                {
                    "question": "question",
                    "options": ["A", "B"],
                    "answer": "A",
                    "explanation": "explanation",
                    "source": "notes.md",
                    "type": "choice",
                }
            ]
        },
    }


class _FakeGraph:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.inputs = []

    async def aget_state(self, _config):
        return self.snapshot

    async def ainvoke(self, value, config=None):
        self.inputs.append(value)
        return {
            **self.snapshot.values,
            "done": True,
            "history": [],
            "terminate_reason": "agent_finish",
        }


@asynccontextmanager
async def _fake_checkpointer(_path):
    yield object()


class TestTutorSubmitContract(unittest.IsolatedAsyncioTestCase):
    async def _submit(self, graph: _FakeGraph, request: TutorSubmitRequest):
        with patch.object(tutor_router, "_require_enabled"), patch.object(
            tutor_router,
            "open_sqlite_checkpointer",
            _fake_checkpointer,
        ), patch.object(
            tutor_router,
            "compile_tutor_graph",
            return_value=graph,
        ), patch.object(
            tutor_router,
            "_mastery_of",
            AsyncMock(return_value=None),
        ):
            return await tutor_router.tutor_submit(request)

    async def test_pending_turn_requires_matching_quiz_session_token(self):
        state = _pending_state()
        graph = _FakeGraph(SimpleNamespace(values=state, next=("wait_for_answers",)))
        token = tutor_session_id(state)

        response = await self._submit(
            graph,
            TutorSubmitRequest(
                thread_id="thread-1",
                quiz_session_id=token,
                answers=[" A "],
            ),
        )

        self.assertTrue(response.done)
        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(graph.inputs[0].resume, ["A"])

    async def test_stale_turn_token_has_no_side_effect(self):
        state = _pending_state()
        graph = _FakeGraph(SimpleNamespace(values=state, next=("wait_for_answers",)))

        with self.assertRaises(HTTPException) as raised:
            await self._submit(
                graph,
                TutorSubmitRequest(
                    thread_id="thread-1",
                    quiz_session_id="tutor_stale-token",
                    answers=["A"],
                ),
            )

        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(graph.inputs, [])

    async def test_post_answer_checkpoint_continues_without_second_resume(self):
        state = _pending_state()
        state["session_id"] = tutor_session_id(state)
        state["answers"] = ["A"]
        graph = _FakeGraph(SimpleNamespace(values=state, next=("grader",)))

        response = await self._submit(
            graph,
            TutorSubmitRequest(
                thread_id="thread-1",
                quiz_session_id=state["session_id"],
                answers=["A"],
            ),
        )

        self.assertTrue(response.done)
        self.assertEqual(graph.inputs, [None])

    async def test_same_thread_submissions_are_serialized(self):
        state = _pending_state()
        token = tutor_session_id(state)
        entered = 0
        peak = 0

        class SlowGraph(_FakeGraph):
            async def aget_state(self, config):
                nonlocal entered, peak
                entered += 1
                peak = max(peak, entered)
                await asyncio.sleep(0.02)
                entered -= 1
                return await super().aget_state(config)

        graph = SlowGraph(SimpleNamespace(values=state, next=("wait_for_answers",)))
        request = TutorSubmitRequest(
            thread_id="thread-serialized",
            quiz_session_id=token,
            answers=["A"],
        )

        with patch.object(tutor_router, "_require_enabled"), patch.object(
            tutor_router,
            "open_sqlite_checkpointer",
            _fake_checkpointer,
        ), patch.object(
            tutor_router,
            "compile_tutor_graph",
            return_value=graph,
        ), patch.object(
            tutor_router,
            "_mastery_of",
            AsyncMock(return_value=None),
        ):
            await asyncio.gather(
                tutor_router.tutor_submit(request),
                tutor_router.tutor_submit(request),
            )
        self.assertEqual(peak, 1)

    def test_answers_are_trimmed_and_bounded(self):
        request = TutorSubmitRequest(
            thread_id="thread-1",
            quiz_session_id="tutor_valid-token",
            answers=[" A "],
        )
        self.assertEqual(request.answers, ["A"])
        for answers in ([], ["   "], ["x" * 4001]):
            with self.subTest(answers=answers), self.assertRaises(ValidationError):
                TutorSubmitRequest(
                    thread_id="thread-1",
                    quiz_session_id="tutor_valid-token",
                    answers=answers,
                )


if __name__ == "__main__":
    unittest.main()
