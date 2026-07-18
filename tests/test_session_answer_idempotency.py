"""Durable retry contracts for per-question quiz submissions."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from main import app
from models.quiz import Question
from models.session import AnswerRequest, QuizSession
import routers.session as session_router
import services.session as session_service
from services.idempotency import IdempotencyStore


def _question(text: str, answer: str) -> Question:
    return Question(
        question=text,
        options=["A. 是", "B. 否"],
        answer=answer,
        explanation="测试解析",
        source="notes.md",
        type="choice",
    )


class TestSessionAnswerIdempotency(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = IdempotencyStore(
            sqlite_path=str(Path(self.tempdir.name) / "receipts.sqlite3")
        )
        self.store_patch = patch.object(
            session_router, "request_idempotency", self.store
        )
        self.store_patch.start()
        session_service.sessions.clear()
        session_router._answer_locks.clear()
        self.client = TestClient(app, raise_server_exceptions=False)

    def tearDown(self) -> None:
        self.store_patch.stop()
        session_service.sessions.clear()
        session_router._answer_locks.clear()
        self.tempdir.cleanup()

    def _seed(self, *, question_count: int = 1) -> QuizSession:
        session = QuizSession(
            session_id="quiz-idempotency",
            document_id="notes.md",
            user_id="user-1",
            questions=[
                _question(f"问题 {index + 1}", "A") for index in range(question_count)
            ],
            user_answers=[],
            status="active",
        )
        session_service.sessions[session.session_id] = session
        return session

    def test_completed_answer_replays_without_advancing_twice(self) -> None:
        session = self._seed()
        write_back = AsyncMock()
        payload = {"answer": "A. 是", "question_index": 0}
        headers = {"Idempotency-Key": "quiz-answer-replay-1"}

        with patch.object(session_service, "_write_back_profile", write_back):
            first = self.client.post(
                f"/session/{session.session_id}/answer",
                json=payload,
                headers=headers,
            )
            replay = self.client.post(
                f"/session/{session.session_id}/answer",
                json=payload,
                headers=headers,
            )

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(replay.json(), first.json())
        self.assertEqual(session.user_answers, ["A. 是"])
        self.assertEqual(write_back.await_count, 1)

    def test_stale_question_index_cannot_answer_the_next_question(self) -> None:
        session = self._seed(question_count=2)
        session.user_answers.append("A. 是")

        response = self.client.post(
            f"/session/{session.session_id}/answer",
            json={"answer": "A. 是", "question_index": 0},
            headers={"Idempotency-Key": "quiz-answer-stale-1"},
        )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(session.user_answers, ["A. 是"])

    def test_failure_after_commit_is_ambiguous_and_session_fails_closed(self) -> None:
        session = self._seed(question_count=2)

        async def mutate_then_fail(
            session_id,
            answer,
            *,
            question_index=None,
            before_commit=None,
        ):
            self.assertEqual(session_id, session.session_id)
            self.assertEqual(question_index, 0)
            await before_commit()
            session.user_answers.append(answer)
            raise RuntimeError("connection lost after commit")

        headers = {"Idempotency-Key": "quiz-answer-ambiguous-1"}
        payload = {"answer": "A. 是", "question_index": 0}
        with patch.object(session_router, "submit_answer", mutate_then_fail):
            first = self.client.post(
                f"/session/{session.session_id}/answer",
                json=payload,
                headers=headers,
            )
            replay = self.client.post(
                f"/session/{session.session_id}/answer",
                json=payload,
                headers=headers,
            )

        self.assertEqual(first.status_code, 409, first.text)
        self.assertEqual(first.json()["code"], "side_effect_ambiguous")
        self.assertEqual(replay.status_code, 409, replay.text)
        self.assertEqual(replay.json()["reason"], "ambiguous")
        self.assertEqual(session.user_answers, ["A. 是"])
        self.assertEqual(session.status, "ambiguous")

    def test_abort_store_failure_after_effect_still_fails_session_closed(self) -> None:
        session = self._seed(question_count=2)

        async def mutate_then_fail(
            session_id,
            answer,
            *,
            question_index=None,
            before_commit=None,
        ):
            await before_commit()
            session.user_answers.append(answer)
            raise RuntimeError("connection lost after commit")

        with (
            patch.object(session_router, "submit_answer", mutate_then_fail),
            patch.object(
                self.store,
                "abort",
                AsyncMock(side_effect=RuntimeError("receipt database unavailable")),
            ),
        ):
            response = self.client.post(
                f"/session/{session.session_id}/answer",
                json={"answer": "A. 是", "question_index": 0},
                headers={"Idempotency-Key": "quiz-answer-abort-failure-1"},
            )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["code"], "side_effect_ambiguous")
        self.assertEqual(session.status, "ambiguous")
        self.assertEqual(session.user_answers, ["A. 是"])

    def test_question_index_is_required(self) -> None:
        session = self._seed()

        response = self.client.post(
            f"/session/{session.session_id}/answer",
            json={"answer": "A. 是"},
        )

        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(session.user_answers, [])


class TestSessionAnswerCancellation(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = IdempotencyStore(
            sqlite_path=str(Path(self.tempdir.name) / "receipts.sqlite3")
        )
        self.store_patch = patch.object(
            session_router,
            "request_idempotency",
            self.store,
        )
        self.store_patch.start()
        session_service.sessions.clear()
        session_router._answer_locks.clear()
        session_service.sessions["quiz-cancel"] = QuizSession(
            session_id="quiz-cancel",
            document_id="notes.md",
            user_id="user-1",
            questions=[_question("问题", "A")],
            user_answers=[],
            status="active",
        )

    async def asyncTearDown(self) -> None:
        self.store_patch.stop()
        session_service.sessions.clear()
        session_router._answer_locks.clear()
        self.tempdir.cleanup()

    async def test_cancelled_lock_wait_releases_clean_receipt(self) -> None:
        lock = asyncio.Lock()
        await lock.acquire()
        session_router._answer_locks["quiz-cancel"] = lock
        claimed = asyncio.Event()
        original_begin = self.store.begin

        async def begin(*args, **kwargs):
            decision = await original_begin(*args, **kwargs)
            claimed.set()
            return decision

        request = AnswerRequest(answer="A. 是", question_index=0)
        key = "quiz-answer-cancelled-1"
        try:
            with patch.object(self.store, "begin", side_effect=begin):
                task = asyncio.create_task(
                    session_router.answer(
                        "quiz-cancel",
                        request,
                        idempotency_key=key,
                    )
                )
                await asyncio.wait_for(claimed.wait(), timeout=2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        finally:
            lock.release()

        retry = await self.store.begin(
            key,
            "session.answer",
            {
                "session_id": "quiz-cancel",
                **request.model_dump(mode="json"),
            },
        )
        self.assertFalse(retry.replayed)
        await self.store.abort(key)


if __name__ == "__main__":
    unittest.main()
