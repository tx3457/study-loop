"""Durable retry contracts for per-question Web Quiz submissions."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from main import app
from models.quiz import Question
from models.session import AnswerRequest, QuizSession, QuizSessionAggregate
import routers.session as session_router
from services.quiz_sessions import QuizSessionStore


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
        root = Path(self.tempdir.name)
        self.quiz_store = QuizSessionStore(
            sqlite_path=str(root / "quiz-sessions.sqlite3")
        )
        self.patches = (patch.object(session_router, "quiz_sessions", self.quiz_store),)
        for active_patch in self.patches:
            active_patch.start()
        self.client = TestClient(app, raise_server_exceptions=False)

    def tearDown(self) -> None:
        self.client.close()
        for active_patch in reversed(self.patches):
            active_patch.stop()
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
        asyncio.run(self.quiz_store.create(QuizSessionAggregate(session=session)))
        return session

    def _stored_session(self, session_id: str) -> QuizSession:
        record = asyncio.run(self.quiz_store.inspect(session_id))
        self.assertIsNotNone(record)
        return record.aggregate.session

    def test_completed_answer_replays_without_advancing_twice(self) -> None:
        session = self._seed()
        write_back = AsyncMock()
        payload = {"answer": "A. 是", "question_index": 0}
        headers = {"Idempotency-Key": "quiz-answer-replay-1"}

        with patch.object(session_router, "write_objective_profile", write_back):
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
        snapshot = self.client.get(f"/session/{session.session_id}")
        self.assertEqual(snapshot.status_code, 200, snapshot.text)
        self.assertNotIn("answer_request_hashes", snapshot.text)
        self.assertNotIn(headers["Idempotency-Key"], snapshot.text)
        record = asyncio.run(self.quiz_store.inspect(session.session_id))
        self.assertEqual(len(record.aggregate.answer_request_hashes), 1)
        self.assertNotIn(
            headers["Idempotency-Key"],
            record.aggregate.answer_request_hashes,
        )
        stored = record.aggregate.session
        self.assertEqual(stored.user_answers, ["A. 是"])
        self.assertEqual(stored.status, "completed")
        write_back.assert_awaited_once()

    def test_stale_question_index_cannot_answer_the_next_question(self) -> None:
        session = self._seed(question_count=2)
        first = self.client.post(
            f"/session/{session.session_id}/answer",
            json={"answer": "A. 是", "question_index": 0},
        )
        self.assertEqual(first.status_code, 200, first.text)

        response = self.client.post(
            f"/session/{session.session_id}/answer",
            json={"answer": "A. 是", "question_index": 0},
            headers={"Idempotency-Key": "quiz-answer-stale-1"},
        )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["code"], "quiz_session_stale")
        self.assertEqual(
            self._stored_session(session.session_id).user_answers,
            ["A. 是"],
        )

    def test_lost_commit_ack_replays_from_durable_answer_binding(self) -> None:
        """The answer and its key committed atomically, but the acknowledgement was lost."""
        session = self._seed(question_count=2)
        original_complete = self.quiz_store.complete

        async def persist_then_fail(*args, **kwargs):
            await original_complete(*args, **kwargs)
            raise RuntimeError("quiz commit acknowledgement lost")

        headers = {"Idempotency-Key": "quiz-answer-commit-ack-lost"}
        payload = {"answer": "A. 是", "question_index": 0}
        with patch.object(
            self.quiz_store,
            "complete",
            side_effect=persist_then_fail,
        ):
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

        self.assertEqual(first.status_code, 503, first.text)
        self.assertEqual(first.json()["code"], "quiz_session_store_unavailable")
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(
            self._stored_session(session.session_id).user_answers,
            ["A. 是"],
        )

    def test_failed_commit_leaves_no_answer_or_key_binding_and_can_retry(self) -> None:
        session = self._seed(question_count=2)
        headers = {"Idempotency-Key": "quiz-answer-commit-failed"}
        payload = {"answer": "A. 是", "question_index": 0}

        with patch.object(
            self.quiz_store,
            "complete",
            AsyncMock(side_effect=RuntimeError("quiz store unavailable")),
        ):
            first = self.client.post(
                f"/session/{session.session_id}/answer",
                json=payload,
                headers=headers,
            )

        after_failure = self._stored_session(session.session_id)
        retry = self.client.post(
            f"/session/{session.session_id}/answer",
            json=payload,
            headers=headers,
        )

        self.assertEqual(first.status_code, 503, first.text)
        self.assertEqual(first.json()["code"], "quiz_session_store_unavailable")
        self.assertEqual(after_failure.user_answers, [])
        self.assertEqual(retry.status_code, 200, retry.text)
        self.assertEqual(
            self._stored_session(session.session_id).user_answers,
            ["A. 是"],
        )

    def test_same_key_with_different_answer_is_rejected(self) -> None:
        session = self._seed(question_count=2)
        headers = {"Idempotency-Key": "quiz-answer-payload-mismatch"}
        first = self.client.post(
            f"/session/{session.session_id}/answer",
            json={"answer": "A. 是", "question_index": 0},
            headers=headers,
        )
        conflict = self.client.post(
            f"/session/{session.session_id}/answer",
            json={"answer": "B. 否", "question_index": 0},
            headers=headers,
        )

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(conflict.status_code, 409, conflict.text)
        self.assertEqual(conflict.json()["reason"], "payload_mismatch")
        self.assertEqual(
            self._stored_session(session.session_id).user_answers,
            ["A. 是"],
        )

    def test_question_index_is_required_without_mutating_session(self) -> None:
        session = self._seed()

        response = self.client.post(
            f"/session/{session.session_id}/answer",
            json={"answer": "A. 是"},
        )

        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self._stored_session(session.session_id).user_answers, [])


class TestSessionAnswerCancellation(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.quiz_store = QuizSessionStore(
            sqlite_path=str(root / "quiz-sessions.sqlite3")
        )
        self.patches = (patch.object(session_router, "quiz_sessions", self.quiz_store),)
        for active_patch in self.patches:
            active_patch.start()
        session = QuizSession(
            session_id="quiz-cancel",
            document_id="notes.md",
            user_id="user-1",
            questions=[_question("问题", "A")],
            user_answers=[],
            status="active",
        )
        await self.quiz_store.create(QuizSessionAggregate(session=session))

    async def asyncTearDown(self) -> None:
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.tempdir.cleanup()

    async def test_cancelled_operation_releases_claim_and_allows_same_key_retry(self) -> None:
        entered = asyncio.Event()

        async def block_after_claim(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        request = AnswerRequest(answer="A. 是", question_index=0)
        key = "quiz-answer-cancelled-1"
        with patch.object(
            session_router,
            "apply_answer_to_session",
            side_effect=block_after_claim,
        ):
            task = asyncio.create_task(
                session_router.answer(
                    "quiz-cancel",
                    request,
                    idempotency_key=key,
                )
            )
            await asyncio.wait_for(entered.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        record = await self.quiz_store.inspect("quiz-cancel")
        self.assertIsNotNone(record)
        self.assertFalse(record.busy)
        self.assertEqual(record.aggregate.session.user_answers, [])
        retry = await session_router.answer(
            "quiz-cancel",
            request,
            idempotency_key=key,
        )
        self.assertTrue(retry.is_last)
        stored = await self.quiz_store.inspect("quiz-cancel")
        self.assertEqual(stored.aggregate.session.user_answers, ["A. 是"])


if __name__ == "__main__":
    unittest.main()
