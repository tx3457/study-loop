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


def _question(
    text: str,
    answer: str,
    *,
    question_type: str = "choice",
) -> Question:
    return Question(
        question=text,
        options=["A. 是", "B. 否"],
        answer=answer,
        explanation="测试解析",
        source="notes.md",
        type=question_type,
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

    def _seed(
        self,
        *,
        question_count: int = 1,
        question_type: str = "choice",
    ) -> QuizSession:
        session = QuizSession(
            session_id="quiz-idempotency",
            document_id="notes.md",
            user_id="user-1",
            questions=[
                _question(
                    f"问题 {index + 1}",
                    "A",
                    question_type=question_type,
                )
                for index in range(question_count)
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

        async def publish_marker(stored_session, _session_id):
            stored_session.profile_written = True

        write_back = AsyncMock(side_effect=publish_marker)
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

    def test_completed_answer_replay_repairs_failed_objective_memory(self) -> None:
        session = self._seed()
        calls = 0

        async def fail_then_publish(stored_session, _session_id):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("memory unavailable")
            stored_session.profile_written = True

        payload = {"answer": "A. 是", "question_index": 0}
        headers = {"Idempotency-Key": "quiz-answer-memory-repair"}
        with patch.object(
            session_router,
            "write_objective_profile",
            side_effect=fail_then_publish,
        ):
            first = self.client.post(
                f"/session/{session.session_id}/answer",
                json=payload,
                headers=headers,
            )
            after_failure = asyncio.run(self.quiz_store.inspect(session.session_id))
            replay = self.client.post(
                f"/session/{session.session_id}/answer",
                json=payload,
                headers=headers,
            )

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertFalse(after_failure.aggregate.session.profile_written)
        self.assertGreater(replay.json()["revision"], first.json()["revision"])
        self.assertEqual(calls, 2)
        persisted = asyncio.run(self.quiz_store.inspect(session.session_id))
        self.assertTrue(persisted.aggregate.session.profile_written)
        self.assertEqual(persisted.aggregate.session.user_answers, ["A. 是"])

    def test_snapshot_repairs_objective_memory_when_answer_was_acknowledged(self) -> None:
        session = self._seed()
        payload = {"answer": "A. 是", "question_index": 0}

        with patch.object(
            session_router,
            "write_objective_profile",
            AsyncMock(side_effect=RuntimeError("memory unavailable")),
        ):
            answered = self.client.post(
                f"/session/{session.session_id}/answer",
                json=payload,
            )

        async def publish_marker(stored_session, _session_id):
            stored_session.profile_written = True

        write_back = AsyncMock(side_effect=publish_marker)
        with patch.object(session_router, "write_objective_profile", write_back):
            recovered = self.client.get(f"/session/{session.session_id}")

        self.assertEqual(answered.status_code, 200, answered.text)
        self.assertEqual(recovered.status_code, 200, recovered.text)
        self.assertEqual(recovered.json()["status"], "completed")
        write_back.assert_awaited_once()
        persisted = asyncio.run(self.quiz_store.inspect(session.session_id))
        self.assertTrue(persisted.aggregate.session.profile_written)

    def test_result_repairs_objective_memory_when_answer_was_acknowledged(self) -> None:
        session = self._seed()
        with patch.object(
            session_router,
            "write_objective_profile",
            AsyncMock(side_effect=RuntimeError("memory unavailable")),
        ):
            answered = self.client.post(
                f"/session/{session.session_id}/answer",
                json={"answer": "A. 是", "question_index": 0},
            )

        async def publish_marker(stored_session, _session_id):
            stored_session.profile_written = True

        write_back = AsyncMock(side_effect=publish_marker)
        with patch.object(session_router, "write_objective_profile", write_back):
            result = self.client.get(f"/session/{session.session_id}/result")

        self.assertEqual(answered.status_code, 200, answered.text)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["correct"], 1)
        write_back.assert_awaited_once()
        persisted = asyncio.run(self.quiz_store.inspect(session.session_id))
        self.assertTrue(persisted.aggregate.session.profile_written)

    def test_short_answer_recovery_never_uses_objective_memory_writer(self) -> None:
        session = self._seed(question_type="short_answer")
        write_back = AsyncMock()

        with patch.object(session_router, "write_objective_profile", write_back):
            answered = self.client.post(
                f"/session/{session.session_id}/answer",
                json={"answer": "free text", "question_index": 0},
            )
            recovered = self.client.get(f"/session/{session.session_id}")
            result = self.client.get(f"/session/{session.session_id}/result")

        self.assertEqual(answered.status_code, 200, answered.text)
        self.assertEqual(recovered.status_code, 200, recovered.text)
        self.assertEqual(result.status_code, 200, result.text)
        write_back.assert_not_awaited()

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

    def test_lost_final_answer_ack_replays_and_repairs_objective_memory(self) -> None:
        session = self._seed()
        original_complete = self.quiz_store.complete

        async def persist_then_fail(*args, **kwargs):
            await original_complete(*args, **kwargs)
            raise RuntimeError("quiz commit acknowledgement lost")

        headers = {"Idempotency-Key": "quiz-final-answer-ack-lost"}
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

        async def publish_marker(stored_session, _session_id):
            stored_session.profile_written = True

        write_back = AsyncMock(side_effect=publish_marker)
        with patch.object(session_router, "write_objective_profile", write_back):
            replay = self.client.post(
                f"/session/{session.session_id}/answer",
                json=payload,
                headers=headers,
            )

        self.assertEqual(first.status_code, 503, first.text)
        self.assertEqual(replay.status_code, 200, replay.text)
        write_back.assert_awaited_once()
        stored = asyncio.run(self.quiz_store.inspect(session.session_id))
        self.assertEqual(stored.aggregate.session.user_answers, ["A. 是"])
        self.assertTrue(stored.aggregate.session.profile_written)

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

    async def test_cancelled_objective_memory_releases_claim_for_repair(self) -> None:
        entered = asyncio.Event()

        async def block_memory(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        request = AnswerRequest(answer="A. 是", question_index=0)
        key = "quiz-answer-memory-cancelled"
        with patch.object(
            session_router,
            "write_objective_profile",
            side_effect=block_memory,
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

        checkpoint = await self.quiz_store.inspect("quiz-cancel")
        self.assertFalse(checkpoint.busy)
        self.assertEqual(checkpoint.aggregate.session.status, "completed")
        self.assertFalse(checkpoint.aggregate.session.profile_written)

        async def publish_marker(stored_session, _session_id):
            stored_session.profile_written = True

        with patch.object(
            session_router,
            "write_objective_profile",
            side_effect=publish_marker,
        ):
            replay = await session_router.answer(
                "quiz-cancel",
                request,
                idempotency_key=key,
            )

        self.assertTrue(replay.is_last)
        repaired = await self.quiz_store.inspect("quiz-cancel")
        self.assertFalse(repaired.busy)
        self.assertTrue(repaired.aggregate.session.profile_written)


if __name__ == "__main__":
    unittest.main()
