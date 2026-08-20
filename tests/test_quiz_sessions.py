import asyncio
import hashlib
import json
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from models.quiz import Question
from models.session import (
    AnswerResult,
    LearningPathQuizSource,
    QuizSession,
    QuizSessionAggregate,
)
from services.quiz_sessions import (
    QuizSessionCapacityError,
    QuizSessionCorruptError,
    QuizSessionPayloadTooLargeError,
    QuizSessionStartConflictError,
    QuizSessionStore,
)


def _question(label: str = "A") -> Question:
    return Question(
        question=f"Question {label}",
        options=["A. Alpha", "B. Beta"],
        answer="A",
        explanation="Alpha is correct",
        source=f"chunk-{label}",
        type="choice",
    )


def _aggregate(
    session_id: str,
    *,
    completed: bool = False,
    origin: str = "standard",
) -> QuizSessionAggregate:
    answers = ["A"] if completed else []
    session = QuizSession(
        session_id=session_id,
        document_id="notes.md",
        user_id="user-1",
        questions=[_question(session_id)],
        user_answers=answers,
        status="completed" if completed else "active",
    )
    return QuizSessionAggregate(
        origin=origin,
        session=session,
        last_answer_index=0 if completed else None,
        last_answer_result=(
            AnswerResult(
                correct=True,
                correct_answer="A",
                explanation="Alpha is correct",
                is_last=True,
            )
            if completed
            else None
        ),
    )


class TestQuizSessionStore(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "quiz.sqlite3")
        self.now = [1000.0]
        self.store = QuizSessionStore(
            sqlite_path=self.db_path,
            ttl_seconds=60,
            operation_lease_seconds=10,
            max_count=3,
            clock=lambda: self.now[0],
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    async def test_reopens_payload_and_replays_start_key(self):
        request = {"document_id": "notes.md", "count": 1}
        created = await self.store.create(
            _aggregate("s-1"),
            start_key="start-key-123",
            start_request=request,
        )
        self.assertTrue(created.created)
        self.assertEqual(created.record.revision, 1)

        peer = QuizSessionStore(
            sqlite_path=self.db_path,
            ttl_seconds=60,
            operation_lease_seconds=10,
            max_count=3,
            clock=lambda: self.now[0],
        )
        replay = await peer.find_start("start-key-123", request)
        self.assertIsNotNone(replay)
        self.assertEqual(replay.aggregate.session.session_id, "s-1")
        self.assertEqual(replay.aggregate.session.questions[0].answer, "A")

        decision = await peer.create(
            _aggregate("s-2"),
            start_key="start-key-123",
            start_request=request,
        )
        self.assertFalse(decision.created)
        self.assertEqual(decision.record.aggregate.session.session_id, "s-1")

        with self.assertRaises(QuizSessionStartConflictError):
            await peer.find_start(
                "start-key-123",
                {"document_id": "other.md", "count": 1},
            )

    def test_optional_path_binding_preserves_legacy_immutable_hash(self):
        aggregate = _aggregate("legacy-hash")
        session = aggregate.session
        canonical = json.dumps(
            {
                "session_id": session.session_id,
                "document_id": session.document_id,
                "user_id": session.user_id,
                "questions": [
                    question.model_dump(mode="json")
                    for question in session.questions
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        legacy_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.assertEqual(self.store._immutable_hash(aggregate), legacy_hash)

        aggregate.learning_path_source = LearningPathQuizSource(
            learning_path_id="lp_00000000000000000000000000000000",
            stage_id=1,
        )
        self.assertNotEqual(self.store._immutable_hash(aggregate), legacy_hash)

    async def test_claim_checkpoint_complete_and_fencing_takeover(self):
        await self.store.create(_aggregate("s-1"))
        first = await self.store.claim("s-1", "grade")
        self.assertTrue(first.claimed)

        peer = QuizSessionStore(
            sqlite_path=self.db_path,
            ttl_seconds=60,
            operation_lease_seconds=10,
            max_count=3,
            clock=lambda: self.now[0],
        )
        blocked = await peer.claim("s-1", "answer")
        self.assertFalse(blocked.claimed)
        self.assertEqual(blocked.reason, "in_progress")

        aggregate = first.record.aggregate.model_copy(deep=True)
        aggregate.session.user_answers.append("A")
        aggregate.session.status = "completed"
        aggregate.last_answer_index = 0
        aggregate.last_answer_result = AnswerResult(
            correct=True,
            correct_answer="A",
            explanation="Alpha is correct",
            is_last=True,
        )
        checkpoint = await self.store.checkpoint("s-1", first.token, aggregate)
        self.assertEqual(checkpoint.revision, 2)
        self.assertTrue(checkpoint.busy)

        self.now[0] += 11
        takeover = await peer.claim("s-1", "report")
        self.assertTrue(takeover.claimed)
        self.assertIsNone(await self.store.complete("s-1", first.token, aggregate))

        completed = await peer.complete("s-1", takeover.token, aggregate)
        self.assertEqual(completed.revision, 3)
        self.assertFalse(completed.busy)
        self.assertEqual(completed.expires_at, self.now[0] + 60)

    async def test_cancelled_claim_releases_ownership_not_returned_to_caller(self):
        await self.store.create(_aggregate("s-cancelled-claim"))
        entered = threading.Event()
        finish = threading.Event()
        original_claim = self.store._claim_sync

        def commit_then_wait(*args):
            decision = original_claim(*args)
            entered.set()
            finish.wait(timeout=2)
            return decision

        with patch.object(self.store, "_claim_sync", side_effect=commit_then_wait):
            task = asyncio.create_task(
                self.store.claim("s-cancelled-claim", "answer")
            )
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            task.cancel()
            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        restored = await self.store.inspect("s-cancelled-claim")
        self.assertIsNotNone(restored)
        self.assertFalse(restored.busy)
        retry = await self.store.claim("s-cancelled-claim", "answer")
        self.assertTrue(retry.claimed)
        self.assertTrue(await self.store.release("s-cancelled-claim", retry.token))

    async def test_immutable_question_change_cannot_commit(self):
        await self.store.create(_aggregate("s-1"))
        claim = await self.store.claim("s-1", "answer")
        aggregate = claim.record.aggregate.model_copy(deep=True)
        aggregate.session.questions[0].answer = "B"
        self.assertIsNone(await self.store.complete("s-1", claim.token, aggregate))
        self.assertTrue(await self.store.release("s-1", claim.token))

        stored = await self.store.inspect("s-1")
        self.assertEqual(stored.aggregate.session.questions[0].answer, "A")

    async def test_ttl_and_capacity_never_evict_live_active_session(self):
        limited = QuizSessionStore(
            sqlite_path=self.db_path,
            ttl_seconds=60,
            operation_lease_seconds=10,
            max_count=2,
            clock=lambda: self.now[0],
        )
        await limited.create(_aggregate("active-1"))
        await limited.create(_aggregate("active-2"))
        with self.assertRaises(QuizSessionCapacityError):
            await limited.create(_aggregate("active-3"))

        self.now[0] += 61
        expired = await limited.inspect("active-1")
        self.assertTrue(expired.expired)
        claim = await limited.claim("active-1", "answer")
        self.assertFalse(claim.claimed)
        self.assertEqual(claim.reason, "expired")

        created = await limited.create(_aggregate("active-3"))
        self.assertTrue(created.created)
        self.assertIsNone(await limited.inspect("active-1"))

    async def test_operation_cannot_checkpoint_or_revive_session_after_ttl(self):
        await self.store.create(_aggregate("expires-during-operation"))
        self.now[0] = 1059.0
        claim = await self.store.claim("expires-during-operation", "grade")
        self.assertTrue(claim.claimed)

        # The operation lease is still live at 1061, while the session TTL
        # ended at 1060. This isolates the session-expiry fence.
        self.now[0] = 1061.0
        aggregate = claim.record.aggregate.model_copy(deep=True)
        self.assertIsNone(
            await self.store.checkpoint(
                "expires-during-operation",
                claim.token,
                aggregate,
            )
        )
        self.assertIsNone(
            await self.store.complete(
                "expires-during-operation",
                claim.token,
                aggregate,
            )
        )
        self.assertTrue(
            await self.store.release("expires-during-operation", claim.token)
        )

        expired = await self.store.inspect("expires-during-operation")
        self.assertIsNotNone(expired)
        self.assertTrue(expired.expired)
        self.assertEqual(expired.expires_at, 1060.0)

    async def test_oldest_completed_session_is_capacity_evictable(self):
        limited = QuizSessionStore(
            sqlite_path=self.db_path,
            max_count=2,
            clock=lambda: self.now[0],
        )
        await limited.create(_aggregate("done", completed=True))
        self.now[0] += 1
        await limited.create(_aggregate("active"))
        self.now[0] += 1
        await limited.create(_aggregate("new"))
        self.assertIsNone(await limited.inspect("done"))
        self.assertIsNotNone(await limited.inspect("active"))

    async def test_corrupt_payload_and_unknown_schema_fail_closed(self):
        await self.store.create(_aggregate("s-1"))
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "UPDATE studyloop_quiz_sessions SET payload_json = ? WHERE session_id = ?",
                (json.dumps({"schema_version": 1, "session": {}}), "s-1"),
            )
            connection.commit()
        with self.assertRaises(QuizSessionCorruptError):
            await self.store.inspect("s-1")

        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("PRAGMA ignore_check_constraints = ON")
            connection.execute(
                "UPDATE studyloop_quiz_sessions SET schema_version = 2 WHERE session_id = ?",
                ("s-1",),
            )
            connection.commit()
        with self.assertRaises(QuizSessionCorruptError):
            await self.store.inspect("s-1")

    async def test_payload_size_is_checked_before_database_write(self):
        tiny = QuizSessionStore(
            sqlite_path=self.db_path,
            max_payload_bytes=200,
            clock=lambda: self.now[0],
        )
        with self.assertRaises(QuizSessionPayloadTooLargeError):
            await tiny.create(_aggregate("too-large"))
        self.assertFalse(Path(self.db_path).exists())


class TestQuizSessionAggregate(unittest.TestCase):
    def test_rejects_inconsistent_status_and_feedback(self):
        session = QuizSession(
            session_id="broken",
            document_id="notes.md",
            user_id="u",
            questions=[_question()],
            user_answers=["A"],
            status="active",
        )
        with self.assertRaises(ValueError):
            QuizSessionAggregate(session=session)

        session.status = "completed"
        with self.assertRaises(ValueError):
            QuizSessionAggregate(session=session)


if __name__ == "__main__":
    unittest.main()
