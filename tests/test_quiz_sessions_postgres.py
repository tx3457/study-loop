"""Live PostgreSQL contracts for durable Web Quiz sessions.

Set ``TEST_DATABASE_URL`` to run these tests. The default suite skips them so
local SQLite development does not require a PostgreSQL service.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from unittest.mock import patch

from models.quiz import Question
from models.session import AnswerResult, QuizSession, QuizSessionAggregate
import services.quiz_sessions as quiz_sessions_module
from services.quiz_sessions import (
    QuizSessionCapacityError,
    QuizSessionStore,
)


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


def _aggregate(session_id: str) -> QuizSessionAggregate:
    return QuizSessionAggregate(
        session=QuizSession(
            session_id=session_id,
            document_id="notes.md",
            user_id="postgres-user",
            questions=[
                Question(
                    question="What is RAG?",
                    options=["A. Retrieval augmented generation", "B. A cache"],
                    answer="A",
                    explanation="RAG grounds generation with retrieved evidence.",
                    source="chunk-postgres",
                    type="choice",
                )
            ],
            user_answers=[],
            status="active",
        )
    )


def _complete(aggregate: QuizSessionAggregate) -> QuizSessionAggregate:
    completed = aggregate.model_copy(deep=True)
    completed.session.user_answers.append("A")
    completed.session.status = "completed"
    completed.last_answer_index = 0
    completed.last_answer_result = AnswerResult(
        correct=True,
        correct_answer="A",
        explanation="RAG grounds generation with retrieved evidence.",
        is_last=True,
    )
    return completed


@unittest.skipUnless(TEST_DATABASE_URL, "TEST_DATABASE_URL is not configured")
class TestPostgresQuizSessionStore(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.now = [1_000.0]
        self.table_name = f"studyloop_quiz_test_{uuid.uuid4().hex}"
        self.table_patch = patch.object(
            quiz_sessions_module,
            "_TABLE",
            self.table_name,
        )
        self.table_patch.start()
        self.store = self._new_store()

    def tearDown(self) -> None:
        import psycopg
        from psycopg import sql

        try:
            with psycopg.connect(TEST_DATABASE_URL) as connection:
                connection.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(
                        sql.Identifier(self.table_name)
                    )
                )
        finally:
            self.table_patch.stop()

    def _new_store(self, *, max_count: int = 100) -> QuizSessionStore:
        return QuizSessionStore(
            database_url=TEST_DATABASE_URL,
            ttl_seconds=60,
            operation_lease_seconds=10,
            max_count=max_count,
            clock=lambda: self.now[0],
        )

    async def test_reopen_and_claim_fencing_work_across_connections(self) -> None:
        session_id = f"quiz-{uuid.uuid4().hex}"
        await self.store.create(_aggregate(session_id))
        reopened = self._new_store()

        restored = await reopened.inspect(session_id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.aggregate.session.session_id, session_id)
        self.assertEqual(restored.aggregate.session.questions[0].answer, "A")

        owner = await self.store.claim(session_id, "answer")
        blocked = await reopened.claim(session_id, "grade")
        self.assertTrue(owner.claimed)
        self.assertFalse(blocked.claimed)
        self.assertEqual(blocked.reason, "in_progress")
        self.assertFalse(await reopened.release(session_id, "wrong-token"))

        completed = await reopened.complete(
            session_id,
            owner.token,
            _complete(owner.record.aggregate),
        )
        self.assertIsNotNone(completed)
        self.assertEqual(completed.revision, 2)
        self.assertFalse(completed.busy)
        self.assertIsNone(
            await self.store.complete(
                session_id,
                owner.token,
                _complete(owner.record.aggregate),
            )
        )

        after_reopen = await self._new_store().inspect(session_id)
        self.assertEqual(after_reopen.aggregate.session.status, "completed")
        self.assertEqual(after_reopen.aggregate.session.user_answers, ["A"])

    async def test_concurrent_same_start_key_creates_one_logical_session(self) -> None:
        first = self._new_store()
        second = self._new_store()
        first_id = f"quiz-{uuid.uuid4().hex}"
        second_id = f"quiz-{uuid.uuid4().hex}"
        start_key = f"quiz-start-{uuid.uuid4().hex}"
        request = {
            "document_id": "notes.md",
            "user_id": "postgres-user",
            "count": 1,
        }

        decisions = await asyncio.gather(
            first.create(
                _aggregate(first_id),
                start_key=start_key,
                start_request=request,
            ),
            second.create(
                _aggregate(second_id),
                start_key=start_key,
                start_request=request,
            ),
        )

        self.assertEqual(sum(decision.created for decision in decisions), 1)
        logical_ids = {
            decision.record.aggregate.session.session_id for decision in decisions
        }
        self.assertEqual(len(logical_ids), 1)
        logical_id = logical_ids.pop()
        self.assertIn(logical_id, {first_id, second_id})
        self.assertIsNotNone(await self._new_store().inspect(logical_id))
        losing_id = second_id if logical_id == first_id else first_id
        self.assertIsNone(await self._new_store().inspect(losing_id))

    async def test_concurrent_creates_preserve_postgres_capacity_limit(self) -> None:
        first = self._new_store(max_count=1)
        second = self._new_store(max_count=1)
        first_id = f"quiz-{uuid.uuid4().hex}"
        second_id = f"quiz-{uuid.uuid4().hex}"

        outcomes = await asyncio.gather(
            first.create(_aggregate(first_id)),
            second.create(_aggregate(second_id)),
            return_exceptions=True,
        )

        successes = [
            outcome
            for outcome in outcomes
            if not isinstance(outcome, BaseException)
        ]
        failures = [
            outcome for outcome in outcomes if isinstance(outcome, BaseException)
        ]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], QuizSessionCapacityError)

        records = [
            await self._new_store(max_count=1).inspect(session_id)
            for session_id in (first_id, second_id)
        ]
        self.assertEqual(sum(record is not None for record in records), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
