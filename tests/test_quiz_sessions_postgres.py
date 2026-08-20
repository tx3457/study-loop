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

_ADMIN_CONNECT_TIMEOUT_SECONDS = 5
_ADMIN_LOCK_TIMEOUT_MS = 2_000
_ADMIN_STATEMENT_TIMEOUT_MS = 5_000
_FAST_LOCK_TIMEOUT_MS = 100
_FAST_STATEMENT_TIMEOUT_MS = 100
_TIMEOUT_ASSERTION_SECONDS = 6.0


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
        self.schema_lock_id = uuid.uuid4().int & ((1 << 63) - 1)
        self.table_patch = patch.object(
            quiz_sessions_module,
            "_TABLE",
            self.table_name,
        )
        self.schema_lock_patch = patch.object(
            quiz_sessions_module,
            "_POSTGRES_SCHEMA_LOCK_ID",
            self.schema_lock_id,
        )
        self.table_patch.start()
        self.schema_lock_patch.start()
        self.store = self._new_store()

    def tearDown(self) -> None:
        from psycopg import sql

        try:
            with self._admin_connect() as connection:
                connection.execute(
                    sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                        sql.Identifier(self.table_name)
                    )
                )
        finally:
            try:
                self.schema_lock_patch.stop()
            finally:
                self.table_patch.stop()

    @staticmethod
    def _admin_connect():
        import psycopg
        from psycopg.conninfo import conninfo_to_dict

        connection_parameters = conninfo_to_dict(TEST_DATABASE_URL)
        if "options" in connection_parameters:
            existing_options = (connection_parameters.get("options") or "").strip()
        else:
            existing_options = os.getenv("PGOPTIONS", "").strip()
        bounded_options = (
            f"-c lock_timeout={_ADMIN_LOCK_TIMEOUT_MS}ms "
            f"-c statement_timeout={_ADMIN_STATEMENT_TIMEOUT_MS}ms"
        )

        return psycopg.connect(
            TEST_DATABASE_URL,
            connect_timeout=_ADMIN_CONNECT_TIMEOUT_SECONDS,
            options=f"{existing_options} {bounded_options}".strip(),
        )

    def _new_store(
        self,
        *,
        max_count: int = 100,
        postgres_lock_timeout_ms: int = 5_000,
        postgres_statement_timeout_ms: int = 15_000,
    ) -> QuizSessionStore:
        return QuizSessionStore(
            database_url=TEST_DATABASE_URL,
            ttl_seconds=60,
            operation_lease_seconds=10,
            max_count=max_count,
            postgres_connect_timeout_seconds=5,
            postgres_lock_timeout_ms=postgres_lock_timeout_ms,
            postgres_statement_timeout_ms=postgres_statement_timeout_ms,
            clock=lambda: self.now[0],
        )

    async def _assert_finishes_with_database_error_while_blocked(
        self,
        operation,
        expected_exception: type[BaseException],
        release_blocker,
    ) -> None:
        task = asyncio.create_task(operation)
        done, _ = await asyncio.wait(
            {task},
            timeout=_TIMEOUT_ASSERTION_SECONDS,
        )
        exception = task.exception() if done else None

        release_blocker()
        if not done:
            # The database lock is gone now, so even a broken timeout
            # implementation can finish without leaking a worker into teardown.
            await asyncio.gather(task, return_exceptions=True)

        self.assertTrue(done, "database operation ignored its configured timeout")
        self.assertIsInstance(exception, expected_exception)

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
        logical_ids = {decision.record.aggregate.session.session_id for decision in decisions}
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

        successes = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
        failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], QuizSessionCapacityError)

        records = [
            await self._new_store(max_count=1).inspect(session_id)
            for session_id in (first_id, second_id)
        ]
        self.assertEqual(sum(record is not None for record in records), 1)

    async def test_schema_advisory_lock_times_out_then_initialization_retries(self) -> None:
        import psycopg

        blocked_store = self._new_store(
            postgres_lock_timeout_ms=_FAST_LOCK_TIMEOUT_MS,
            postgres_statement_timeout_ms=2_000,
        )
        with self._admin_connect() as blocker:
            blocker.execute(
                "SELECT pg_advisory_xact_lock(%s)",
                (self.schema_lock_id,),
            )
            await self._assert_finishes_with_database_error_while_blocked(
                blocked_store.inspect(f"missing-{uuid.uuid4().hex}"),
                psycopg.errors.LockNotAvailable,
                blocker.rollback,
            )

        self.assertIsNone(await blocked_store.inspect("missing-after-retry"))

    async def test_target_row_lock_times_out_then_claim_retries(self) -> None:
        import psycopg
        from psycopg import sql

        store = self._new_store(
            postgres_lock_timeout_ms=_FAST_LOCK_TIMEOUT_MS,
            postgres_statement_timeout_ms=2_000,
        )
        session_id = f"quiz-{uuid.uuid4().hex}"
        await store.create(_aggregate(session_id))
        with self._admin_connect() as blocker:
            locked = blocker.execute(
                sql.SQL("SELECT session_id FROM {} WHERE session_id = %s FOR UPDATE").format(
                    sql.Identifier(self.table_name)
                ),
                (session_id,),
            ).fetchone()
            self.assertEqual(locked, (session_id,))
            await self._assert_finishes_with_database_error_while_blocked(
                store.claim(session_id, "answer"),
                psycopg.errors.LockNotAvailable,
                blocker.rollback,
            )

        retry = await store.claim(session_id, "answer")
        self.assertTrue(retry.claimed)

    async def test_capacity_table_lock_times_out_then_create_retries(self) -> None:
        import psycopg
        from psycopg import sql

        store = self._new_store(
            postgres_lock_timeout_ms=_FAST_LOCK_TIMEOUT_MS,
            postgres_statement_timeout_ms=2_000,
        )
        self.assertIsNone(await store.inspect("initialize-schema"))
        with self._admin_connect() as blocker:
            blocker.execute(
                sql.SQL("LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE").format(
                    sql.Identifier(self.table_name)
                )
            )
            session_id = f"quiz-{uuid.uuid4().hex}"
            await self._assert_finishes_with_database_error_while_blocked(
                store.create(_aggregate(session_id)),
                psycopg.errors.LockNotAvailable,
                blocker.rollback,
            )

        created = await store.create(_aggregate(session_id))
        self.assertTrue(created.created)

    async def test_statement_timeout_cancels_pg_sleep_then_store_retries(self) -> None:
        import psycopg

        store = self._new_store(
            postgres_lock_timeout_ms=2_000,
            postgres_statement_timeout_ms=_FAST_STATEMENT_TIMEOUT_MS,
        )

        def execute_slow_query() -> None:
            with store._transaction() as connection:
                connection.execute("SELECT pg_sleep(1)").fetchone()

        def execute_fast_query():
            with store._transaction() as connection:
                return connection.execute("SELECT 1").fetchone()

        task = asyncio.create_task(asyncio.to_thread(execute_slow_query))
        done, _ = await asyncio.wait(
            {task},
            timeout=_TIMEOUT_ASSERTION_SECONDS,
        )
        exception = task.exception() if done else None
        if not done:
            await asyncio.gather(task, return_exceptions=True)

        self.assertTrue(done, "pg_sleep ignored the configured statement timeout")
        self.assertIsInstance(exception, psycopg.errors.QueryCanceled)
        self.assertEqual(await asyncio.to_thread(execute_fast_query), (1,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
