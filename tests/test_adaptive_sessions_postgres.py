"""Live PostgreSQL contracts for durable Adaptive sessions.

Set TEST_DATABASE_URL to run these tests. The default suite skips them so
local development does not require a PostgreSQL service.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from unittest.mock import patch

from models.adaptive import AdaptiveTurn, NextStepDecision
from models.adaptive_session import (
    AdaptiveSessionAggregate,
    AdaptiveTurnArtifact,
)
import services.adaptive_sessions as adaptive_sessions_module
from services.adaptive_sessions import (
    AdaptiveSessionCapacityError,
    AdaptiveSessionStore,
)


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

_ADMIN_CONNECT_TIMEOUT_SECONDS = 5
_ADMIN_LOCK_TIMEOUT_MS = 2_000
_ADMIN_STATEMENT_TIMEOUT_MS = 5_000
_FAST_LOCK_TIMEOUT_MS = 100
_FAST_STATEMENT_TIMEOUT_MS = 100
_TIMEOUT_ASSERTION_SECONDS = 6.0


def _aggregate(
    session_id: str,
    *,
    lesson: str = "Compare the midpoint and retain the possible half.",
) -> AdaptiveSessionAggregate:
    return AdaptiveSessionAggregate(
        adaptive_session_id=session_id,
        user_id="postgres-user",
        document_id="notes.md",
        goal="learn binary search",
        status="active",
        current_quiz=None,
        current_artifact=AdaptiveTurnArtifact(
            adaptive_session_id=session_id,
            turn=1,
            turn_type="teach",
            lesson=lesson,
            decision=NextStepDecision(
                action="teach",
                topic="binary search",
                difficulty="medium",
                difficulty_score=0.5,
                question_type="choice",
                count=1,
                reason="Teach before the next quiz.",
            ),
            trajectory=[
                AdaptiveTurn(
                    turn=1,
                    action="teach",
                    topic="binary search",
                    difficulty_score=0.5,
                    reason="Teach before the next quiz.",
                )
            ],
        ),
    )


@unittest.skipUnless(TEST_DATABASE_URL, "TEST_DATABASE_URL is not configured")
class TestPostgresAdaptiveSessionStore(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.now = [1_000.0]
        self.table_name = f"studyloop_adaptive_test_{uuid.uuid4().hex}"
        self.schema_lock_id = uuid.uuid4().int & ((1 << 63) - 1)
        self.table_patch = patch.object(
            adaptive_sessions_module,
            "_TABLE",
            self.table_name,
        )
        self.schema_lock_patch = patch.object(
            adaptive_sessions_module,
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
    ) -> AdaptiveSessionStore:
        return AdaptiveSessionStore(
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
        completed_while_blocked = task in done
        exception = task.exception() if completed_while_blocked else None

        release_blocker()
        await asyncio.gather(task, return_exceptions=True)

        self.assertTrue(
            completed_while_blocked,
            "database operation ignored its configured timeout",
        )
        self.assertIsInstance(exception, expected_exception)

    async def test_two_connections_reopen_take_over_and_fence(self) -> None:
        session_id = f"adaptive-{uuid.uuid4().hex}"
        await self.store.create(_aggregate(session_id))
        peer = self._new_store()

        restored = await peer.inspect(session_id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.aggregate.adaptive_session_id, session_id)

        owner = await self.store.claim(session_id, "submit")
        blocked = await peer.claim(session_id, "submit")
        self.assertTrue(owner.claimed)
        self.assertFalse(blocked.claimed)
        self.assertEqual(blocked.reason, "in_progress")

        self.now[0] += 11
        takeover = await peer.claim(session_id, "submit")
        self.assertTrue(takeover.claimed)
        self.assertNotEqual(owner.token, takeover.token)
        self.assertIsNone(
            await self.store.checkpoint(
                session_id,
                owner.token,
                _aggregate(session_id, lesson="Stale owner."),
                expected_revision=1,
            )
        )

        saved = await peer.complete(
            session_id,
            takeover.token,
            _aggregate(session_id, lesson="Winning owner."),
            expected_revision=1,
        )
        self.assertEqual(saved.revision, 2)
        self.assertFalse(saved.busy)
        self.assertEqual(
            saved.aggregate.current_artifact.lesson,
            "Winning owner.",
        )

    async def test_concurrent_same_start_key_creates_one_session(self) -> None:
        first = self._new_store()
        second = self._new_store()
        first_id = f"adaptive-{uuid.uuid4().hex}"
        second_id = f"adaptive-{uuid.uuid4().hex}"
        start_key = f"adaptive-start-{uuid.uuid4().hex}"
        request = {
            "user_id": "postgres-user",
            "document_id": "notes.md",
            "goal": "learn binary search",
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
        logical_ids = {decision.record.aggregate.adaptive_session_id for decision in decisions}
        self.assertEqual(len(logical_ids), 1)
        logical_id = logical_ids.pop()
        self.assertIn(logical_id, {first_id, second_id})

    async def test_concurrent_creates_preserve_capacity_limit(self) -> None:
        first = self._new_store(max_count=1)
        second = self._new_store(max_count=1)
        first_id = f"adaptive-{uuid.uuid4().hex}"
        second_id = f"adaptive-{uuid.uuid4().hex}"

        outcomes = await asyncio.gather(
            first.create(_aggregate(first_id)),
            second.create(_aggregate(second_id)),
            return_exceptions=True,
        )

        successes = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
        failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], AdaptiveSessionCapacityError)

    async def test_schema_advisory_lock_times_out_then_initialization_retries(
        self,
    ) -> None:
        import psycopg

        store = self._new_store(
            postgres_lock_timeout_ms=_FAST_LOCK_TIMEOUT_MS,
            postgres_statement_timeout_ms=2_000,
        )
        with self._admin_connect() as blocker:
            blocker.execute(
                "SELECT pg_advisory_xact_lock(%s)",
                (self.schema_lock_id,),
            )
            await self._assert_finishes_with_database_error_while_blocked(
                store.inspect(f"missing-{uuid.uuid4().hex}"),
                psycopg.errors.LockNotAvailable,
                blocker.rollback,
            )

        self.assertIsNone(await store.inspect("missing-after-retry"))

    async def test_session_row_lock_times_out_then_claim_retries(self) -> None:
        import psycopg
        from psycopg import sql

        store = self._new_store(
            postgres_lock_timeout_ms=_FAST_LOCK_TIMEOUT_MS,
            postgres_statement_timeout_ms=2_000,
        )
        session_id = f"adaptive-{uuid.uuid4().hex}"
        await store.create(_aggregate(session_id))
        with self._admin_connect() as blocker:
            row = blocker.execute(
                sql.SQL("SELECT session_id FROM {} WHERE session_id = %s FOR UPDATE").format(
                    sql.Identifier(self.table_name)
                ),
                (session_id,),
            ).fetchone()
            self.assertEqual(row, (session_id,))
            await self._assert_finishes_with_database_error_while_blocked(
                store.claim(session_id, "submit"),
                psycopg.errors.LockNotAvailable,
                blocker.rollback,
            )

        retry = await store.claim(session_id, "submit")
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
            session_id = f"adaptive-{uuid.uuid4().hex}"
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

        task = asyncio.create_task(asyncio.to_thread(execute_slow_query))
        done, _ = await asyncio.wait(
            {task},
            timeout=_TIMEOUT_ASSERTION_SECONDS,
        )
        completed = task in done
        exception = task.exception() if completed else None
        await asyncio.gather(task, return_exceptions=True)

        self.assertTrue(completed, "pg_sleep ignored the configured statement timeout")
        self.assertIsInstance(exception, psycopg.errors.QueryCanceled)
        # 恢复检查换一个正常超时的 store。100ms 是为了让 pg_sleep(1) 必然被
        # 取消才设的；拿它去跑一次真实查询——首次调用还要建表——在慢一点的
        # runner 上会因为建表本身超时而误报成回归。这里要验证的是超时之后
        # store 仍然可用，不是它能在 100ms 内完成建表。
        recovered = self._new_store(postgres_lock_timeout_ms=2_000)
        self.assertIsNone(await recovered.inspect("missing-after-statement-timeout"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
