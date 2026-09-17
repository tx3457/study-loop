"""Live PostgreSQL contract tests for Autonomous HITL pause snapshots.

Set TEST_DATABASE_URL to run these tests.  The default suite skips them so
local SQLite development does not require a database service.
"""

from __future__ import annotations

import asyncio
import os
import threading
import unittest
import uuid
from unittest.mock import patch

import services.autonomous_sessions as sessions_module
from services.autonomous_sessions import AutonomousSessionStore


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
FINGERPRINT = "a" * 64

_ADMIN_CONNECT_TIMEOUT_SECONDS = 5
_ADMIN_LOCK_TIMEOUT_MS = 2_000
_ADMIN_STATEMENT_TIMEOUT_MS = 5_000
_ADMIN_TCP_USER_TIMEOUT_MS = 30_000
_FAST_LOCK_TIMEOUT_MS = 100
_FAST_STATEMENT_TIMEOUT_MS = 100
_TIMEOUT_ASSERTION_SECONDS = 6.0


def _payload(label: str) -> dict:
    return {
        "schema_version": 1,
        "messages": [{"role": "user", "content": label}],
        "steps": [{"round_index": 0, "tool_name": "ask_user"}],
    }


@unittest.skipUnless(TEST_DATABASE_URL, "TEST_DATABASE_URL is not configured")
class TestPostgresAutonomousSessionStore(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.now = [1_000.0]
        # Keep derived index names below PostgreSQL's 63-byte identifier limit.
        self.table_name = f"sl_auto_{uuid.uuid4().hex[:20]}"
        self.schema_lock_id = (uuid.uuid4().int % ((1 << 63) - 1)) + 1
        self.table_patch = patch.object(
            sessions_module,
            "_TABLE",
            self.table_name,
        )
        self.schema_lock_patch = patch.object(
            sessions_module,
            "_POSTGRES_SCHEMA_LOCK_ID",
            self.schema_lock_id,
        )
        self.table_patch.start()
        self.schema_lock_patch.start()
        self.conversation_id = f"postgres-session-{uuid.uuid4().hex}"
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
        environment_options = os.getenv("PGOPTIONS", "").strip()
        explicit_options = connection_parameters.get("options")
        existing_options = (
            str(explicit_options) if explicit_options is not None else environment_options
        ).strip()
        bounded_options = (
            f"-c lock_timeout={_ADMIN_LOCK_TIMEOUT_MS}ms "
            f"-c statement_timeout={_ADMIN_STATEMENT_TIMEOUT_MS}ms"
        )
        options = " ".join(option for option in (existing_options, bounded_options) if option)
        return psycopg.connect(
            TEST_DATABASE_URL,
            connect_timeout=_ADMIN_CONNECT_TIMEOUT_SECONDS,
            tcp_user_timeout=_ADMIN_TCP_USER_TIMEOUT_MS,
            options=options,
        )

    def _new_store(
        self,
        *,
        ttl_seconds: float = 60,
        lease_seconds: float = 10 * 60,
        max_count: int = 10_000,
        postgres_lock_timeout_ms: int = 5_000,
        postgres_statement_timeout_ms: int = 15_000,
    ) -> AutonomousSessionStore:
        return AutonomousSessionStore(
            database_url=TEST_DATABASE_URL,
            ttl_seconds=ttl_seconds,
            operation_lease_seconds=lease_seconds,
            max_count=max_count,
            postgres_connect_timeout_seconds=5,
            postgres_lock_timeout_ms=postgres_lock_timeout_ms,
            postgres_statement_timeout_ms=postgres_statement_timeout_ms,
            postgres_tcp_user_timeout_ms=30_000,
            clock=lambda: self.now[0],
        )

    async def _assert_finishes_with_database_error_while_blocked(
        self,
        operation,
        expected_exception: type[BaseException],
        release_blocker,
    ) -> None:
        task = asyncio.create_task(operation)
        completed_while_blocked = False
        exception = None
        try:
            done, _ = await asyncio.wait(
                {task},
                timeout=_TIMEOUT_ASSERTION_SECONDS,
            )
            completed_while_blocked = task in done
            exception = task.exception() if completed_while_blocked else None
        finally:
            try:
                release_blocker()
            finally:
                await asyncio.gather(task, return_exceptions=True)

        self.assertTrue(
            completed_while_blocked,
            "database operation ignored its configured timeout",
        )
        self.assertIsInstance(exception, expected_exception)

    async def test_snapshot_reopens_from_a_new_connection(self):
        await self.store.save(self.conversation_id, _payload("reopen"))

        inspection = await self._new_store().inspect(self.conversation_id)

        self.assertEqual(inspection.state, "paused")
        self.assertEqual(inspection.payload, _payload("reopen"))
        claim = await self.store.claim(self.conversation_id, FINGERPRINT)
        self.assertTrue(await self.store.consume(self.conversation_id, claim.claim_token))

    async def test_two_connections_allow_only_one_claim_owner(self):
        peer = self._new_store()
        await self.store.save(self.conversation_id, _payload("race"))

        first, second = await asyncio.gather(
            self.store.claim(self.conversation_id, FINGERPRINT),
            peer.claim(self.conversation_id, FINGERPRINT),
        )

        owners = [decision for decision in (first, second) if decision.claimed]
        rejected = [decision for decision in (first, second) if not decision.claimed]
        self.assertEqual(len(owners), 1)
        self.assertEqual(rejected[0].reason, "in_progress")
        self.assertTrue(await self.store.consume(self.conversation_id, owners[0].claim_token))

    async def test_owner_cas_and_progress_marker_survive_reopen(self):
        await self.store.save(self.conversation_id, _payload("owner"))
        claim = await self.store.claim(self.conversation_id, FINGERPRINT)
        peer = self._new_store()

        self.assertFalse(await peer.release(self.conversation_id, "wrong-token"))
        self.assertTrue(await peer.mark_progress(self.conversation_id, claim.claim_token))
        self.assertFalse(await self.store.release(self.conversation_id, claim.claim_token))
        inspection = await peer.inspect(self.conversation_id)
        self.assertTrue(inspection.progress_started)
        self.assertTrue(await peer.consume(self.conversation_id, claim.claim_token))

    async def test_expired_pause_cannot_be_claimed(self):
        self.store = self._new_store(ttl_seconds=10)
        await self.store.save(self.conversation_id, _payload("expired"))
        self.now[0] += 11

        decision = await self._new_store(ttl_seconds=10).claim(
            self.conversation_id,
            FINGERPRINT,
        )

        self.assertFalse(decision.claimed)
        self.assertEqual(decision.reason, "expired")
        self.assertIsNone(await self.store.inspect(self.conversation_id))

    async def test_handoff_replaces_claim_when_capacity_is_one(self):
        self.store = self._new_store(max_count=1)
        await self.store.save(self.conversation_id, _payload("old"))
        claim = await self.store.claim(self.conversation_id, FINGERPRINT)
        next_id = f"{self.conversation_id}-next"

        self.assertTrue(
            await self.store.handoff(
                self.conversation_id,
                claim.claim_token,
                next_id,
                _payload("next"),
                {"awaiting_user_input": True, "conversation_id": next_id},
            )
        )

        old = await self.store.inspect(self.conversation_id)
        self.assertEqual(old.state, "completed")
        self.assertEqual(
            old.outcome,
            {"awaiting_user_input": True, "conversation_id": next_id},
        )
        next_claim = await self.store.claim(next_id, "b" * 64)
        self.assertEqual(next_claim.payload, _payload("next"))
        self.assertTrue(await self.store.consume(next_id, next_claim.claim_token))

    async def test_concurrent_saves_preserve_postgres_capacity_limit(self):
        first_id = f"{self.conversation_id}-first"
        second_id = f"{self.conversation_id}-second"
        first = self._new_store(max_count=1)
        second = self._new_store(max_count=1)

        await asyncio.gather(
            first.save(first_id, _payload("first")),
            second.save(second_id, _payload("second")),
        )

        states = [await first.status(first_id), await second.status(second_id)]
        self.assertEqual(states.count("paused"), 1)
        self.assertEqual(states.count(None), 1)
        remaining_id = first_id if states[0] == "paused" else second_id
        claim = await first.claim(remaining_id, FINGERPRINT)
        self.assertTrue(await first.consume(remaining_id, claim.claim_token))

    async def test_clean_stale_claim_is_taken_over_and_old_owner_is_fenced(self):
        first = self._new_store(lease_seconds=10)
        peer = self._new_store(lease_seconds=10)
        await first.save(self.conversation_id, _payload("clean takeover"))
        old = await first.claim(self.conversation_id, FINGERPRINT)
        self.now[0] += 11

        current = await peer.claim(self.conversation_id, FINGERPRINT)

        self.assertTrue(current.claimed)
        self.assertNotEqual(old.claim_token, current.claim_token)
        self.assertFalse(await first.renew(self.conversation_id, old.claim_token))
        self.assertFalse(
            await first.finish(
                self.conversation_id,
                old.claim_token,
                {"final_answer": "stale"},
            )
        )
        self.assertTrue(
            await peer.finish(
                self.conversation_id,
                current.claim_token,
                {"final_answer": "canonical"},
            )
        )
        reopened = await first.inspect(self.conversation_id)
        self.assertEqual(reopened.state, "completed")
        self.assertEqual(reopened.outcome, {"final_answer": "canonical"})

    async def test_progressed_stale_claim_becomes_ambiguous_across_connections(self):
        first = self._new_store(lease_seconds=10)
        peer = self._new_store(lease_seconds=10)
        await first.save(self.conversation_id, _payload("progressed"))
        old = await first.claim(self.conversation_id, FINGERPRINT)
        self.assertTrue(await first.mark_progress(self.conversation_id, old.claim_token))
        self.now[0] += 11

        decision = await peer.claim(self.conversation_id, FINGERPRINT)
        inspection = await first.inspect(self.conversation_id)

        self.assertFalse(decision.claimed)
        self.assertEqual(decision.reason, "ambiguous")
        self.assertEqual(inspection.state, "ambiguous")
        self.assertFalse(
            await first.finish(
                self.conversation_id,
                old.claim_token,
                {"final_answer": "late"},
            )
        )

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

        self.assertIsNone(await store.inspect("missing-after-schema-lock-retry"))

    async def test_target_row_lock_times_out_then_claim_retries(self) -> None:
        import psycopg
        from psycopg import sql

        store = self._new_store(
            postgres_lock_timeout_ms=_FAST_LOCK_TIMEOUT_MS,
            postgres_statement_timeout_ms=2_000,
        )
        conversation_id = f"row-lock-{uuid.uuid4().hex}"
        await store.save(conversation_id, _payload("row lock"))

        with self._admin_connect() as blocker:
            locked = blocker.execute(
                sql.SQL(
                    "SELECT conversation_id FROM {} WHERE conversation_id = %s FOR UPDATE"
                ).format(sql.Identifier(self.table_name)),
                (conversation_id,),
            ).fetchone()
            self.assertEqual(locked, (conversation_id,))
            await self._assert_finishes_with_database_error_while_blocked(
                store.claim(conversation_id, FINGERPRINT),
                psycopg.errors.LockNotAvailable,
                blocker.rollback,
            )

        retry = await store.claim(conversation_id, FINGERPRINT)
        self.assertTrue(retry.claimed)
        self.assertTrue(await store.consume(conversation_id, retry.claim_token))

    async def test_capacity_table_lock_times_out_then_save_retries(self) -> None:
        import psycopg
        from psycopg import sql

        store = self._new_store(
            postgres_lock_timeout_ms=_FAST_LOCK_TIMEOUT_MS,
            postgres_statement_timeout_ms=2_000,
        )
        self.assertIsNone(await store.inspect("initialize-schema"))
        conversation_id = f"capacity-lock-{uuid.uuid4().hex}"

        with self._admin_connect() as blocker:
            blocker.execute(
                # Compatible with ordinary INSERT/UPDATE RowExclusive locks,
                # but conflicts with the store's explicit capacity lock.
                sql.SQL("LOCK TABLE {} IN SHARE UPDATE EXCLUSIVE MODE").format(
                    sql.Identifier(self.table_name)
                )
            )
            await self._assert_finishes_with_database_error_while_blocked(
                store.save(conversation_id, _payload("capacity lock")),
                psycopg.errors.LockNotAvailable,
                blocker.rollback,
            )

        await store.save(conversation_id, _payload("capacity lock"))
        retry = await store.inspect(conversation_id)
        self.assertIsNotNone(retry)
        self.assertEqual(retry.payload, _payload("capacity lock"))

    async def test_statement_timeout_cancels_pg_sleep_then_store_retries(
        self,
    ) -> None:
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

    async def test_concurrent_upgrade_of_legacy_inflight_row_is_fail_closed(
        self,
    ) -> None:
        from psycopg import sql

        table_name = f"studyloop_auto_old_{uuid.uuid4().hex[:16]}"
        legacy_id = f"legacy-{uuid.uuid4().hex}"
        with self._admin_connect() as connection:
            connection.execute(
                sql.SQL("""
                CREATE TABLE {} (
                    conversation_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('paused', 'in_flight')),
                    claim_token TEXT,
                    progress_started INTEGER NOT NULL DEFAULT 0,
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL,
                    expires_at DOUBLE PRECISION NOT NULL,
                    claimed_at DOUBLE PRECISION
                )
            """).format(sql.Identifier(table_name))
            )
            connection.execute(
                sql.SQL("""
                INSERT INTO {} (
                    conversation_id, payload_json, state, claim_token,
                    progress_started, created_at, updated_at, expires_at,
                    claimed_at
                ) VALUES (%s, %s, 'in_flight', 'legacy-owner', 0,
                          900, 900, 2000, 900)
            """).format(sql.Identifier(table_name)),
                (
                    legacy_id,
                    '{"legacy":true}',
                ),
            )

        barrier = threading.Barrier(2)

        class ConcurrentUpgradeStore(AutonomousSessionStore):
            def __init__(self):
                super().__init__(
                    database_url=TEST_DATABASE_URL,
                    operation_lease_seconds=10,
                    postgres_connect_timeout_seconds=5,
                    postgres_lock_timeout_ms=5_000,
                    postgres_statement_timeout_ms=15_000,
                    postgres_tcp_user_timeout_ms=30_000,
                    clock=lambda: self_outer.now[0],
                )
                self._first_connect_pending = True

            def _connect(self):
                connection = super()._connect()
                try:
                    if self._first_connect_pending:
                        self._first_connect_pending = False
                        barrier.wait(timeout=10)
                except BaseException:
                    connection.close()
                    raise
                return connection

        self_outer = self
        first = ConcurrentUpgradeStore()
        second = ConcurrentUpgradeStore()
        try:
            with patch.object(sessions_module, "_TABLE", table_name):
                states = await asyncio.gather(
                    first.status(legacy_id),
                    second.status(legacy_id),
                )
                self.assertEqual(states, ["in_flight", "in_flight"])
                self.now[0] += 11
                decision = await first.claim(legacy_id, FINGERPRINT)
                self.assertFalse(decision.claimed)
                self.assertEqual(decision.reason, "ambiguous")
                inspection = await second.inspect(legacy_id)
                self.assertEqual(inspection.state, "ambiguous")
        finally:
            with self._admin_connect() as connection:
                connection.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name))
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
