"""Live PostgreSQL contract tests for durable request receipts.

Set TEST_DATABASE_URL to run these tests. The default suite skips them so local
SQLite development does not require a database service.
"""

import asyncio
import os
import threading
import unittest
import uuid
from unittest.mock import patch

import services.idempotency as idempotency_module
from services.idempotency import IdempotencyConflictError, IdempotencyStore


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

_ADMIN_CONNECT_TIMEOUT_SECONDS = 5
_ADMIN_LOCK_TIMEOUT_MS = 2_000
_ADMIN_STATEMENT_TIMEOUT_MS = 5_000
_FAST_LOCK_TIMEOUT_MS = 100
_FAST_STATEMENT_TIMEOUT_MS = 100
_TIMEOUT_ASSERTION_SECONDS = 6.0


@unittest.skipUnless(TEST_DATABASE_URL, "TEST_DATABASE_URL is not configured")
class TestPostgresIdempotencyStore(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = IdempotencyStore(database_url=TEST_DATABASE_URL)
        self.key = f"postgres-{uuid.uuid4().hex}"

    async def test_completed_response_replays_from_a_new_connection(self):
        payload = {"query": "learn RAG"}
        decision = await self.store.begin(self.key, "agent.autonomous", payload)
        self.assertFalse(decision.replayed)
        await self.store.complete(decision.lease, {"final_answer": "done"})

        reopened = IdempotencyStore(database_url=TEST_DATABASE_URL)
        replay = await reopened.begin(self.key, "agent.autonomous", payload)
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.response, {"final_answer": "done"})

    async def test_two_connections_allow_only_one_concurrent_owner(self):
        peer = IdempotencyStore(database_url=TEST_DATABASE_URL)

        async def claim(store):
            try:
                return await store.begin(self.key, "agent.autonomous", {"query": "q"})
            except IdempotencyConflictError as exc:
                return exc.reason

        results = await asyncio.gather(claim(self.store), claim(peer))
        owners = [result for result in results if not isinstance(result, str)]
        conflicts = [result for result in results if isinstance(result, str)]
        self.assertEqual(len(owners), 1)
        self.assertEqual(conflicts, ["in_progress"])

    async def test_two_stores_initialize_a_fresh_schema_concurrently(self):
        import psycopg
        from psycopg import sql

        first_connect = threading.Barrier(2)

        class ConcurrentInitStore(IdempotencyStore):
            def __init__(self):
                super().__init__(database_url=TEST_DATABASE_URL)
                self._first_connect_pending = True

            def _connect(self):
                connection = super()._connect()
                try:
                    if self._first_connect_pending:
                        self._first_connect_pending = False
                        first_connect.wait(timeout=10)
                except BaseException:
                    connection.close()
                    raise
                return connection

        table_name = f"studyloop_idempotency_test_{uuid.uuid4().hex}"
        first = ConcurrentInitStore()
        second = ConcurrentInitStore()

        try:
            with patch.object(idempotency_module, "_TABLE", table_name):
                decisions = await asyncio.gather(
                    first.begin(
                        f"postgres-{uuid.uuid4().hex}",
                        "agent.autonomous",
                        {"query": "first"},
                    ),
                    second.begin(
                        f"postgres-{uuid.uuid4().hex}",
                        "agent.autonomous",
                        {"query": "second"},
                    ),
                    return_exceptions=True,
                )
            errors = [result for result in decisions if isinstance(result, BaseException)]
            if errors:
                raise errors[0]
            self.assertTrue(all(not decision.replayed for decision in decisions))
        finally:
            with psycopg.connect(TEST_DATABASE_URL) as connection:
                connection.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name))
                )

    async def test_changed_payload_is_rejected_across_connections(self):
        await self.store.begin(self.key, "agent.autonomous", {"query": "first"})
        peer = IdempotencyStore(database_url=TEST_DATABASE_URL)

        with self.assertRaises(IdempotencyConflictError) as raised:
            await peer.begin(self.key, "agent.autonomous", {"query": "changed"})

        self.assertEqual(raised.exception.reason, "payload_mismatch")

    async def test_ambiguous_effect_state_persists_across_connections(self):
        payload = {"query": "update profile"}
        decision = await self.store.begin(self.key, "agent.autonomous", payload)
        await self.store.mark_effect_started(decision.lease, "update_learning_profile")
        self.assertTrue(await self.store.abort(decision.lease))

        reopened = IdempotencyStore(database_url=TEST_DATABASE_URL)
        with self.assertRaises(IdempotencyConflictError) as raised:
            await reopened.begin(self.key, "agent.autonomous", payload)

        self.assertEqual(raised.exception.reason, "ambiguous")

    async def test_expired_clean_owner_is_taken_over_and_stale_token_is_fenced(self):
        now = [100.0]
        first = IdempotencyStore(
            database_url=TEST_DATABASE_URL,
            lease_seconds=5,
            clock=lambda: now[0],
        )
        peer = IdempotencyStore(
            database_url=TEST_DATABASE_URL,
            lease_seconds=5,
            clock=lambda: now[0],
        )
        payload = {"query": "lease takeover"}
        old = await first.begin(self.key, "agent.autonomous", payload)
        now[0] = 106.0

        current = await peer.begin(self.key, "agent.autonomous", payload)

        self.assertNotEqual(old.lease.owner_token, current.lease.owner_token)
        self.assertEqual(old.lease.recovery_token, current.lease.recovery_token)
        self.assertIsNone(await first.renew(old.lease))
        with self.assertRaises(IdempotencyConflictError):
            await first.complete(old.lease, {"final_answer": "stale"})
        with self.assertRaises(IdempotencyConflictError):
            await first.mark_effect_started(old.lease, "stale_tool")
        self.assertFalse(await first.abort(old.lease))

        await peer.complete(current.lease, {"final_answer": "canonical"})
        replay = await first.begin(self.key, "agent.autonomous", payload)
        self.assertEqual(replay.response, {"final_answer": "canonical"})

    async def test_effect_receipt_reconciles_from_canonical_outcome_across_connections(self):
        payload = {"conversation_id": "c", "user_reply": "继续"}
        operation = "agent.autonomous.continue"
        decision = await self.store.begin(self.key, operation, payload)
        await self.store.mark_effect_started(decision.lease, "update_learning_profile")
        peer = IdempotencyStore(database_url=TEST_DATABASE_URL)
        response = {"final_answer": "canonical", "rounds_used": 2}

        await peer.reconcile_completed(
            self.key,
            operation,
            payload,
            response,
            allow_effect_started=True,
        )

        replay = await self.store.begin(self.key, operation, payload)
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.response, response)

    async def test_concurrent_upgrade_of_legacy_v2_row_fails_closed(self):
        import psycopg
        from psycopg import sql

        table_name = f"studyloop_idem_old_{uuid.uuid4().hex[:16]}"
        operation = "agent.autonomous"
        payload = {"query": "legacy pending"}
        fingerprint = idempotency_module.request_fingerprint(operation, payload)
        legacy_key = f"postgres-{uuid.uuid4().hex}"
        with psycopg.connect(TEST_DATABASE_URL) as connection:
            connection.execute(
                sql.SQL("""
                CREATE TABLE {} (
                    idempotency_key TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    state TEXT NOT NULL,
                    response_json TEXT,
                    effect_tool TEXT,
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL
                )
            """).format(sql.Identifier(table_name))
            )
            connection.execute(
                sql.SQL("""
                INSERT INTO {} (
                    idempotency_key, operation, request_fingerprint, state,
                    created_at, updated_at
                ) VALUES (%s, %s, %s, 'pending_v2', 1, 1)
            """).format(sql.Identifier(table_name)),
                (
                    legacy_key,
                    operation,
                    fingerprint,
                ),
            )

        first = IdempotencyStore(database_url=TEST_DATABASE_URL)
        second = IdempotencyStore(database_url=TEST_DATABASE_URL)
        try:
            with patch.object(idempotency_module, "_TABLE", table_name):
                results = await asyncio.gather(
                    first.begin(legacy_key, operation, payload),
                    second.begin(legacy_key, operation, payload),
                    return_exceptions=True,
                )
                self.assertTrue(
                    all(
                        isinstance(result, IdempotencyConflictError)
                        and result.reason == "ambiguous"
                        for result in results
                    )
                )
                with psycopg.connect(TEST_DATABASE_URL) as connection:
                    columns = {
                        row[0]
                        for row in connection.execute(
                            """
                            SELECT column_name FROM information_schema.columns
                            WHERE table_name = %s
                            """,
                            (table_name,),
                        )
                    }
                self.assertTrue({"owner_token", "recovery_token", "lease_expires_at"} <= columns)
        finally:
            with psycopg.connect(TEST_DATABASE_URL) as connection:
                connection.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name))
                )


@unittest.skipUnless(TEST_DATABASE_URL, "TEST_DATABASE_URL is not configured")
class TestPostgresIdempotencyRuntimeBounds(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.table_name = f"studyloop_idem_bound_{uuid.uuid4().hex}"
        self.schema_lock_id = uuid.uuid4().int & ((1 << 63) - 1)
        self.table_patch = patch.object(
            idempotency_module,
            "_TABLE",
            self.table_name,
        )
        self.schema_lock_patch = patch.object(
            idempotency_module,
            "_POSTGRES_SCHEMA_LOCK_ID",
            self.schema_lock_id,
        )
        self.table_patch.start()
        self.schema_lock_patch.start()

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

    @staticmethod
    def _new_store(
        *,
        postgres_lock_timeout_ms: int = 5_000,
        postgres_statement_timeout_ms: int = 15_000,
    ) -> IdempotencyStore:
        return IdempotencyStore(
            database_url=TEST_DATABASE_URL,
            postgres_connect_timeout_seconds=5,
            postgres_lock_timeout_ms=postgres_lock_timeout_ms,
            postgres_statement_timeout_ms=postgres_statement_timeout_ms,
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
            await asyncio.gather(task, return_exceptions=True)

        self.assertTrue(done, "database operation ignored its configured timeout")
        self.assertIsInstance(exception, expected_exception)

    async def test_schema_advisory_lock_times_out_then_initialization_retries(self):
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
                store.begin(
                    f"postgres-{uuid.uuid4().hex}",
                    "agent.autonomous",
                    {"query": "blocked schema"},
                ),
                psycopg.errors.LockNotAvailable,
                blocker.rollback,
            )

        decision = await store.begin(
            f"postgres-{uuid.uuid4().hex}",
            "agent.autonomous",
            {"query": "schema retry"},
        )
        self.assertFalse(decision.replayed)

    async def test_receipt_row_lock_times_out_then_renew_retries(self):
        import psycopg
        from psycopg import sql

        store = self._new_store(
            postgres_lock_timeout_ms=_FAST_LOCK_TIMEOUT_MS,
            postgres_statement_timeout_ms=2_000,
        )
        key = f"postgres-{uuid.uuid4().hex}"
        decision = await store.begin(
            key,
            "agent.autonomous",
            {"query": "locked receipt"},
        )
        with self._admin_connect() as blocker:
            row = blocker.execute(
                sql.SQL(
                    "SELECT idempotency_key FROM {} WHERE idempotency_key = %s FOR UPDATE"
                ).format(sql.Identifier(self.table_name)),
                (key,),
            ).fetchone()
            self.assertEqual(row, (key,))
            await self._assert_finishes_with_database_error_while_blocked(
                store.renew(decision.lease),
                psycopg.errors.LockNotAvailable,
                blocker.rollback,
            )

        renewed = await store.renew(decision.lease)
        self.assertIsNotNone(renewed)

    async def test_statement_timeout_cancels_pg_sleep_then_store_retries(self):
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
        exception = task.exception() if done else None
        if not done:
            await asyncio.gather(task, return_exceptions=True)

        self.assertTrue(done, "pg_sleep ignored the configured statement timeout")
        self.assertIsInstance(exception, psycopg.errors.QueryCanceled)

        decision = await store.begin(
            f"postgres-{uuid.uuid4().hex}",
            "agent.autonomous",
            {"query": "statement retry"},
        )
        self.assertFalse(decision.replayed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
