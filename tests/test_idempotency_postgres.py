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
                return await store.begin(
                    self.key, "agent.autonomous", {"query": "q"}
                )
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
            errors = [
                result for result in decisions if isinstance(result, BaseException)
            ]
            if errors:
                raise errors[0]
            self.assertTrue(all(not decision.replayed for decision in decisions))
        finally:
            with psycopg.connect(TEST_DATABASE_URL) as connection:
                connection.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(
                        sql.Identifier(table_name)
                    )
                )

    async def test_changed_payload_is_rejected_across_connections(self):
        await self.store.begin(
            self.key, "agent.autonomous", {"query": "first"}
        )
        peer = IdempotencyStore(database_url=TEST_DATABASE_URL)

        with self.assertRaises(IdempotencyConflictError) as raised:
            await peer.begin(
                self.key, "agent.autonomous", {"query": "changed"}
            )

        self.assertEqual(raised.exception.reason, "payload_mismatch")

    async def test_ambiguous_effect_state_persists_across_connections(self):
        payload = {"query": "update profile"}
        decision = await self.store.begin(self.key, "agent.autonomous", payload)
        await self.store.mark_effect_started(
            decision.lease, "update_learning_profile"
        )
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
        await self.store.mark_effect_started(
            decision.lease, "update_learning_profile"
        )
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
        fingerprint = idempotency_module._fingerprint(operation, payload)
        legacy_key = f"postgres-{uuid.uuid4().hex}"
        with psycopg.connect(TEST_DATABASE_URL) as connection:
            connection.execute(sql.SQL("""
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
            """).format(sql.Identifier(table_name)))
            connection.execute(sql.SQL("""
                INSERT INTO {} (
                    idempotency_key, operation, request_fingerprint, state,
                    created_at, updated_at
                ) VALUES (%s, %s, %s, 'pending_v2', 1, 1)
            """).format(sql.Identifier(table_name)), (
                legacy_key, operation, fingerprint,
            ))

        first = IdempotencyStore(database_url=TEST_DATABASE_URL)
        second = IdempotencyStore(database_url=TEST_DATABASE_URL)
        try:
            with patch.object(idempotency_module, "_TABLE", table_name):
                results = await asyncio.gather(
                    first.begin(legacy_key, operation, payload),
                    second.begin(legacy_key, operation, payload),
                    return_exceptions=True,
                )
                self.assertTrue(all(
                    isinstance(result, IdempotencyConflictError)
                    and result.reason == "ambiguous"
                    for result in results
                ))
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
                self.assertTrue(
                    {"owner_token", "recovery_token", "lease_expires_at"}
                    <= columns
                )
        finally:
            with psycopg.connect(TEST_DATABASE_URL) as connection:
                connection.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(
                        sql.Identifier(table_name)
                    )
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
