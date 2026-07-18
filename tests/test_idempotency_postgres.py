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
        await self.store.complete(self.key, {"final_answer": "done"})

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
        await self.store.begin(self.key, "agent.autonomous", payload)
        await self.store.mark_effect_started(
            self.key, "update_learning_profile"
        )
        self.assertTrue(await self.store.abort(self.key))

        reopened = IdempotencyStore(database_url=TEST_DATABASE_URL)
        with self.assertRaises(IdempotencyConflictError) as raised:
            await reopened.begin(self.key, "agent.autonomous", payload)

        self.assertEqual(raised.exception.reason, "ambiguous")


if __name__ == "__main__":
    unittest.main(verbosity=2)
