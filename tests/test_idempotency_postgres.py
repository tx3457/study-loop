"""Live PostgreSQL contract tests for durable request receipts.

Set TEST_DATABASE_URL to run these tests. The default suite skips them so local
SQLite development does not require a database service.
"""

import asyncio
import os
import unittest
import uuid

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
