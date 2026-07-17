"""Live PostgreSQL contract tests for Autonomous HITL pause snapshots.

Set TEST_DATABASE_URL to run these tests.  The default suite skips them so
local SQLite development does not require a database service.
"""

import asyncio
import os
import unittest
import uuid

from services.autonomous_sessions import AutonomousSessionStore


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


def _payload(label: str) -> dict:
    return {
        "schema_version": 1,
        "messages": [{"role": "user", "content": label}],
        "steps": [{"round_index": 0, "tool_name": "ask_user"}],
    }


@unittest.skipUnless(TEST_DATABASE_URL, "TEST_DATABASE_URL is not configured")
class TestPostgresAutonomousSessionStore(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = [1_000.0]
        self.conversation_id = f"postgres-session-{uuid.uuid4().hex}"
        self.store = self._new_store()

    def _new_store(self, *, ttl_seconds: float = 60, max_count: int = 10_000):
        return AutonomousSessionStore(
            database_url=TEST_DATABASE_URL,
            ttl_seconds=ttl_seconds,
            max_count=max_count,
            clock=lambda: self.now[0],
        )

    async def test_snapshot_reopens_from_a_new_connection(self):
        await self.store.save(self.conversation_id, _payload("reopen"))

        inspection = await self._new_store().inspect(self.conversation_id)

        self.assertEqual(inspection.state, "paused")
        self.assertEqual(inspection.payload, _payload("reopen"))
        claim = await self.store.claim(self.conversation_id)
        self.assertTrue(
            await self.store.consume(self.conversation_id, claim.claim_token)
        )

    async def test_two_connections_allow_only_one_claim_owner(self):
        peer = self._new_store()
        await self.store.save(self.conversation_id, _payload("race"))

        first, second = await asyncio.gather(
            self.store.claim(self.conversation_id),
            peer.claim(self.conversation_id),
        )

        owners = [decision for decision in (first, second) if decision.claimed]
        rejected = [decision for decision in (first, second) if not decision.claimed]
        self.assertEqual(len(owners), 1)
        self.assertEqual(rejected[0].reason, "in_progress")
        self.assertTrue(
            await self.store.consume(self.conversation_id, owners[0].claim_token)
        )

    async def test_owner_cas_and_progress_marker_survive_reopen(self):
        await self.store.save(self.conversation_id, _payload("owner"))
        claim = await self.store.claim(self.conversation_id)
        peer = self._new_store()

        self.assertFalse(await peer.release(self.conversation_id, "wrong-token"))
        self.assertTrue(
            await peer.mark_progress(self.conversation_id, claim.claim_token)
        )
        self.assertFalse(
            await self.store.release(self.conversation_id, claim.claim_token)
        )
        inspection = await peer.inspect(self.conversation_id)
        self.assertTrue(inspection.progress_started)
        self.assertTrue(await peer.consume(self.conversation_id, claim.claim_token))

    async def test_expired_pause_cannot_be_claimed(self):
        self.store = self._new_store(ttl_seconds=10)
        await self.store.save(self.conversation_id, _payload("expired"))
        self.now[0] += 11

        decision = await self._new_store(ttl_seconds=10).claim(self.conversation_id)

        self.assertFalse(decision.claimed)
        self.assertEqual(decision.reason, "expired")
        self.assertIsNone(await self.store.inspect(self.conversation_id))

    async def test_handoff_replaces_claim_when_capacity_is_one(self):
        self.store = self._new_store(max_count=1)
        await self.store.save(self.conversation_id, _payload("old"))
        claim = await self.store.claim(self.conversation_id)
        next_id = f"{self.conversation_id}-next"

        self.assertTrue(
            await self.store.handoff(
                self.conversation_id,
                claim.claim_token,
                next_id,
                _payload("next"),
            )
        )

        self.assertIsNone(await self.store.status(self.conversation_id))
        next_claim = await self.store.claim(next_id)
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
        claim = await first.claim(remaining_id)
        self.assertTrue(await first.consume(remaining_id, claim.claim_token))


if __name__ == "__main__":
    unittest.main(verbosity=2)
