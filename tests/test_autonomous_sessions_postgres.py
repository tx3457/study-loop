"""Live PostgreSQL contract tests for Autonomous HITL pause snapshots.

Set TEST_DATABASE_URL to run these tests.  The default suite skips them so
local SQLite development does not require a database service.
"""

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

    def _new_store(
        self,
        *,
        ttl_seconds: float = 60,
        lease_seconds: float = 10 * 60,
        max_count: int = 10_000,
    ):
        return AutonomousSessionStore(
            database_url=TEST_DATABASE_URL,
            ttl_seconds=ttl_seconds,
            operation_lease_seconds=lease_seconds,
            max_count=max_count,
            clock=lambda: self.now[0],
        )

    async def test_snapshot_reopens_from_a_new_connection(self):
        await self.store.save(self.conversation_id, _payload("reopen"))

        inspection = await self._new_store().inspect(self.conversation_id)

        self.assertEqual(inspection.state, "paused")
        self.assertEqual(inspection.payload, _payload("reopen"))
        claim = await self.store.claim(self.conversation_id, FINGERPRINT)
        self.assertTrue(
            await self.store.consume(self.conversation_id, claim.claim_token)
        )

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
        self.assertTrue(
            await self.store.consume(self.conversation_id, owners[0].claim_token)
        )

    async def test_owner_cas_and_progress_marker_survive_reopen(self):
        await self.store.save(self.conversation_id, _payload("owner"))
        claim = await self.store.claim(self.conversation_id, FINGERPRINT)
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
        self.assertFalse(await first.finish(
            self.conversation_id,
            old.claim_token,
            {"final_answer": "stale"},
        ))
        self.assertTrue(await peer.finish(
            self.conversation_id,
            current.claim_token,
            {"final_answer": "canonical"},
        ))
        reopened = await first.inspect(self.conversation_id)
        self.assertEqual(reopened.state, "completed")
        self.assertEqual(reopened.outcome, {"final_answer": "canonical"})

    async def test_progressed_stale_claim_becomes_ambiguous_across_connections(self):
        first = self._new_store(lease_seconds=10)
        peer = self._new_store(lease_seconds=10)
        await first.save(self.conversation_id, _payload("progressed"))
        old = await first.claim(self.conversation_id, FINGERPRINT)
        self.assertTrue(await first.mark_progress(
            self.conversation_id, old.claim_token
        ))
        self.now[0] += 11

        decision = await peer.claim(self.conversation_id, FINGERPRINT)
        inspection = await first.inspect(self.conversation_id)

        self.assertFalse(decision.claimed)
        self.assertEqual(decision.reason, "ambiguous")
        self.assertEqual(inspection.state, "ambiguous")
        self.assertFalse(await first.finish(
            self.conversation_id,
            old.claim_token,
            {"final_answer": "late"},
        ))

    async def test_concurrent_upgrade_of_legacy_inflight_row_is_fail_closed(self):
        import psycopg
        from psycopg import sql

        table_name = f"studyloop_auto_old_{uuid.uuid4().hex[:16]}"
        legacy_id = f"legacy-{uuid.uuid4().hex}"
        with psycopg.connect(TEST_DATABASE_URL) as connection:
            connection.execute(sql.SQL("""
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
            """).format(sql.Identifier(table_name)))
            connection.execute(sql.SQL("""
                INSERT INTO {} (
                    conversation_id, payload_json, state, claim_token,
                    progress_started, created_at, updated_at, expires_at,
                    claimed_at
                ) VALUES (%s, %s, 'in_flight', 'legacy-owner', 0,
                          900, 900, 2000, 900)
            """).format(sql.Identifier(table_name)), (
                legacy_id, '{"legacy":true}',
            ))

        barrier = threading.Barrier(2)

        class ConcurrentUpgradeStore(AutonomousSessionStore):
            def __init__(self):
                super().__init__(
                    database_url=TEST_DATABASE_URL,
                    operation_lease_seconds=10,
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
            with psycopg.connect(TEST_DATABASE_URL) as connection:
                connection.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(
                        sql.Identifier(table_name)
                    )
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
