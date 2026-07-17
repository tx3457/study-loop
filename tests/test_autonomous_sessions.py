"""Durable Autonomous HITL session-store contract tests."""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from services.autonomous_sessions import (
    AutonomousSessionStore,
    SessionAlreadyExistsError,
    SessionCapacityError,
    SessionPayloadTooLargeError,
)


def _payload(label: str) -> dict:
    return {
        "schema_version": 1,
        "messages": [{"role": "user", "content": label}],
        "steps": [{"round_index": 0, "tool_name": "ask_user"}],
        "evidence_registry": {
            "doc_chunk_1": {
                "chunk_id": "doc_chunk_1",
                "document_id": "doc",
                "text": "evidence",
                "rank": 1,
            }
        },
    }


class TestAutonomousSessionStore(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        self.db_path = str(Path(self.temp_dir.name) / "sessions.sqlite3")
        self.now = [1_000.0]

    def tearDown(self):
        self.temp_dir.cleanup()

    def _store(self, *, ttl_seconds: float = 60, max_count: int = 10):
        return AutonomousSessionStore(
            sqlite_path=self.db_path,
            ttl_seconds=ttl_seconds,
            max_count=max_count,
            clock=lambda: self.now[0],
        )

    async def test_round_trips_json_payload_after_store_reopen(self):
        first = self._store()
        await first.save("conv-reopen", _payload("resume me"))

        reopened = self._store()
        inspection = await reopened.inspect("conv-reopen")

        self.assertIsNotNone(inspection)
        self.assertEqual(inspection.state, "paused")
        self.assertEqual(inspection.payload, _payload("resume me"))
        self.assertEqual(inspection.expires_at, self.now[0] + 60)

    async def test_two_store_instances_allow_only_one_atomic_claim(self):
        owner = self._store()
        peer = self._store()
        await owner.save("conv-race", _payload("race"))

        first, second = await asyncio.gather(
            owner.claim("conv-race"), peer.claim("conv-race")
        )

        claimed = [decision for decision in (first, second) if decision.claimed]
        rejected = [decision for decision in (first, second) if not decision.claimed]
        self.assertEqual(len(claimed), 1)
        self.assertEqual(rejected[0].reason, "in_progress")
        self.assertEqual(claimed[0].payload, _payload("race"))
        self.assertTrue(claimed[0].claim_token)

    async def test_in_flight_claim_survives_reopen_and_is_never_auto_released(self):
        first = self._store()
        await first.save("conv-crash", _payload("crash"))
        claim = await first.claim("conv-crash")
        self.assertTrue(claim.claimed)

        reopened = self._store()
        inspection = await reopened.inspect("conv-crash")
        retry = await reopened.claim("conv-crash")

        self.assertEqual(inspection.state, "in_flight")
        self.assertFalse(retry.claimed)
        self.assertEqual(retry.reason, "in_progress")

    async def test_cancelled_claim_releases_a_committed_clean_claim(self):
        store = self._store()
        await store.save("conv-cancelled", _payload("cancelled"))
        committed = threading.Event()
        release_worker = threading.Event()
        original_claim = store._claim_sync

        def pause_after_commit(*args):
            decision = original_claim(*args)
            committed.set()
            release_worker.wait(timeout=5)
            return decision

        with patch.object(store, "_claim_sync", side_effect=pause_after_commit):
            task = asyncio.create_task(store.claim("conv-cancelled"))
            self.assertTrue(await asyncio.to_thread(committed.wait, 5))
            task.cancel()
            release_worker.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        reopened = self._store()
        retry = await reopened.claim("conv-cancelled")
        self.assertTrue(retry.claimed)
        self.assertTrue(await reopened.consume("conv-cancelled", retry.claim_token))

    async def test_only_claim_owner_can_release_or_consume(self):
        store = self._store()
        await store.save("conv-owner", _payload("owner"))
        claim = await store.claim("conv-owner")

        self.assertFalse(await store.release("conv-owner", "wrong-token"))
        self.assertFalse(await store.consume("conv-owner", "wrong-token"))
        self.assertTrue(await store.release("conv-owner", claim.claim_token))

        reclaimed = await store.claim("conv-owner")
        self.assertTrue(reclaimed.claimed)
        self.assertTrue(await store.consume("conv-owner", reclaimed.claim_token))
        self.assertIsNone(await store.inspect("conv-owner"))

    async def test_progress_marker_prevents_even_the_owner_from_releasing(self):
        store = self._store()
        await store.save("conv-progress", _payload("progress"))
        claim = await store.claim("conv-progress")

        self.assertTrue(await store.mark_progress("conv-progress", claim.claim_token))
        self.assertFalse(await store.release("conv-progress", claim.claim_token))
        inspection = await store.inspect("conv-progress")
        self.assertEqual(inspection.state, "in_flight")
        self.assertTrue(inspection.progress_started)
        self.assertTrue(await store.consume("conv-progress", claim.claim_token))

    async def test_handoff_replaces_progressed_claim_at_full_capacity(self):
        store = self._store(max_count=1)
        await store.save("conv-old", _payload("old"))
        claim = await store.claim("conv-old")
        self.assertTrue(await store.mark_progress("conv-old", claim.claim_token))

        self.assertTrue(
            await store.handoff(
                "conv-old",
                claim.claim_token,
                "conv-next",
                _payload("next"),
            )
        )

        self.assertIsNone(await store.status("conv-old"))
        next_session = await store.inspect("conv-next")
        self.assertEqual(next_session.state, "paused")
        self.assertEqual(next_session.payload, _payload("next"))

    async def test_failed_handoff_keeps_the_owned_claim_unchanged(self):
        store = self._store(max_count=2)
        await store.save("conv-old", _payload("old"))
        await store.save("conv-existing", _payload("existing"))
        claim = await store.claim("conv-old")

        self.assertFalse(
            await store.handoff(
                "conv-old", "wrong-token", "conv-next", _payload("next")
            )
        )
        with self.assertRaises(SessionAlreadyExistsError):
            await store.handoff(
                "conv-old",
                claim.claim_token,
                "conv-existing",
                _payload("replacement"),
            )

        self.assertEqual((await store.inspect("conv-old")).state, "in_flight")
        self.assertEqual(
            (await store.inspect("conv-existing")).payload,
            _payload("existing"),
        )
        self.assertIsNone(await store.status("conv-next"))
        self.assertTrue(await store.consume("conv-old", claim.claim_token))

    async def test_expiry_is_enforced_when_claiming_not_only_when_saving(self):
        store = self._store(ttl_seconds=10)
        await store.save("conv-expired", _payload("expired"))
        self.now[0] += 11

        claim = await store.claim("conv-expired")

        self.assertFalse(claim.claimed)
        self.assertEqual(claim.reason, "expired")
        self.assertIsNone(await store.inspect("conv-expired"))

    async def test_capacity_evicts_only_the_oldest_paused_session(self):
        store = self._store(max_count=2)
        await store.save("conv-oldest", _payload("one"))
        self.now[0] += 1
        await store.save("conv-claimed", _payload("two"))
        claimed = await store.claim("conv-claimed")
        self.now[0] += 1
        await store.save("conv-new", _payload("three"))
        self.now[0] += 1
        await store.save("conv-newest", _payload("four"))

        self.assertIsNone(await store.inspect("conv-oldest"))
        self.assertEqual((await store.inspect("conv-claimed")).state, "in_flight")
        self.assertIsNone(await store.inspect("conv-new"))
        self.assertEqual((await store.inspect("conv-newest")).state, "paused")
        self.assertFalse(await store.consume("conv-claimed", "wrong-token"))
        self.assertTrue(await store.consume("conv-claimed", claimed.claim_token))

    async def test_capacity_fails_instead_of_evicting_in_flight_sessions(self):
        store = self._store(max_count=2)
        await store.save("conv-first", _payload("one"))
        await store.save("conv-second", _payload("two"))
        first = await store.claim("conv-first")
        second = await store.claim("conv-second")

        with self.assertRaises(SessionCapacityError):
            await store.save("conv-third", _payload("three"))

        self.assertEqual((await store.inspect("conv-first")).state, "in_flight")
        self.assertEqual((await store.inspect("conv-second")).state, "in_flight")
        self.assertIsNone(await store.inspect("conv-third"))
        self.assertTrue(await store.consume("conv-first", first.claim_token))
        self.assertTrue(await store.consume("conv-second", second.claim_token))

    async def test_duplicate_ids_and_non_json_payloads_fail_before_replacement(self):
        store = self._store()
        await store.save("conv-duplicate", _payload("original"))

        with self.assertRaises(SessionAlreadyExistsError):
            await store.save("conv-duplicate", _payload("replacement"))
        with self.assertRaises(TypeError):
            await store.save("conv-invalid", {"bad": object()})

        inspection = await store.inspect("conv-duplicate")
        self.assertEqual(inspection.payload, _payload("original"))

    async def test_payload_size_limit_fails_before_database_mutation(self):
        store = AutonomousSessionStore(
            sqlite_path=self.db_path,
            ttl_seconds=60,
            max_count=10,
            max_payload_bytes=32,
            clock=lambda: self.now[0],
        )

        with self.assertRaises(SessionPayloadTooLargeError):
            await store.save("conv-large", {"text": "x" * 100})

        self.assertIsNone(await store.inspect("conv-large"))

    async def test_corrupt_json_can_be_claimed_only_for_fail_closed_cleanup(self):
        store = self._store()
        await store.save("conv-corrupt", _payload("valid-before-corruption"))
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                UPDATE studyloop_autonomous_sessions
                SET payload_json = '{'
                WHERE conversation_id = 'conv-corrupt'
                """
            )

        self.assertEqual(await store.status("conv-corrupt"), "paused")
        claim = await store.claim("conv-corrupt")

        self.assertTrue(claim.claimed)
        self.assertEqual(claim.reason, "invalid_payload")
        self.assertIsNone(claim.payload)
        self.assertTrue(await store.consume("conv-corrupt", claim.claim_token))
        self.assertIsNone(await store.status("conv-corrupt"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
