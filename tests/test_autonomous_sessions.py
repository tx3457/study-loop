"""Durable Autonomous HITL session-store contract tests."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import sys
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

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


def _fingerprint(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _response(label: str, *, conversation_id: str | None = None) -> dict:
    value = {"response_schema_version": 2, "final_answer": label}
    if conversation_id is not None:
        value.update(
            {
                "final_answer": "",
                "awaiting_user_input": True,
                "user_question": label,
                "conversation_id": conversation_id,
            }
        )
    return value


class TestAutonomousSessionStore(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "sessions.sqlite3")
        self.now = [1_000.0]

    def tearDown(self):
        self.temp_dir.cleanup()

    def _store(
        self,
        *,
        ttl_seconds: float = 60,
        lease_seconds: float = 10,
        max_count: int = 10,
        max_payload_bytes: int = 2 * 1024 * 1024,
        cancel_drain_timeout_seconds: float = 0.05,
    ):
        return AutonomousSessionStore(
            sqlite_path=self.db_path,
            ttl_seconds=ttl_seconds,
            operation_lease_seconds=lease_seconds,
            max_count=max_count,
            max_payload_bytes=max_payload_bytes,
            cancel_drain_timeout_seconds=cancel_drain_timeout_seconds,
            clock=lambda: self.now[0],
        )

    async def _save_and_claim(self, conversation_id: str, request: str = "reply"):
        store = self._store()
        await store.save(conversation_id, _payload(conversation_id))
        claim = await store.claim(conversation_id, _fingerprint(request))
        self.assertTrue(claim.claimed)
        return store, claim

    async def _raw_state(self, conversation_id: str):
        def read():
            with closing(sqlite3.connect(self.db_path)) as connection:
                return connection.execute(
                    "SELECT state, outcome_json FROM studyloop_autonomous_sessions "
                    "WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()

        return await asyncio.to_thread(read)

    async def test_round_trips_json_payload_after_store_reopen(self):
        await self._store().save("conv-reopen", _payload("resume me"))

        inspection = await self._store().inspect("conv-reopen")

        self.assertIsNotNone(inspection)
        self.assertEqual(inspection.state, "paused")
        self.assertEqual(inspection.payload, _payload("resume me"))
        self.assertEqual(inspection.expires_at, self.now[0] + 60)
        self.assertIsNone(inspection.outcome)

    async def test_two_store_instances_allow_only_one_atomic_claim(self):
        owner = self._store()
        peer = self._store()
        await owner.save("conv-race", _payload("race"))
        fingerprint = _fingerprint("same request")

        first, second = await asyncio.gather(
            owner.claim("conv-race", fingerprint),
            peer.claim("conv-race", fingerprint),
        )

        claimed = [decision for decision in (first, second) if decision.claimed]
        rejected = [decision for decision in (first, second) if not decision.claimed]
        self.assertEqual(len(claimed), 1)
        self.assertEqual(rejected[0].reason, "in_progress")
        self.assertEqual(claimed[0].payload, _payload("race"))
        self.assertTrue(claimed[0].claim_token)

    async def test_clean_stale_lease_is_reclaimed_and_old_token_is_fenced(self):
        first, old_claim = await self._save_and_claim("conv-crash")
        self.now[0] += 11

        reopened = self._store()
        new_claim = await reopened.claim("conv-crash", _fingerprint("reply"))

        self.assertTrue(new_claim.claimed)
        self.assertNotEqual(new_claim.claim_token, old_claim.claim_token)
        self.assertFalse(await first.renew("conv-crash", old_claim.claim_token))
        self.assertFalse(await first.mark_progress("conv-crash", old_claim.claim_token))
        self.assertFalse(await first.cancel("conv-crash", old_claim.claim_token))
        self.assertFalse(await first.finish("conv-crash", old_claim.claim_token, _response("late")))
        self.assertFalse(
            await first.handoff(
                "conv-crash",
                old_claim.claim_token,
                "conv-late-next",
                _payload("late"),
                _response("late", conversation_id="conv-late-next"),
            )
        )
        self.assertFalse(await first.consume("conv-crash", old_claim.claim_token))
        self.assertTrue(await reopened.consume("conv-crash", new_claim.claim_token))

    async def test_stale_clean_claim_rejects_a_different_request_fingerprint(self):
        store = self._store()
        await store.save("conv-bound", _payload("bound"))
        first = await store.claim("conv-bound", _fingerprint("first"))
        self.now[0] += 11

        mismatch = await self._store().claim("conv-bound", _fingerprint("changed"))
        inspected = await self._store().inspect("conv-bound")
        mismatch_after_reap = await self._store().claim("conv-bound", _fingerprint("changed"))
        replay = await self._store().claim("conv-bound", _fingerprint("first"))

        self.assertFalse(mismatch.claimed)
        self.assertEqual(mismatch.reason, "payload_mismatch")
        self.assertEqual(inspected.state, "paused")
        self.assertEqual(inspected.continue_fingerprint, _fingerprint("first"))
        self.assertEqual(mismatch_after_reap.reason, "payload_mismatch")
        self.assertTrue(replay.claimed)
        self.assertNotEqual(first.claim_token, replay.claim_token)

    async def test_progressed_stale_lease_becomes_durable_ambiguous_tombstone(self):
        store, claim = await self._save_and_claim("conv-progress")
        self.assertTrue(await store.mark_progress("conv-progress", claim.claim_token))
        self.now[0] += 11

        # A mismatched retry is observational only: it cannot poison the row
        # before the exact original request performs the stale transition.
        mismatch = await self._store().claim("conv-progress", _fingerprint("changed"))
        before_exact = await self._raw_state("conv-progress")
        retry = await self._store().claim("conv-progress", _fingerprint("reply"))
        inspection = await self._store().inspect("conv-progress")
        reopened = await self._store().claim("conv-progress", _fingerprint("reply"))

        self.assertEqual(mismatch.reason, "payload_mismatch")
        self.assertEqual(before_exact, ("in_flight", None))
        self.assertFalse(retry.claimed)
        self.assertEqual(retry.reason, "ambiguous")
        self.assertEqual(inspection.state, "ambiguous")
        self.assertIsNone(inspection.outcome)
        ambiguous_raw = await self._raw_state("conv-progress")
        self.assertEqual(ambiguous_raw[0], "in_flight")
        self.assertIsNotNone(ambiguous_raw[1])
        self.assertEqual(reopened.reason, "ambiguous")
        self.assertFalse(await store.finish("conv-progress", claim.claim_token, _response("late")))
        self.assertFalse(await store.consume("conv-progress", claim.claim_token))

    async def test_renew_extends_live_lease_but_cannot_revive_expired_owner(self):
        store, claim = await self._save_and_claim("conv-renew")
        self.now[0] += 6
        self.assertTrue(await store.renew("conv-renew", claim.claim_token))
        renewed = await store.inspect("conv-renew")
        self.assertEqual(renewed.claim_expires_at, 1_016.0)

        self.now[0] += 5
        blocked = await self._store().claim("conv-renew", _fingerprint("reply"))
        self.assertEqual(blocked.reason, "in_progress")

        self.now[0] += 6
        self.assertFalse(await store.renew("conv-renew", claim.claim_token))
        takeover = await self._store().claim("conv-renew", _fingerprint("reply"))
        self.assertTrue(takeover.claimed)

    async def test_cancel_releases_only_a_live_clean_claim(self):
        store, claim = await self._save_and_claim("conv-cancel")
        self.assertFalse(await store.cancel("conv-cancel", "wrong-token"))
        self.assertTrue(await store.cancel("conv-cancel", claim.claim_token))
        self.assertEqual((await store.inspect("conv-cancel")).state, "paused")

        changed = await store.claim("conv-cancel", _fingerprint("changed reply"))
        self.assertTrue(changed.claimed)
        self.assertTrue(await store.mark_progress("conv-cancel", changed.claim_token))
        self.assertFalse(await store.cancel("conv-cancel", changed.claim_token))

    async def test_discard_paused_distinguishes_all_public_states(self):
        store = self._store()
        await store.save("conv-paused", _payload("paused"))
        canceled = await store.discard_paused("conv-paused")
        missing = await store.discard_paused("conv-missing")

        await store.save("conv-running", _payload("running"))
        running = await store.claim("conv-running", _fingerprint("running"))
        in_progress = await store.discard_paused("conv-running")
        self.assertTrue(await store.consume("conv-running", running.claim_token))

        await store.save("conv-done", _payload("done"))
        done = await store.claim("conv-done", _fingerprint("done"))
        self.assertTrue(await store.finish("conv-done", done.claim_token, _response("finished")))
        completed = await store.discard_paused("conv-done")

        await store.save("conv-ambiguous", _payload("ambiguous"))
        ambiguous_claim = await store.claim("conv-ambiguous", _fingerprint("ambiguous"))
        self.assertTrue(await store.mark_progress("conv-ambiguous", ambiguous_claim.claim_token))
        self.now[0] += 11
        self.assertEqual(
            (await store.claim("conv-ambiguous", _fingerprint("ambiguous"))).reason,
            "ambiguous",
        )
        ambiguous = await store.discard_paused("conv-ambiguous")

        self.assertTrue(canceled.discarded)
        self.assertEqual(canceled.reason, "canceled")
        self.assertEqual(missing.reason, "missing")
        self.assertEqual(in_progress.reason, "in_progress")
        self.assertEqual(completed.reason, "completed")
        self.assertEqual(ambiguous.reason, "ambiguous")

    async def test_finish_persists_exact_outcome_for_reopen_and_binding(self):
        store, claim = await self._save_and_claim("conv-finish")
        response = _response("canonical answer")
        self.assertTrue(await store.mark_progress("conv-finish", claim.claim_token))
        self.assertTrue(await store.finish("conv-finish", claim.claim_token, response))

        inspection = await self._store().inspect("conv-finish")
        replay = await self._store().claim("conv-finish", _fingerprint("reply"))
        mismatch = await self._store().claim("conv-finish", _fingerprint("different"))

        self.assertEqual(inspection.state, "completed")
        self.assertEqual(inspection.outcome, response)
        self.assertFalse(replay.claimed)
        self.assertEqual(replay.reason, "completed")
        self.assertEqual(replay.outcome, response)
        self.assertEqual(mismatch.reason, "payload_mismatch")
        raw = await self._raw_state("conv-finish")
        self.assertEqual(raw[0], "in_flight")
        self.assertIsNotNone(raw[1])
        self.assertFalse(
            await store.finish("conv-finish", claim.claim_token, _response("overwrite"))
        )

    async def test_schema_recheck_does_not_extend_outcome_tombstone_claim(self):
        store, claim = await self._save_and_claim("conv-finished-migration")
        self.assertTrue(
            await store.finish("conv-finished-migration", claim.claim_token, _response("done"))
        )

        reopened = self._store()
        self.assertEqual(
            (await reopened.inspect("conv-finished-migration")).state,
            "completed",
        )

        def read_claim_expiry():
            with closing(sqlite3.connect(self.db_path)) as connection:
                return connection.execute(
                    "SELECT claim_expires_at FROM studyloop_autonomous_sessions "
                    "WHERE conversation_id = ?",
                    ("conv-finished-migration",),
                ).fetchone()[0]

        self.assertIsNone(await asyncio.to_thread(read_claim_expiry))

    async def test_handoff_atomically_keeps_old_outcome_and_new_pause(self):
        store = self._store(max_count=1)
        await store.save("conv-old", _payload("old"))
        fingerprint = _fingerprint("reply")
        claim = await store.claim("conv-old", fingerprint)
        self.assertTrue(await store.mark_progress("conv-old", claim.claim_token))
        response = _response("next question", conversation_id="conv-next")

        self.assertTrue(
            await store.handoff(
                "conv-old",
                claim.claim_token,
                "conv-next",
                _payload("next"),
                response,
            )
        )

        reopened = self._store(max_count=1)
        old = await reopened.inspect("conv-old")
        next_session = await reopened.inspect("conv-next")
        replay = await reopened.claim("conv-old", fingerprint)
        self.assertEqual(old.state, "completed")
        self.assertEqual(old.outcome, response)
        self.assertEqual(next_session.state, "paused")
        self.assertEqual(next_session.payload, _payload("next"))
        self.assertEqual(replay.reason, "completed")
        self.assertEqual(replay.outcome, response)

    async def test_outcome_tombstones_cannot_be_claimed_by_legacy_sql(self):
        store, finished_claim = await self._save_and_claim("conv-old-finished")
        self.assertTrue(
            await store.finish(
                "conv-old-finished",
                finished_claim.claim_token,
                _response("finished"),
            )
        )

        progressed_store, progressed_claim = await self._save_and_claim("conv-old-ambiguous")
        self.assertTrue(
            await progressed_store.mark_progress("conv-old-ambiguous", progressed_claim.claim_token)
        )
        self.now[0] += 11
        self.assertEqual(
            (await progressed_store.claim("conv-old-ambiguous", _fingerprint("reply"))).reason,
            "ambiguous",
        )

        def legacy_claim(conversation_id: str) -> int:
            with closing(sqlite3.connect(self.db_path)) as connection:
                cursor = connection.execute(
                    """
                    UPDATE studyloop_autonomous_sessions
                    SET state = 'in_flight', claim_token = 'legacy-owner',
                        claimed_at = 9999, updated_at = 9999,
                        progress_started = 0
                    WHERE conversation_id = ? AND state = 'paused'
                      AND expires_at > 1000
                    """,
                    (conversation_id,),
                )
                connection.commit()
                return cursor.rowcount

        self.assertEqual(await asyncio.to_thread(legacy_claim, "conv-old-finished"), 0)
        self.assertEqual(await asyncio.to_thread(legacy_claim, "conv-old-ambiguous"), 0)

    async def test_failed_handoff_rolls_back_the_predecessor_outcome(self):
        store = self._store(max_count=2)
        await store.save("conv-old", _payload("old"))
        await store.save("conv-existing", _payload("existing"))
        claim = await store.claim("conv-old", _fingerprint("reply"))

        self.assertFalse(
            await store.handoff(
                "conv-old",
                "wrong-token",
                "conv-next",
                _payload("next"),
                _response("next", conversation_id="conv-next"),
            )
        )
        with self.assertRaises(SessionAlreadyExistsError):
            await store.handoff(
                "conv-old",
                claim.claim_token,
                "conv-existing",
                _payload("replacement"),
                _response("existing", conversation_id="conv-existing"),
            )

        self.assertEqual((await store.inspect("conv-old")).state, "in_flight")
        self.assertEqual((await store.inspect("conv-existing")).payload, _payload("existing"))
        self.assertIsNone(await store.status("conv-next"))
        self.assertTrue(await store.finish("conv-old", claim.claim_token, _response("done")))

    async def test_expiry_is_enforced_when_claiming(self):
        store = self._store(ttl_seconds=10)
        await store.save("conv-expired", _payload("expired"))
        self.now[0] += 11

        claim = await store.claim("conv-expired", _fingerprint("reply"))

        self.assertFalse(claim.claimed)
        self.assertEqual(claim.reason, "expired")
        self.assertIsNone(await store.inspect("conv-expired"))

    async def test_completed_tombstone_expires_without_consuming_capacity(self):
        store = self._store(ttl_seconds=10, max_count=1)
        await store.save("conv-done", _payload("done"))
        claim = await store.claim("conv-done", _fingerprint("reply"))
        self.assertTrue(await store.finish("conv-done", claim.claim_token, _response("done")))
        await store.save("conv-active", _payload("active"))
        self.assertEqual((await store.inspect("conv-active")).state, "paused")

        self.now[0] += 11
        self.assertIsNone(await store.inspect("conv-done"))

    async def test_capacity_evicts_paused_but_not_a_live_owner(self):
        store = self._store(max_count=2)
        await store.save("conv-first", _payload("one"))
        await store.save("conv-second", _payload("two"))
        first = await store.claim("conv-first", _fingerprint("one"))
        second = await store.claim("conv-second", _fingerprint("two"))

        with self.assertRaises(SessionCapacityError):
            await store.save("conv-third", _payload("three"))

        self.assertEqual((await store.inspect("conv-first")).state, "in_flight")
        self.assertEqual((await store.inspect("conv-second")).state, "in_flight")
        self.assertTrue(await store.consume("conv-first", first.claim_token))
        self.assertTrue(await store.consume("conv-second", second.claim_token))

    async def test_stale_claims_no_longer_exhaust_capacity(self):
        store = self._store(max_count=2)
        await store.save("conv-clean", _payload("clean"))
        await store.save("conv-progress", _payload("progress"))
        clean = await store.claim("conv-clean", _fingerprint("clean"))
        progressed = await store.claim("conv-progress", _fingerprint("progress"))
        self.assertTrue(await store.mark_progress("conv-progress", progressed.claim_token))
        self.now[0] += 11

        await store.save("conv-new", _payload("new"))
        await store.save("conv-newest", _payload("newest"))

        self.assertFalse(await store.renew("conv-clean", clean.claim_token))
        self.assertEqual((await store.inspect("conv-progress")).state, "ambiguous")
        self.assertEqual((await store.inspect("conv-newest")).state, "paused")

    async def test_duplicate_invalid_and_oversized_payloads_fail_safely(self):
        store = self._store()
        await store.save("conv-duplicate", _payload("original"))

        with self.assertRaises(SessionAlreadyExistsError):
            await store.save("conv-duplicate", _payload("replacement"))
        with self.assertRaises(TypeError):
            await store.save("conv-invalid", {"bad": object()})
        with self.assertRaises(SessionPayloadTooLargeError):
            await self._store(max_payload_bytes=32).save("conv-large", {"text": "x" * 100})
        with self.assertRaises(SessionPayloadTooLargeError):
            claim = await store.claim("conv-duplicate", _fingerprint("reply"))
            await self._store(max_payload_bytes=32).finish(
                "conv-duplicate", claim.claim_token, {"text": "x" * 100}
            )

        self.assertEqual((await store.inspect("conv-duplicate")).payload, _payload("original"))
        self.assertIsNone(await store.inspect("conv-large"))

    async def test_corrupt_json_can_only_be_claimed_for_fenced_cleanup(self):
        store = self._store()
        await store.save("conv-corrupt", _payload("valid-before-corruption"))
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "UPDATE studyloop_autonomous_sessions SET payload_json = '{' "
                "WHERE conversation_id = 'conv-corrupt'"
            )
            connection.commit()

        claim = await store.claim("conv-corrupt", _fingerprint("reply"))

        self.assertTrue(claim.claimed)
        self.assertEqual(claim.reason, "invalid_payload")
        self.assertIsNone(claim.payload)
        self.assertTrue(await store.consume("conv-corrupt", claim.claim_token))
        self.assertIsNone(await store.status("conv-corrupt"))

    async def test_cancelled_claim_releases_a_committed_clean_claim(self):
        store = self._store(cancel_drain_timeout_seconds=1)
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
            task = asyncio.create_task(store.claim("conv-cancelled", _fingerprint("reply")))
            self.assertTrue(await asyncio.to_thread(committed.wait, 5))
            task.cancel()
            release_worker.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        retry = await self._store().claim("conv-cancelled", _fingerprint("reply"))
        self.assertTrue(retry.claimed)

    async def test_cancelled_db_wait_is_bounded_and_finishes_in_background(self):
        store = self._store(cancel_drain_timeout_seconds=0.01)
        await store.save("conv-drain", _payload("drain"))
        started = threading.Event()
        release_worker = threading.Event()
        original_inspect = store._inspect_sync

        def blocked_inspect(*args):
            started.set()
            release_worker.wait(timeout=5)
            return original_inspect(*args)

        with patch.object(store, "_inspect_sync", side_effect=blocked_inspect):
            task = asyncio.create_task(store.inspect("conv-drain"))
            self.assertTrue(await asyncio.to_thread(started.wait, 5))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=0.5)
            self.assertEqual(len(store._background_workers), 1)
            release_worker.set()
            for _ in range(100):
                if not store._background_workers:
                    break
                await asyncio.sleep(0.01)
            self.assertFalse(store._background_workers)

    async def test_clock_is_read_after_sqlite_write_lock_is_acquired(self):
        store = self._store(ttl_seconds=10)
        await store.save("conv-lock-clock", _payload("clock"))
        blocker = sqlite3.connect(self.db_path, timeout=10)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            task = asyncio.create_task(store.claim("conv-lock-clock", _fingerprint("reply")))
            await asyncio.sleep(0.05)
            self.assertFalse(task.done())
            self.now[0] += 11
            blocker.rollback()
            decision = await task
        finally:
            blocker.close()

        self.assertFalse(decision.claimed)
        self.assertEqual(decision.reason, "expired")

    async def test_claimed_writes_recheck_clock_after_waiting_for_sqlite_lock(self):
        async def expire_while_blocked(conversation_id, operation):
            store, claim = await self._save_and_claim(conversation_id)
            blocker = sqlite3.connect(self.db_path, timeout=10)
            blocker.execute("BEGIN IMMEDIATE")
            try:
                task = asyncio.create_task(operation(store, claim.claim_token))
                await asyncio.sleep(0.05)
                self.assertFalse(task.done())
                self.now[0] += 11
                blocker.rollback()
                return await task
            finally:
                blocker.close()

        renew = await expire_while_blocked(
            "conv-lock-renew",
            lambda store, token: store.renew("conv-lock-renew", token),
        )
        finish = await expire_while_blocked(
            "conv-lock-finish",
            lambda store, token: store.finish("conv-lock-finish", token, _response("late")),
        )
        handoff = await expire_while_blocked(
            "conv-lock-handoff",
            lambda store, token: store.handoff(
                "conv-lock-handoff",
                token,
                "conv-lock-next",
                _payload("next"),
                _response("next", conversation_id="conv-lock-next"),
            ),
        )

        self.assertFalse(renew)
        self.assertFalse(finish)
        self.assertFalse(handoff)
        self.assertIsNone(await self._store().inspect("conv-lock-next"))

    async def test_old_schema_is_migrated_and_legacy_claim_never_replays(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                """
                CREATE TABLE studyloop_autonomous_sessions (
                    conversation_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('paused', 'in_flight')),
                    claim_token TEXT,
                    progress_started INTEGER NOT NULL DEFAULT 0
                        CHECK (progress_started IN (0, 1)),
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL,
                    expires_at DOUBLE PRECISION NOT NULL,
                    claimed_at DOUBLE PRECISION,
                    CHECK (
                        (state = 'paused' AND claim_token IS NULL
                         AND claimed_at IS NULL AND progress_started = 0)
                        OR
                        (state = 'in_flight' AND claim_token IS NOT NULL
                         AND claimed_at IS NOT NULL)
                    )
                )
                """
            )
            connection.execute(
                """
                INSERT INTO studyloop_autonomous_sessions
                    (conversation_id, payload_json, state, claim_token,
                     progress_started, created_at, updated_at, expires_at, claimed_at)
                VALUES (?, ?, 'in_flight', 'legacy-token', 0, ?, ?, ?, ?)
                """,
                ("conv-legacy", '{"legacy":true}', 990.0, 999.0, 1100.0, 999.0),
            )
            connection.commit()

        store = self._store(lease_seconds=10)
        self.assertEqual(await store.status("conv-legacy"), "in_flight")
        self.now[0] += 11
        decision = await store.claim("conv-legacy", _fingerprint("reply"))

        self.assertFalse(decision.claimed)
        self.assertEqual(decision.reason, "ambiguous")
        self.assertEqual((await store.inspect("conv-legacy")).state, "ambiguous")
        with closing(sqlite3.connect(self.db_path)) as connection:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(studyloop_autonomous_sessions)")
            }
        self.assertTrue({"claim_expires_at", "continue_fingerprint", "outcome_json"} <= columns)

    def test_postgres_connection_has_connect_lock_and_statement_timeouts(self):
        connect = MagicMock(return_value=SimpleNamespace())
        fake_psycopg = ModuleType("psycopg")
        fake_psycopg.connect = connect
        fake_conninfo = ModuleType("psycopg.conninfo")
        fake_conninfo.conninfo_to_dict = MagicMock(return_value={})
        store = AutonomousSessionStore(
            database_url="postgresql://example/studyloop",
            postgres_connect_timeout_seconds=7,
            postgres_lock_timeout_ms=8000,
            postgres_statement_timeout_ms=19000,
            postgres_tcp_user_timeout_ms=23000,
        )

        with (
            patch.dict(
                sys.modules,
                {"psycopg": fake_psycopg, "psycopg.conninfo": fake_conninfo},
            ),
            patch.dict(os.environ, {}, clear=True),
        ):
            connection = store._connect()

        self.assertIs(connection, connect.return_value)
        connect.assert_called_once_with(
            "postgresql://example/studyloop",
            connect_timeout=7,
            tcp_user_timeout=23000,
            options="-c lock_timeout=8000ms -c statement_timeout=19000ms",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
