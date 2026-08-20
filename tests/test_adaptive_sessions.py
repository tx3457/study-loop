from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import tempfile
import threading
import types
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from models.adaptive import AdaptiveTurn, NextStepDecision
from models.adaptive_session import (
    AdaptiveSessionAggregate,
    AdaptiveTurnArtifact,
)
from services.adaptive_sessions import (
    AdaptiveSessionCapacityError,
    AdaptiveSessionCorruptError,
    AdaptiveSessionPayloadTooLargeError,
    AdaptiveSessionStartConflictError,
    AdaptiveSessionStore,
)


def _decision(action: str) -> NextStepDecision:
    return NextStepDecision(
        action=action,
        topic="binary search",
        difficulty="medium",
        difficulty_score=0.5,
        question_type="choice",
        count=1,
        reason=f"{action} is the next learning step",
    )


def _trajectory(action: str) -> list[AdaptiveTurn]:
    return [
        AdaptiveTurn(
            turn=1,
            action=action,
            topic="binary search",
            difficulty_score=0.5,
            reason=f"{action} is appropriate",
        )
    ]


def _active_aggregate(
    session_id: str,
    *,
    user_id: str = "user-1",
    document_id: str = "notes.md",
    goal: str = "learn binary search",
    lesson: str = "Compare the midpoint and retain the possible half.",
) -> AdaptiveSessionAggregate:
    return AdaptiveSessionAggregate(
        adaptive_session_id=session_id,
        user_id=user_id,
        document_id=document_id,
        goal=goal,
        status="active",
        current_quiz=None,
        current_artifact=AdaptiveTurnArtifact(
            adaptive_session_id=session_id,
            turn=1,
            turn_type="teach",
            lesson=lesson,
            decision=_decision("teach"),
            trajectory=_trajectory("teach"),
        ),
    )


def _completed_aggregate(
    session_id: str,
    *,
    user_id: str = "user-1",
    document_id: str = "notes.md",
    goal: str = "learn binary search",
) -> AdaptiveSessionAggregate:
    return AdaptiveSessionAggregate(
        adaptive_session_id=session_id,
        user_id=user_id,
        document_id=document_id,
        goal=goal,
        status="completed",
        current_quiz=None,
        current_artifact=AdaptiveTurnArtifact(
            adaptive_session_id=session_id,
            turn=1,
            done=True,
            decision=_decision("finish"),
            trajectory=_trajectory("finish"),
            summary="The learning goal has been met.",
            terminate_reason="agent_finish",
        ),
    )


def _changed(
    aggregate: AdaptiveSessionAggregate,
    *,
    lesson: str,
) -> AdaptiveSessionAggregate:
    changed = aggregate.model_copy(deep=True)
    changed.current_artifact.lesson = lesson
    return changed


class TestAdaptiveSessionStore(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "adaptive.sqlite3")
        self.now = [1_000.0]
        self.store = AdaptiveSessionStore(
            sqlite_path=self.db_path,
            ttl_seconds=60,
            operation_lease_seconds=10,
            max_count=3,
            clock=lambda: self.now[0],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _peer(self, *, max_count: int = 3) -> AdaptiveSessionStore:
        return AdaptiveSessionStore(
            sqlite_path=self.db_path,
            ttl_seconds=60,
            operation_lease_seconds=10,
            max_count=max_count,
            clock=lambda: self.now[0],
        )

    async def test_reopens_aggregate_and_replays_start_key(self) -> None:
        request = {
            "user_id": "user-1",
            "document_id": "notes.md",
            "goal": "learn binary search",
        }
        created = await self.store.create(
            _active_aggregate("s-1"),
            start_key="start-key-123",
            start_request=request,
        )
        self.assertTrue(created.created)
        self.assertEqual(created.record.revision, 1)
        self.assertEqual(created.record.aggregate.adaptive_session_id, "s-1")

        peer = self._peer()
        replay = await peer.find_start("start-key-123", request)
        self.assertIsNotNone(replay)
        self.assertEqual(replay.aggregate, _active_aggregate("s-1"))

        duplicate = await peer.create(
            _active_aggregate("s-2"),
            start_key="start-key-123",
            start_request=request,
        )
        self.assertFalse(duplicate.created)
        self.assertEqual(duplicate.record.aggregate.adaptive_session_id, "s-1")

        with self.assertRaises(AdaptiveSessionStartConflictError) as mismatch:
            await peer.find_start(
                "start-key-123",
                {**request, "document_id": "other.md"},
            )
        self.assertEqual(mismatch.exception.reason, "payload_mismatch")

    async def test_concurrent_same_start_key_creates_one_logical_session(self) -> None:
        request = {"document_id": "notes.md", "goal": "learn"}
        first, second = await asyncio.gather(
            self.store.create(
                _active_aggregate("s-a"),
                start_key="same-key",
                start_request=request,
            ),
            self._peer().create(
                _active_aggregate("s-b"),
                start_key="same-key",
                start_request=request,
            ),
        )

        self.assertEqual(sum(result.created for result in (first, second)), 1)
        self.assertEqual(
            first.record.aggregate.adaptive_session_id,
            second.record.aggregate.adaptive_session_id,
        )

    async def test_checkpoint_and_complete_require_token_and_revision(self) -> None:
        await self.store.create(_active_aggregate("s-1"))
        claim = await self.store.claim("s-1", "submit")
        self.assertTrue(claim.claimed)
        self.assertEqual(claim.record.revision, 1)

        changed = _changed(
            claim.record.aggregate,
            lesson="Checkpointed lesson.",
        )
        self.assertIsNone(
            await self.store.checkpoint(
                "s-1",
                claim.token,
                changed,
                expected_revision=2,
            )
        )
        checkpoint = await self.store.checkpoint(
            "s-1",
            claim.token,
            changed,
            expected_revision=1,
        )
        self.assertEqual(checkpoint.revision, 2)

        self.assertIsNone(
            await self.store.complete(
                "s-1",
                claim.token,
                changed,
                expected_revision=1,
            )
        )
        completed_operation = await self.store.complete(
            "s-1",
            claim.token,
            changed,
            expected_revision=2,
        )
        self.assertEqual(completed_operation.revision, 3)
        self.assertFalse(completed_operation.busy)

    async def test_lease_takeover_fences_previous_owner(self) -> None:
        await self.store.create(_active_aggregate("s-1"))
        first = await self.store.claim("s-1", "submit")
        blocked = await self._peer().claim("s-1", "submit")
        self.assertFalse(blocked.claimed)
        self.assertEqual(blocked.reason, "in_progress")

        self.now[0] += 11
        peer = self._peer()
        takeover = await peer.claim("s-1", "submit")
        self.assertTrue(takeover.claimed)
        self.assertNotEqual(first.token, takeover.token)

        changed = _changed(first.record.aggregate, lesson="Stale owner.")
        self.assertIsNone(
            await self.store.checkpoint(
                "s-1",
                first.token,
                changed,
                expected_revision=1,
            )
        )
        winner = _changed(takeover.record.aggregate, lesson="New owner.")
        saved = await peer.complete(
            "s-1",
            takeover.token,
            winner,
            expected_revision=1,
        )
        self.assertEqual(saved.aggregate.current_artifact.lesson, "New owner.")

    async def test_claim_and_checkpoint_slide_ttl_at_expiry_boundary(self) -> None:
        created = await self.store.create(_active_aggregate("near-expiry"))
        self.assertEqual(created.record.expires_at, 1_060.0)

        self.now[0] = 1_059.0
        claim = await self.store.claim("near-expiry", "submit")
        self.assertTrue(claim.claimed)
        self.assertEqual(claim.record.expires_at, 1_119.0)

        # The original TTL has passed, but the legitimate claim extended it.
        self.now[0] = 1_061.0
        checkpoint = await self.store.checkpoint(
            "near-expiry",
            claim.token,
            _changed(claim.record.aggregate, lesson="Still progressing."),
            expected_revision=1,
        )
        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint.expires_at, 1_121.0)
        self.assertEqual(checkpoint.revision, 2)

    async def test_live_lease_never_outlives_claimed_session_ttl(self) -> None:
        short_path = str(Path(self.temp_dir.name) / "short-ttl.sqlite3")
        store = AdaptiveSessionStore(
            sqlite_path=short_path,
            ttl_seconds=5,
            operation_lease_seconds=10,
            clock=lambda: self.now[0],
        )
        created = await store.create(_active_aggregate("short-ttl"))
        self.assertEqual(created.record.expires_at, 1_005.0)

        self.now[0] = 1_004.0
        claim = await store.claim("short-ttl", "submit")
        self.assertTrue(claim.claimed)
        self.assertEqual(claim.record.expires_at, 1_014.0)

        self.now[0] = 1_012.0
        saved = await store.complete(
            "short-ttl",
            claim.token,
            claim.record.aggregate,
            expected_revision=1,
        )
        self.assertIsNotNone(saved)
        self.assertEqual(saved.expires_at, 1_017.0)

    async def test_clock_is_read_after_sqlite_write_lock_is_acquired(self) -> None:
        await self.store.create(_active_aggregate("locked"))
        self.now[0] = 1_059.0
        clock_called = threading.Event()

        def clock() -> float:
            clock_called.set()
            return self.now[0]

        self.store._clock = clock
        original_connect = self.store._connect
        connection_opened = threading.Event()

        def observed_connect():
            connection = original_connect()
            connection_opened.set()
            return connection

        blocker = sqlite3.connect(self.db_path, timeout=10)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            with patch.object(self.store, "_connect", side_effect=observed_connect):
                task = asyncio.create_task(self.store.claim("locked", "submit"))
                self.assertTrue(await asyncio.to_thread(connection_opened.wait, 2))
                await asyncio.sleep(0.05)
                self.assertFalse(clock_called.is_set())
                self.now[0] = 1_061.0
                blocker.commit()
                decision = await task
        finally:
            blocker.close()

        self.assertTrue(clock_called.is_set())
        self.assertFalse(decision.claimed)
        self.assertEqual(decision.reason, "expired")

    async def test_identity_change_cannot_commit(self) -> None:
        original = _active_aggregate("s-1")
        await self.store.create(original)
        claim = await self.store.claim("s-1", "submit")

        changed_identity = claim.record.aggregate.model_copy(deep=True)
        changed_identity.goal = "a different immutable goal"
        self.assertIsNone(
            await self.store.complete(
                "s-1",
                claim.token,
                changed_identity,
                expected_revision=1,
            )
        )
        self.assertTrue(await self.store.release("s-1", claim.token))
        restored = await self.store.inspect("s-1")
        self.assertEqual(restored.aggregate.goal, original.goal)

    async def test_every_write_revalidates_mutated_nested_models(self) -> None:
        aggregate = _active_aggregate("invalid")
        aggregate.current_artifact.lesson = None

        with self.assertRaises(ValueError):
            await self.store.create(aggregate)
        self.assertFalse(Path(self.db_path).exists())

    async def test_cancelled_claim_releases_unreturned_token(self) -> None:
        await self.store.create(_active_aggregate("cancelled"))
        entered = threading.Event()
        finish = threading.Event()
        original_claim = self.store._claim_sync

        def commit_then_wait(*args):
            decision = original_claim(*args)
            entered.set()
            finish.wait(timeout=2)
            return decision

        with patch.object(self.store, "_claim_sync", side_effect=commit_then_wait):
            task = asyncio.create_task(self.store.claim("cancelled", "submit"))
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            task.cancel()
            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        restored = await self.store.inspect("cancelled")
        self.assertFalse(restored.busy)
        retry = await self.store.claim("cancelled", "submit")
        self.assertTrue(retry.claimed)

    async def test_session_cannot_be_revived_after_slid_ttl_expires(self) -> None:
        await self.store.create(_active_aggregate("expired"))
        self.now[0] = 1_059.0
        claim = await self.store.claim("expired", "submit")
        self.assertEqual(claim.record.expires_at, 1_119.0)

        self.now[0] = 1_120.0
        changed = _changed(claim.record.aggregate, lesson="Too late.")
        self.assertIsNone(
            await self.store.checkpoint(
                "expired",
                claim.token,
                changed,
                expected_revision=1,
            )
        )
        self.assertIsNone(
            await self.store.complete(
                "expired",
                claim.token,
                changed,
                expected_revision=1,
            )
        )
        self.assertTrue(await self.store.release("expired", claim.token))
        restored = await self.store.inspect("expired")
        self.assertTrue(restored.expired)
        self.assertEqual(restored.expires_at, 1_119.0)

    async def test_done_is_terminal_and_capacity_only_evicts_done(self) -> None:
        limited = self._peer(max_count=2)
        await limited.create(_active_aggregate("active-1"))
        await limited.create(_active_aggregate("active-2"))
        with self.assertRaises(AdaptiveSessionCapacityError):
            await limited.create(_active_aggregate("active-3"))

        claim = await limited.claim("active-1", "finish")
        done = await limited.complete(
            "active-1",
            claim.token,
            _completed_aggregate("active-1"),
            expected_revision=1,
        )
        self.assertEqual(done.aggregate.status, "completed")
        refused = await limited.claim("active-1", "submit")
        self.assertFalse(refused.claimed)
        self.assertEqual(refused.reason, "done")

        created = await limited.create(_active_aggregate("active-3"))
        self.assertTrue(created.created)
        self.assertIsNone(await limited.inspect("active-1"))
        self.assertIsNotNone(await limited.inspect("active-2"))

    async def test_corrupt_row_payload_identity_and_status_fail_closed(self) -> None:
        await self.store.create(_active_aggregate("s-1"))
        tampered = _active_aggregate("s-1", goal="tampered goal")
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                """
                UPDATE studyloop_adaptive_sessions
                SET payload_json = ? WHERE session_id = ?
                """,
                (
                    json.dumps(
                        tampered.model_dump(mode="json"),
                        sort_keys=True,
                    ),
                    "s-1",
                ),
            )
            connection.commit()
        with self.assertRaises(AdaptiveSessionCorruptError):
            await self.store.inspect("s-1")

        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                """
                UPDATE studyloop_adaptive_sessions
                SET payload_json = ?, status = 'done' WHERE session_id = ?
                """,
                (
                    _active_aggregate("s-1").model_dump_json(),
                    "s-1",
                ),
            )
            connection.commit()
        with self.assertRaises(AdaptiveSessionCorruptError):
            await self.store.inspect("s-1")

    async def test_payload_limit_is_checked_before_database_write(self) -> None:
        tiny_path = str(Path(self.temp_dir.name) / "tiny.sqlite3")
        tiny = AdaptiveSessionStore(
            sqlite_path=tiny_path,
            max_payload_bytes=100,
            clock=lambda: self.now[0],
        )
        with self.assertRaises(AdaptiveSessionPayloadTooLargeError):
            await tiny.create(_active_aggregate("too-large"))
        self.assertFalse(Path(tiny_path).exists())

    async def test_postgres_timeouts_are_connection_startup_options(self) -> None:
        captured: dict[str, object] = {}

        class Connection:
            def close(self) -> None:
                captured["closed"] = True

        def connect(database_url: str, **kwargs):
            captured["database_url"] = database_url
            captured.update(kwargs)
            return Connection()

        fake_psycopg = types.ModuleType("psycopg")
        fake_psycopg.connect = connect
        fake_conninfo = types.ModuleType("psycopg.conninfo")
        fake_conninfo.conninfo_to_dict = lambda _database_url: {}
        store = AdaptiveSessionStore(
            database_url="postgresql://example.invalid/studyloop",
            postgres_connect_timeout_seconds=7,
            postgres_lock_timeout_ms=1_234,
            postgres_statement_timeout_ms=5_678,
        )
        with (
            patch.dict(
                sys.modules,
                {"psycopg": fake_psycopg, "psycopg.conninfo": fake_conninfo},
            ),
            patch.dict(os.environ, {}, clear=True),
        ):
            connection = store._connect()
            connection.close()

        self.assertEqual(captured["connect_timeout"], 7)
        self.assertEqual(captured["tcp_user_timeout"], 30_000)
        self.assertEqual(
            captured["options"],
            "-c lock_timeout=1234ms -c statement_timeout=5678ms",
        )
        self.assertTrue(captured["closed"])

    async def test_non_finite_ttl_and_lease_are_rejected(self) -> None:
        for value in (float("nan"), float("inf")):
            with self.subTest(ttl=value):
                with self.assertRaises(ValueError):
                    AdaptiveSessionStore(sqlite_path=self.db_path, ttl_seconds=value)
            with self.subTest(lease=value):
                with self.assertRaises(ValueError):
                    AdaptiveSessionStore(
                        sqlite_path=self.db_path,
                        operation_lease_seconds=value,
                    )


if __name__ == "__main__":
    unittest.main()
