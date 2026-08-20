"""Bounded PostgreSQL and cancellation contracts for Adaptive sessions."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from models.adaptive import AdaptiveTurn, NextStepDecision
from models.adaptive_session import AdaptiveSessionAggregate, AdaptiveTurnArtifact
from services.adaptive_sessions import AdaptiveSessionClaim, AdaptiveSessionStore


def _aggregate(
    session_id: str,
    *,
    lesson: str = "Compare the midpoint and retain the possible half.",
) -> AdaptiveSessionAggregate:
    return AdaptiveSessionAggregate(
        adaptive_session_id=session_id,
        user_id="runtime-user",
        document_id="notes.md",
        goal="learn binary search",
        status="active",
        current_quiz=None,
        current_artifact=AdaptiveTurnArtifact(
            adaptive_session_id=session_id,
            turn=1,
            turn_type="teach",
            lesson=lesson,
            decision=NextStepDecision(
                action="teach",
                topic="binary search",
                difficulty="medium",
                difficulty_score=0.5,
                question_type="choice",
                count=1,
                reason="Teach before the next quiz.",
            ),
            trajectory=[
                AdaptiveTurn(
                    turn=1,
                    action="teach",
                    topic="binary search",
                    difficulty_score=0.5,
                    reason="Teach before the next quiz.",
                )
            ],
        ),
    )


class TestAdaptiveSessionRuntimeBounds(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = str(Path(self.temp_dir.name) / "adaptive.sqlite3")
        self.now = [1_000.0]

    async def asyncTearDown(self) -> None:
        self.temp_dir.cleanup()

    def _store(
        self,
        *,
        cancel_drain_timeout_seconds: float = 0.01,
    ) -> AdaptiveSessionStore:
        return AdaptiveSessionStore(
            sqlite_path=self.database_path,
            ttl_seconds=60,
            operation_lease_seconds=10,
            cancel_drain_timeout_seconds=cancel_drain_timeout_seconds,
            clock=lambda: self.now[0],
        )

    async def _wait_for_thread_event(self, event: threading.Event) -> None:
        self.assertTrue(await asyncio.to_thread(event.wait, 2))

    async def _drain_background(self, store: AdaptiveSessionStore) -> None:
        workers = tuple(store._background_workers)
        if workers:
            await asyncio.wait_for(
                asyncio.gather(*workers, return_exceptions=True),
                timeout=2,
            )
            await asyncio.sleep(0)
        self.assertFalse(store._background_workers)

    def test_storage_budgets_must_be_positive_and_finite(self) -> None:
        parameter_names = (
            "postgres_connect_timeout_seconds",
            "postgres_lock_timeout_ms",
            "postgres_statement_timeout_ms",
            "postgres_tcp_user_timeout_ms",
            "schema_init_wait_timeout_seconds",
            "cancel_drain_timeout_seconds",
        )
        invalid_values = (0, -1, float("nan"), float("inf"), float("-inf"))

        for parameter_name in parameter_names:
            for invalid_value in invalid_values:
                with self.subTest(
                    parameter_name=parameter_name,
                    invalid_value=invalid_value,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"{parameter_name} must be positive",
                    ):
                        AdaptiveSessionStore(
                            sqlite_path=self.database_path,
                            **{parameter_name: invalid_value},
                        )

    def test_from_environment_reads_explicit_storage_budgets(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://example/studyloop",
                "ADAPTIVE_SESSION_PG_CONNECT_TIMEOUT_SECONDS": "7",
                "ADAPTIVE_SESSION_PG_LOCK_TIMEOUT_MS": "1234",
                "ADAPTIVE_SESSION_PG_STATEMENT_TIMEOUT_MS": "5678",
                "ADAPTIVE_SESSION_PG_TCP_USER_TIMEOUT_MS": "9876",
                "ADAPTIVE_SESSION_SCHEMA_INIT_WAIT_TIMEOUT_SECONDS": "9.5",
                "ADAPTIVE_SESSION_CANCEL_DRAIN_TIMEOUT_SECONDS": "2.5",
            },
            clear=True,
        ):
            store = AdaptiveSessionStore.from_environment()

        self.assertEqual(store._database_url, "postgresql://example/studyloop")
        self.assertEqual(store._postgres_connect_timeout_seconds, 7)
        self.assertEqual(store._postgres_lock_timeout_ms, 1234)
        self.assertEqual(store._postgres_statement_timeout_ms, 5678)
        self.assertEqual(store._postgres_tcp_user_timeout_ms, 9876)
        self.assertEqual(store._schema_init_wait_timeout_seconds, 9.5)
        self.assertEqual(store._cancel_drain_timeout_seconds, 2.5)

    def test_postgres_connection_preserves_options_and_overrides_unsafe_bounds(
        self,
    ) -> None:
        connect = MagicMock(return_value=SimpleNamespace())
        conninfo_to_dict = MagicMock(
            return_value={
                "options": (
                    "-csearch_path=tenant_schema -c lock_timeout=0 -c statement_timeout=999999999"
                )
            }
        )
        fake_psycopg = ModuleType("psycopg")
        fake_psycopg.connect = connect
        fake_conninfo = ModuleType("psycopg.conninfo")
        fake_conninfo.conninfo_to_dict = conninfo_to_dict
        database_url = (
            "postgresql://example/studyloop"
            "?connect_timeout=999&tcp_user_timeout=999999999"
            "&options=-csearch_path%3Dtenant_schema"
        )
        store = AdaptiveSessionStore(
            database_url=database_url,
            postgres_connect_timeout_seconds=7,
            postgres_lock_timeout_ms=1234,
            postgres_statement_timeout_ms=5678,
            postgres_tcp_user_timeout_ms=9876,
        )

        with (
            patch.dict(
                sys.modules,
                {"psycopg": fake_psycopg, "psycopg.conninfo": fake_conninfo},
            ),
            patch.dict(
                os.environ,
                {"PGOPTIONS": "-csearch_path=ignored_environment_schema"},
                clear=True,
            ),
        ):
            connection = store._connect()

        self.assertIs(connection, connect.return_value)
        conninfo_to_dict.assert_called_once_with(database_url)
        connect.assert_called_once_with(
            database_url,
            connect_timeout=7,
            tcp_user_timeout=9876,
            options=(
                "-csearch_path=tenant_schema -c lock_timeout=0 "
                "-c statement_timeout=999999999 "
                "-c lock_timeout=1234ms -c statement_timeout=5678ms"
            ),
        )

    def test_postgres_connection_uses_pgoptions_without_dsn_options(self) -> None:
        connect = MagicMock(return_value=SimpleNamespace())
        conninfo_to_dict = MagicMock(return_value={"dbname": "studyloop"})
        fake_psycopg = ModuleType("psycopg")
        fake_psycopg.connect = connect
        fake_conninfo = ModuleType("psycopg.conninfo")
        fake_conninfo.conninfo_to_dict = conninfo_to_dict
        database_url = "postgresql://example/studyloop"
        store = AdaptiveSessionStore(
            database_url=database_url,
            postgres_connect_timeout_seconds=7,
            postgres_lock_timeout_ms=1234,
            postgres_statement_timeout_ms=5678,
            postgres_tcp_user_timeout_ms=9876,
        )

        with (
            patch.dict(
                sys.modules,
                {"psycopg": fake_psycopg, "psycopg.conninfo": fake_conninfo},
            ),
            patch.dict(
                os.environ,
                {"PGOPTIONS": "-csearch_path=environment_schema"},
                clear=True,
            ),
        ):
            store._connect()

        connect.assert_called_once_with(
            database_url,
            connect_timeout=7,
            tcp_user_timeout=9876,
            options=(
                "-csearch_path=environment_schema "
                "-c lock_timeout=1234ms -c statement_timeout=5678ms"
            ),
        )

    def test_hidden_service_options_and_parser_errors_are_sanitized(self) -> None:
        secret = "private-password-that-must-not-leak"
        database_url = f"service=private-service password={secret}"
        connect = MagicMock(return_value=SimpleNamespace())
        conninfo_to_dict = MagicMock(return_value={"service": "private-service"})
        fake_psycopg = ModuleType("psycopg")
        fake_psycopg.connect = connect
        fake_conninfo = ModuleType("psycopg.conninfo")
        fake_conninfo.conninfo_to_dict = conninfo_to_dict
        store = AdaptiveSessionStore(database_url=database_url)

        with (
            patch.dict(
                sys.modules,
                {"psycopg": fake_psycopg, "psycopg.conninfo": fake_conninfo},
            ),
            patch.dict(os.environ, {"PGOPTIONS": "  \t"}, clear=True),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "must expose connection options",
            ) as hidden:
                store._connect()

        self.assertNotIn(secret, str(hidden.exception))
        connect.assert_not_called()

        conninfo_to_dict.side_effect = RuntimeError(f"could not parse {database_url}")
        with patch.dict(
            sys.modules,
            {"psycopg": fake_psycopg, "psycopg.conninfo": fake_conninfo},
        ):
            with self.assertRaisesRegex(
                ValueError,
                "PostgreSQL Adaptive session DATABASE_URL is invalid",
            ) as invalid:
                store._connect()

        self.assertNotIn(secret, str(invalid.exception))
        self.assertIsNone(invalid.exception.__cause__)

    async def test_schema_lock_timeout_and_failure_remain_retryable(self) -> None:
        store = AdaptiveSessionStore(
            sqlite_path=self.database_path,
            schema_init_wait_timeout_seconds=0.01,
        )
        self.assertTrue(store._schema_lock.acquire(blocking=False))
        try:
            initialization = asyncio.create_task(asyncio.to_thread(store._ensure_schema))
            done, _ = await asyncio.wait({initialization}, timeout=0.5)
            completed_while_blocked = initialization in done
        finally:
            store._schema_lock.release()
            await asyncio.gather(initialization, return_exceptions=True)

        self.assertTrue(
            completed_while_blocked,
            "schema initialization ignored its configured lock deadline",
        )
        with self.assertRaisesRegex(
            TimeoutError,
            "Adaptive session schema initialization timed out",
        ):
            initialization.result()
        self.assertFalse(store._schema_ready)
        await asyncio.to_thread(store._ensure_schema)
        self.assertTrue(store._schema_ready)

        failed = AdaptiveSessionStore(sqlite_path=str(Path(self.temp_dir.name) / "failed.sqlite3"))
        with patch.object(
            failed,
            "_transaction",
            side_effect=RuntimeError("schema failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "schema failed"):
                await asyncio.to_thread(failed._ensure_schema)
        self.assertTrue(failed._schema_lock.acquire(blocking=False))
        failed._schema_lock.release()
        self.assertFalse(failed._schema_ready)

    async def test_repeated_cancellation_uses_one_deadline_and_consumes_failure(
        self,
    ) -> None:
        store = self._store(cancel_drain_timeout_seconds=0.08)
        started = threading.Event()
        allow_return = threading.Event()

        def blocked_worker(*_args):
            started.set()
            if not allow_return.wait(timeout=5):
                raise TimeoutError("test did not release worker")
            raise RuntimeError("late Adaptive worker failure")

        with (
            patch.object(
                store,
                "_inspect_sync",
                side_effect=blocked_worker,
            ),
            patch("services.adaptive_sessions.logger.error") as log_error,
        ):
            task = asyncio.create_task(store.inspect("bounded"))
            await self._wait_for_thread_event(started)

            async def repeat_cancellation() -> None:
                for _ in range(8):
                    await asyncio.sleep(0.02)
                    task.cancel("later cancellation")

            started_at = time.monotonic()
            task.cancel("first cancellation")
            repeated = asyncio.create_task(repeat_cancellation())
            try:
                done, _ = await asyncio.wait({task}, timeout=0.3)
                completed_before_release = task in done
                elapsed = time.monotonic() - started_at
                background_before_release = len(store._background_workers)
            finally:
                repeated.cancel()
                await asyncio.gather(repeated, return_exceptions=True)
                allow_return.set()
                await asyncio.gather(task, return_exceptions=True)
                await self._drain_background(store)

        self.assertTrue(
            completed_before_release,
            "repeated cancellation extended the first drain deadline",
        )
        with self.assertRaises(asyncio.CancelledError) as raised:
            task.result()
        self.assertEqual(raised.exception.args, ("first cancellation",))
        self.assertLess(elapsed, 0.15)
        self.assertEqual(background_before_release, 1)
        log_error.assert_called()

    async def test_cancelled_complete_preserves_committed_state_after_deadline(
        self,
    ) -> None:
        store = self._store()
        await store.create(_aggregate("complete-session"))
        claim = await store.claim("complete-session", "submit")
        original_complete = store._complete_sync
        committed = threading.Event()
        allow_return = threading.Event()

        def commit_then_wait(*args):
            result = original_complete(*args)
            committed.set()
            if not allow_return.wait(timeout=5):
                raise TimeoutError("test did not release completed worker")
            return result

        with patch.object(store, "_complete_sync", side_effect=commit_then_wait):
            task = asyncio.create_task(
                store.complete(
                    "complete-session",
                    claim.token,
                    _aggregate("complete-session", lesson="Committed lesson."),
                    expected_revision=1,
                )
            )
            await self._wait_for_thread_event(committed)
            task.cancel("client disconnected")
            try:
                done, _ = await asyncio.wait({task}, timeout=0.5)
                completed_before_release = task in done
                background_before_release = len(store._background_workers)
            finally:
                allow_return.set()
                await asyncio.gather(task, return_exceptions=True)
                await self._drain_background(store)

        self.assertTrue(
            completed_before_release,
            "cancelled complete ignored its configured drain deadline",
        )
        with self.assertRaises(asyncio.CancelledError) as raised:
            task.result()
        self.assertEqual(raised.exception.args, ("client disconnected",))
        self.assertEqual(background_before_release, 1)
        restored = await store.inspect("complete-session")
        self.assertEqual(restored.revision, 2)
        self.assertFalse(restored.busy)
        self.assertEqual(
            restored.aggregate.current_artifact.lesson,
            "Committed lesson.",
        )

    async def test_cancelled_claim_cleans_commit_after_worker_ack_failure(self) -> None:
        store = self._store()
        await store.create(_aggregate("ack-loss-session"))
        committed = threading.Event()
        allow_failure = threading.Event()
        original_claim = store._claim_sync

        def commit_then_fail(*args):
            original_claim(*args)
            committed.set()
            if not allow_failure.wait(timeout=5):
                raise TimeoutError("test did not release claim worker")
            raise RuntimeError("claim commit acknowledgement failed")

        with (
            patch.object(
                store,
                "_claim_sync",
                side_effect=commit_then_fail,
            ),
            patch("services.adaptive_sessions.logger.error") as log_error,
        ):
            task = asyncio.create_task(store.claim("ack-loss-session", "submit"))
            await self._wait_for_thread_event(committed)
            task.cancel("client disconnected")
            try:
                done, _ = await asyncio.wait({task}, timeout=0.5)
                completed_before_release = task in done
                background_before_release = len(store._background_workers)
            finally:
                allow_failure.set()
                await asyncio.gather(task, return_exceptions=True)
                await self._drain_background(store)

        self.assertTrue(
            completed_before_release,
            "cancelled claim ignored its configured drain deadline",
        )
        with self.assertRaises(asyncio.CancelledError) as raised:
            task.result()
        self.assertEqual(raised.exception.args, ("client disconnected",))
        self.assertEqual(background_before_release, 1)
        log_error.assert_called()
        retry = await store.claim("ack-loss-session", "submit")
        self.assertTrue(retry.claimed)
        await store.release("ack-loss-session", retry.token)

    async def test_late_old_claim_cleanup_does_not_release_takeover(self) -> None:
        first = self._store()
        peer = self._store()
        await first.create(_aggregate("takeover-session"))
        committed = threading.Event()
        allow_return = threading.Event()
        observed = {}
        released = []
        original_claim = first._claim_sync
        original_release = first._release_sync

        def commit_then_wait(*args):
            decision = original_claim(*args)
            observed["decision"] = decision
            committed.set()
            if not allow_return.wait(timeout=5):
                raise TimeoutError("test did not release claim worker")
            return decision

        def capture_release(session_id, token):
            released.append((session_id, token))
            return original_release(session_id, token)

        with (
            patch.object(first, "_claim_sync", side_effect=commit_then_wait),
            patch.object(first, "_release_sync", side_effect=capture_release),
        ):
            task = asyncio.create_task(first.claim("takeover-session", "submit"))
            await self._wait_for_thread_event(committed)
            task.cancel("client disconnected")
            try:
                done, _ = await asyncio.wait({task}, timeout=0.5)
                completed_before_release = task in done
                background_before_release = len(first._background_workers)

                self.now[0] += 11
                takeover = await peer.claim("takeover-session", "submit")
                self.assertTrue(takeover.claimed)
                self.assertNotEqual(
                    observed["decision"].token,
                    takeover.token,
                )
            finally:
                allow_return.set()
                await asyncio.gather(task, return_exceptions=True)
                await self._drain_background(first)

        self.assertTrue(
            completed_before_release,
            "cancelled claim ignored its configured drain deadline",
        )
        with self.assertRaises(asyncio.CancelledError) as raised:
            task.result()
        self.assertEqual(raised.exception.args, ("client disconnected",))
        self.assertEqual(background_before_release, 1)
        self.assertEqual(
            released,
            [("takeover-session", observed["decision"].token)],
        )
        restored = await peer.inspect("takeover-session")
        self.assertTrue(restored.busy)
        self.assertTrue(await peer.release("takeover-session", takeover.token))

    async def test_claim_worker_and_cleanup_share_one_deadline(self) -> None:
        store = self._store(cancel_drain_timeout_seconds=1)
        started = threading.Event()
        allow_return = threading.Event()
        deadlines = []
        original_drain = store._drain_cancelled_worker

        def delayed_claim(_session_id, _operation, token):
            started.set()
            if not allow_return.wait(timeout=5):
                raise TimeoutError("test did not release claim worker")
            return AdaptiveSessionClaim(claimed=True, token=token)

        async def observe_deadline(worker, **kwargs):
            deadlines.append(kwargs.get("deadline"))
            return await original_drain(worker, **kwargs)

        with (
            patch.object(store, "_claim_sync", side_effect=delayed_claim),
            patch.object(
                store,
                "_release_sync",
                side_effect=RuntimeError("cleanup failed"),
            ),
            patch.object(
                store,
                "_drain_cancelled_worker",
                side_effect=observe_deadline,
            ),
            patch("services.adaptive_sessions.logger.error"),
        ):
            task = asyncio.create_task(store.claim("deadline-session", "submit"))
            await self._wait_for_thread_event(started)
            task.cancel("client disconnected")
            try:
                allow_return.set()
                done, _ = await asyncio.wait({task}, timeout=0.5)
                completed = task in done
            finally:
                allow_return.set()
                await asyncio.gather(task, return_exceptions=True)
                await self._drain_background(store)

        self.assertTrue(completed)
        with self.assertRaises(asyncio.CancelledError) as raised:
            task.result()
        self.assertEqual(raised.exception.args, ("client disconnected",))
        self.assertEqual(len(deadlines), 2)
        self.assertIsNotNone(deadlines[0])
        self.assertEqual(deadlines[0], deadlines[1])
        self.assertFalse(store._background_workers)

    async def test_claim_cleanup_timeout_is_tracked_from_the_first_deadline(
        self,
    ) -> None:
        store = self._store(cancel_drain_timeout_seconds=0.2)
        claim_started = threading.Event()
        allow_claim = threading.Event()
        cleanup_started = threading.Event()
        allow_cleanup = threading.Event()

        def delayed_claim(_session_id, _operation, token):
            claim_started.set()
            if not allow_claim.wait(timeout=5):
                raise TimeoutError("test did not release claim worker")
            return AdaptiveSessionClaim(claimed=True, token=token)

        def delayed_release(_session_id, _token):
            cleanup_started.set()
            if not allow_cleanup.wait(timeout=5):
                raise TimeoutError("test did not release cleanup worker")
            return False

        with (
            patch.object(store, "_claim_sync", side_effect=delayed_claim),
            patch.object(store, "_release_sync", side_effect=delayed_release),
        ):
            task = asyncio.create_task(store.claim("tracked-session", "submit"))
            await self._wait_for_thread_event(claim_started)
            started_at = time.monotonic()
            task.cancel("client disconnected")
            try:
                await asyncio.sleep(0.12)
                allow_claim.set()
                await self._wait_for_thread_event(cleanup_started)
                done, _ = await asyncio.wait({task}, timeout=0.14)
                completed_before_cleanup_release = task in done
                elapsed = time.monotonic() - started_at
                background_before_release = len(store._background_workers)
            finally:
                allow_claim.set()
                allow_cleanup.set()
                await asyncio.gather(task, return_exceptions=True)
                await self._drain_background(store)

        self.assertTrue(
            completed_before_cleanup_release,
            "claim cleanup started a fresh cancellation deadline",
        )
        with self.assertRaises(asyncio.CancelledError) as raised:
            task.result()
        self.assertEqual(raised.exception.args, ("client disconnected",))
        self.assertLess(elapsed, 0.4)
        self.assertEqual(background_before_release, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
