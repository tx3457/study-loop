"""Bounded PostgreSQL and cancellation contracts for Autonomous sessions."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from services.autonomous_sessions import AutonomousSessionStore, SessionClaim


def _payload(label: str) -> dict:
    return {
        "schema_version": 1,
        "messages": [{"role": "user", "content": label}],
        "steps": [{"round_index": 0, "tool_name": "ask_user"}],
        "evidence_registry": {},
    }


def _fingerprint(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _response(label: str, *, conversation_id: str | None = None) -> dict:
    response = {"response_schema_version": 2, "final_answer": label}
    if conversation_id is not None:
        response.update(
            {
                "final_answer": "",
                "awaiting_user_input": True,
                "user_question": label,
                "conversation_id": conversation_id,
            }
        )
    return response


class TestAutonomousSessionRuntimeBounds(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = str(Path(self.temp_dir.name) / "autonomous.sqlite3")
        self.now = [1_000.0]

    async def asyncTearDown(self) -> None:
        self.temp_dir.cleanup()

    def _store(
        self,
        *,
        cancel_drain_timeout_seconds: float = 0.01,
    ) -> AutonomousSessionStore:
        return AutonomousSessionStore(
            sqlite_path=self.database_path,
            ttl_seconds=60,
            operation_lease_seconds=10,
            cancel_drain_timeout_seconds=cancel_drain_timeout_seconds,
            clock=lambda: self.now[0],
        )

    async def _wait_for_thread_event(self, event: threading.Event) -> None:
        self.assertTrue(await asyncio.to_thread(event.wait, 2))

    async def _drain_background(self, store: AutonomousSessionStore) -> None:
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
                        AutonomousSessionStore(
                            sqlite_path=self.database_path,
                            **{parameter_name: invalid_value},
                        )

    def test_from_environment_reads_explicit_storage_budgets(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://example/studyloop",
                "AUTONOMOUS_SESSION_PG_CONNECT_TIMEOUT_SECONDS": "7",
                "AUTONOMOUS_SESSION_PG_LOCK_TIMEOUT_MS": "1234",
                "AUTONOMOUS_SESSION_PG_STATEMENT_TIMEOUT_MS": "5678",
                "AUTONOMOUS_SESSION_PG_TCP_USER_TIMEOUT_MS": "9876",
                "AUTONOMOUS_SESSION_SCHEMA_INIT_WAIT_TIMEOUT_SECONDS": "9.5",
                "AUTONOMOUS_SESSION_CANCEL_DRAIN_TIMEOUT_SECONDS": "2.5",
            },
            clear=True,
        ):
            store = AutonomousSessionStore.from_environment()

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
        store = AutonomousSessionStore(
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
        store = AutonomousSessionStore(
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
        store = AutonomousSessionStore(database_url=database_url)

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
                "PostgreSQL Autonomous session DATABASE_URL is invalid",
            ) as invalid:
                store._connect()

        self.assertNotIn(secret, str(invalid.exception))
        self.assertIsNone(invalid.exception.__cause__)

        conninfo_to_dict.side_effect = None
        conninfo_to_dict.return_value = {"dbname": "studyloop"}
        with (
            patch.dict(
                sys.modules,
                {"psycopg": fake_psycopg, "psycopg.conninfo": fake_conninfo},
            ),
            patch.dict(
                os.environ,
                {"PGSERVICE": "hidden-service", "PGOPTIONS": ""},
                clear=True,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "must expose connection options"):
                store._connect()

    async def test_schema_lock_timeout_and_failure_remain_retryable(self) -> None:
        store = AutonomousSessionStore(
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
            "Autonomous session schema initialization timed out",
        ):
            initialization.result()
        self.assertFalse(store._schema_ready)
        await asyncio.to_thread(store._ensure_schema)
        self.assertTrue(store._schema_ready)

        failed = AutonomousSessionStore(
            sqlite_path=str(Path(self.temp_dir.name) / "failed.sqlite3")
        )
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

    async def test_repeated_cancellation_is_bounded_and_consumes_late_failure(
        self,
    ) -> None:
        store = self._store(cancel_drain_timeout_seconds=0.08)
        started = threading.Event()
        allow_return = threading.Event()

        def blocked_worker(*_args):
            started.set()
            if not allow_return.wait(timeout=5):
                raise TimeoutError("test did not release Autonomous worker")
            raise RuntimeError("late Autonomous worker failure")

        with (
            patch.object(store, "_inspect_sync", side_effect=blocked_worker),
            patch("services.autonomous_sessions.logger.error") as log_error,
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
                done, _ = await asyncio.wait({task}, timeout=0.5)
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
        self.assertLess(elapsed, 0.35)
        self.assertEqual(background_before_release, 1)
        log_error.assert_called()

    async def test_cancelled_claim_releases_commit_when_worker_returns(self) -> None:
        store = self._store(cancel_drain_timeout_seconds=1)
        await store.save("normal-claim", _payload("normal"))
        committed = threading.Event()
        allow_return = threading.Event()
        original_claim = store._claim_sync

        def commit_then_wait(*args):
            decision = original_claim(*args)
            committed.set()
            if not allow_return.wait(timeout=5):
                raise TimeoutError("test did not release claim worker")
            return decision

        with patch.object(store, "_claim_sync", side_effect=commit_then_wait):
            task = asyncio.create_task(store.claim("normal-claim", _fingerprint("reply")))
            await self._wait_for_thread_event(committed)
            task.cancel("client disconnected")
            allow_return.set()
            done, _ = await asyncio.wait({task}, timeout=0.5)

        self.assertIn(task, done)
        with self.assertRaises(asyncio.CancelledError) as raised:
            task.result()
        self.assertEqual(raised.exception.args, ("client disconnected",))
        self.assertFalse(store._background_workers)
        retry = await store.claim("normal-claim", _fingerprint("reply"))
        self.assertTrue(retry.claimed)
        self.assertTrue(await store.cancel("normal-claim", retry.claim_token))

    async def test_cancelled_claim_cleans_commit_after_worker_ack_failure(
        self,
    ) -> None:
        store = self._store()
        await store.save("ack-loss-claim", _payload("ack loss"))
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
            patch.object(store, "_claim_sync", side_effect=commit_then_fail),
            patch("services.autonomous_sessions.logger.error") as log_error,
        ):
            task = asyncio.create_task(store.claim("ack-loss-claim", _fingerprint("reply")))
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
        retry = await store.claim("ack-loss-claim", _fingerprint("reply"))
        self.assertTrue(retry.claimed)
        self.assertTrue(await store.cancel("ack-loss-claim", retry.claim_token))

    async def test_claim_worker_and_cleanup_share_the_first_deadline(self) -> None:
        store = self._store(cancel_drain_timeout_seconds=0.2)
        claim_started = threading.Event()
        allow_claim = threading.Event()
        cleanup_started = threading.Event()
        allow_cleanup = threading.Event()

        def delayed_claim(_conversation_id, _fingerprint_value, claim_token):
            claim_started.set()
            if not allow_claim.wait(timeout=5):
                raise TimeoutError("test did not release claim worker")
            return SessionClaim(claimed=True, claim_token=claim_token)

        def delayed_cancel(_conversation_id, _claim_token):
            cleanup_started.set()
            if not allow_cleanup.wait(timeout=5):
                raise TimeoutError("test did not release cleanup worker")
            return False

        with (
            patch.object(store, "_claim_sync", side_effect=delayed_claim),
            patch.object(store, "_cancel_sync", side_effect=delayed_cancel),
        ):
            task = asyncio.create_task(store.claim("deadline-claim", _fingerprint("reply")))
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

    async def test_late_old_claim_cleanup_does_not_release_takeover(self) -> None:
        first = self._store()
        peer = self._store()
        await first.save("takeover-claim", _payload("takeover"))
        fingerprint = _fingerprint("reply")
        committed = threading.Event()
        allow_return = threading.Event()
        observed = {}
        cancelled = []
        original_claim = first._claim_sync
        original_cancel = first._cancel_sync

        def commit_then_wait(*args):
            decision = original_claim(*args)
            observed["decision"] = decision
            committed.set()
            if not allow_return.wait(timeout=5):
                raise TimeoutError("test did not release claim worker")
            return decision

        def capture_cancel(conversation_id, claim_token):
            cancelled.append((conversation_id, claim_token))
            return original_cancel(conversation_id, claim_token)

        with (
            patch.object(first, "_claim_sync", side_effect=commit_then_wait),
            patch.object(first, "_cancel_sync", side_effect=capture_cancel),
        ):
            task = asyncio.create_task(first.claim("takeover-claim", fingerprint))
            await self._wait_for_thread_event(committed)
            task.cancel("client disconnected")
            try:
                done, _ = await asyncio.wait({task}, timeout=0.5)
                completed_before_release = task in done
                background_before_release = len(first._background_workers)

                self.now[0] += 11
                takeover = await peer.claim("takeover-claim", fingerprint)
                self.assertTrue(takeover.claimed)
                self.assertNotEqual(
                    observed["decision"].claim_token,
                    takeover.claim_token,
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
            cancelled,
            [("takeover-claim", observed["decision"].claim_token)],
        )
        blocked = await peer.claim("takeover-claim", fingerprint)
        self.assertFalse(blocked.claimed)
        self.assertEqual(blocked.reason, "in_progress")
        self.assertTrue(await peer.cancel("takeover-claim", takeover.claim_token))

    async def test_cancelled_finish_can_commit_after_the_drain_deadline(self) -> None:
        store = self._store()
        await store.save("late-finish", _payload("finish"))
        claim = await store.claim("late-finish", _fingerprint("reply"))
        response = _response("durable final answer")
        started = threading.Event()
        allow_commit = threading.Event()
        original_finish = store._finish_sync

        def commit_after_release(*args):
            started.set()
            if not allow_commit.wait(timeout=5):
                raise TimeoutError("test did not release finish worker")
            return original_finish(*args)

        with patch.object(store, "_finish_sync", side_effect=commit_after_release):
            task = asyncio.create_task(store.finish("late-finish", claim.claim_token, response))
            await self._wait_for_thread_event(started)
            task.cancel("client disconnected")
            try:
                done, _ = await asyncio.wait({task}, timeout=0.5)
                completed_before_commit = task in done
                background_before_commit = len(store._background_workers)
            finally:
                allow_commit.set()
                await asyncio.gather(task, return_exceptions=True)
                await self._drain_background(store)

        self.assertTrue(
            completed_before_commit,
            "cancelled finish ignored its configured drain deadline",
        )
        with self.assertRaises(asyncio.CancelledError) as raised:
            task.result()
        self.assertEqual(raised.exception.args, ("client disconnected",))
        self.assertEqual(background_before_commit, 1)
        restored = await store.inspect("late-finish")
        self.assertEqual(restored.state, "completed")
        self.assertEqual(restored.outcome, response)

    async def test_cancelled_handoff_can_commit_after_the_drain_deadline(self) -> None:
        store = self._store()
        await store.save("late-handoff", _payload("handoff"))
        fingerprint = _fingerprint("reply")
        claim = await store.claim("late-handoff", fingerprint)
        response = _response("next question", conversation_id="late-next")
        started = threading.Event()
        allow_commit = threading.Event()
        original_handoff = store._handoff_sync

        def commit_after_release(*args):
            started.set()
            if not allow_commit.wait(timeout=5):
                raise TimeoutError("test did not release handoff worker")
            return original_handoff(*args)

        with patch.object(store, "_handoff_sync", side_effect=commit_after_release):
            task = asyncio.create_task(
                store.handoff(
                    "late-handoff",
                    claim.claim_token,
                    "late-next",
                    _payload("next"),
                    response,
                )
            )
            await self._wait_for_thread_event(started)
            task.cancel("client disconnected")
            try:
                done, _ = await asyncio.wait({task}, timeout=0.5)
                completed_before_commit = task in done
                background_before_commit = len(store._background_workers)
            finally:
                allow_commit.set()
                await asyncio.gather(task, return_exceptions=True)
                await self._drain_background(store)

        self.assertTrue(
            completed_before_commit,
            "cancelled handoff ignored its configured drain deadline",
        )
        with self.assertRaises(asyncio.CancelledError) as raised:
            task.result()
        self.assertEqual(raised.exception.args, ("client disconnected",))
        self.assertEqual(background_before_commit, 1)

        predecessor = await store.inspect("late-handoff")
        successor = await store.inspect("late-next")
        replay = await store.claim("late-handoff", fingerprint)
        self.assertEqual(predecessor.state, "completed")
        self.assertEqual(predecessor.outcome, response)
        self.assertEqual(successor.state, "paused")
        self.assertEqual(successor.payload, _payload("next"))
        self.assertEqual(replay.reason, "completed")
        self.assertEqual(replay.outcome, response)


if __name__ == "__main__":
    unittest.main(verbosity=2)
