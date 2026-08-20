"""Bounded PostgreSQL and cancellation contracts for receipt storage."""

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

from services.idempotency import (
    BeginDecision,
    IdempotencyConflictError,
    IdempotencyStore,
    ReceiptLease,
    abort_idempotency_claim,
)


class TestIdempotencyRuntimeBounds(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = str(Path(self.temp_dir.name) / "receipts.sqlite3")

    async def asyncTearDown(self):
        self.temp_dir.cleanup()

    async def _wait_for_thread_event(self, event: threading.Event) -> None:
        self.assertTrue(await asyncio.to_thread(event.wait, 2))

    async def _drain_background(self, store: IdempotencyStore) -> None:
        workers = tuple(store._background_workers)
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
            await asyncio.sleep(0)
        self.assertFalse(store._background_workers)

    async def _cancel_after_sync_commit(
        self,
        store: IdempotencyStore,
        sync_method_name: str,
        operation,
    ) -> None:
        committed = threading.Event()
        allow_return = threading.Event()
        original = getattr(store, sync_method_name)

        def commit_then_wait(*args):
            result = original(*args)
            committed.set()
            if not allow_return.wait(timeout=5):
                raise TimeoutError("test did not release committed worker")
            return result

        with patch.object(store, sync_method_name, side_effect=commit_then_wait):
            task = asyncio.create_task(operation())
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
                "cancelled operation ignored its configured drain deadline",
            )
            with self.assertRaises(asyncio.CancelledError) as raised:
                task.result()
            self.assertEqual(raised.exception.args, ("client disconnected",))
            self.assertEqual(background_before_release, 1)

    def test_storage_budgets_must_be_positive_and_finite(self):
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
                        IdempotencyStore(
                            sqlite_path=self.database_path,
                            **{parameter_name: invalid_value},
                        )

    def test_from_environment_reads_explicit_storage_budgets(self):
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://example/studyloop",
                "IDEMPOTENCY_RECEIPT_LEASE_SECONDS": "12.5",
                "IDEMPOTENCY_PG_CONNECT_TIMEOUT_SECONDS": "7",
                "IDEMPOTENCY_PG_LOCK_TIMEOUT_MS": "1234",
                "IDEMPOTENCY_PG_STATEMENT_TIMEOUT_MS": "5678",
                "IDEMPOTENCY_PG_TCP_USER_TIMEOUT_MS": "9876",
                "IDEMPOTENCY_SCHEMA_INIT_WAIT_TIMEOUT_SECONDS": "9.5",
                "IDEMPOTENCY_CANCEL_DRAIN_TIMEOUT_SECONDS": "2.5",
            },
            clear=True,
        ):
            store = IdempotencyStore.from_environment()

        self.assertEqual(store._database_url, "postgresql://example/studyloop")
        self.assertEqual(store._lease_seconds, 12.5)
        self.assertEqual(store._postgres_connect_timeout_seconds, 7)
        self.assertEqual(store._postgres_lock_timeout_ms, 1234)
        self.assertEqual(store._postgres_statement_timeout_ms, 5678)
        self.assertEqual(store._postgres_tcp_user_timeout_ms, 9876)
        self.assertEqual(store._schema_init_wait_timeout_seconds, 9.5)
        self.assertEqual(store._cancel_drain_timeout_seconds, 2.5)

    def test_postgres_connection_preserves_explicit_options_and_adds_bounds(self):
        connect = MagicMock(return_value=SimpleNamespace())
        conninfo_to_dict = MagicMock(
            return_value={
                "options": (
                    "-csearch_path=tenant_schema -c lock_timeout=0 -c statement_timeout=999999999"
                ),
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
        store = IdempotencyStore(
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

    def test_postgres_connection_inherits_pgoptions_without_dsn_options(self):
        connect = MagicMock(return_value=SimpleNamespace())
        conninfo_to_dict = MagicMock(return_value={"dbname": "studyloop"})
        fake_psycopg = ModuleType("psycopg")
        fake_psycopg.connect = connect
        fake_conninfo = ModuleType("psycopg.conninfo")
        fake_conninfo.conninfo_to_dict = conninfo_to_dict
        database_url = "postgresql://example/studyloop"
        store = IdempotencyStore(
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

    def test_hidden_service_options_and_parser_errors_fail_without_secret_leaks(self):
        secret = "private-password-that-must-not-leak"
        database_url = f"service=private-service password={secret}"
        connect = MagicMock(return_value=SimpleNamespace())
        conninfo_to_dict = MagicMock(return_value={"service": "private-service"})
        fake_psycopg = ModuleType("psycopg")
        fake_psycopg.connect = connect
        fake_conninfo = ModuleType("psycopg.conninfo")
        fake_conninfo.conninfo_to_dict = conninfo_to_dict
        store = IdempotencyStore(database_url=database_url)

        with (
            patch.dict(
                sys.modules,
                {"psycopg": fake_psycopg, "psycopg.conninfo": fake_conninfo},
            ),
            patch.dict(os.environ, {}, clear=True),
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
                "PostgreSQL idempotency DATABASE_URL is invalid",
            ) as invalid:
                store._connect()

        self.assertNotIn(secret, str(invalid.exception))
        self.assertIsNone(invalid.exception.__cause__)

        conninfo_to_dict.side_effect = None
        with (
            patch.dict(
                sys.modules,
                {"psycopg": fake_psycopg, "psycopg.conninfo": fake_conninfo},
            ),
            patch.dict(os.environ, {"PGOPTIONS": "   \t"}, clear=True),
        ):
            with self.assertRaisesRegex(ValueError, "must expose connection options"):
                store._connect()

    async def test_schema_lock_timeout_and_initialization_failure_remain_retryable(self):
        store = IdempotencyStore(
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
            "idempotency schema initialization timed out",
        ):
            initialization.result()

        self.assertFalse(store._schema_ready)
        await asyncio.to_thread(store._ensure_schema)
        self.assertTrue(store._schema_ready)

        failed = IdempotencyStore(sqlite_path=str(Path(self.temp_dir.name) / "failed.sqlite3"))
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

    async def test_cancelled_thread_returns_on_deadline_and_consumes_late_failure(self):
        store = IdempotencyStore(
            sqlite_path=self.database_path,
            cancel_drain_timeout_seconds=0.01,
        )
        started = threading.Event()
        allow_finish = threading.Event()
        drain_entered = asyncio.Event()
        original_drain = store._drain_cancelled_worker

        def blocked_worker(*_args):
            started.set()
            if not allow_finish.wait(timeout=5):
                raise TimeoutError("test did not release worker")
            raise RuntimeError("late worker failure")

        async def observe_drain(worker, **kwargs):
            drain_entered.set()
            return await original_drain(worker, **kwargs)

        with (
            patch.object(
                store,
                "_has_effect_started_sync",
                side_effect=blocked_worker,
            ),
            patch.object(
                store,
                "_drain_cancelled_worker",
                side_effect=observe_drain,
            ),
            patch("services.idempotency.logger.error") as log_error,
        ):
            task = asyncio.create_task(store.has_effect_started("key"))
            await self._wait_for_thread_event(started)

            started_at = time.monotonic()
            task.cancel("client disconnected")
            await asyncio.wait_for(drain_entered.wait(), timeout=1)
            task.cancel("shutdown cancellation")
            try:
                done, _ = await asyncio.wait({task}, timeout=0.5)
                completed_before_release = task in done
                elapsed = time.monotonic() - started_at
                background_before_release = len(store._background_workers)
            finally:
                allow_finish.set()
                await asyncio.gather(task, return_exceptions=True)
                await self._drain_background(store)

            self.assertTrue(
                completed_before_release,
                "cancelled worker ignored its configured drain deadline",
            )
            with self.assertRaises(asyncio.CancelledError) as raised:
                task.result()
            self.assertEqual(raised.exception.args, ("client disconnected",))
            self.assertLess(elapsed, 0.5)
            self.assertEqual(background_before_release, 1)

        log_error.assert_called()

    async def test_repeated_cancellation_does_not_extend_the_first_deadline(self):
        store = IdempotencyStore(
            sqlite_path=self.database_path,
            cancel_drain_timeout_seconds=0.08,
        )
        started = threading.Event()
        allow_return = threading.Event()

        def blocked_worker(*_args):
            started.set()
            if not allow_return.wait(timeout=5):
                raise TimeoutError("test did not release worker")
            return False

        with patch.object(
            store,
            "_has_effect_started_sync",
            side_effect=blocked_worker,
        ):
            task = asyncio.create_task(store.has_effect_started("key"))
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

    async def test_cancelled_complete_keeps_canonical_commit_after_drain_timeout(self):
        store = IdempotencyStore(
            sqlite_path=self.database_path,
            cancel_drain_timeout_seconds=0.01,
        )
        payload = {"query": "complete after cancellation"}
        decision = await store.begin("complete-key", "operation", payload)
        response = {"final_answer": "canonical"}

        await self._cancel_after_sync_commit(
            store,
            "_complete_sync",
            lambda: store.complete(decision.lease, response),
        )

        replay = await store.begin("complete-key", "operation", payload)
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.response, response)

    async def test_cancelled_effect_marker_remains_fail_closed_after_drain_timeout(self):
        store = IdempotencyStore(
            sqlite_path=self.database_path,
            cancel_drain_timeout_seconds=0.01,
        )
        payload = {"query": "effect after cancellation"}
        decision = await store.begin("effect-key", "operation", payload)

        await self._cancel_after_sync_commit(
            store,
            "_mark_effect_started_sync",
            lambda: store.mark_effect_started(decision.lease, "stateful_tool"),
        )

        self.assertTrue(await store.has_effect_started("effect-key"))
        with self.assertRaises(IdempotencyConflictError) as raised:
            await store.begin("effect-key", "operation", payload)
        self.assertEqual(raised.exception.reason, "ambiguous")

    async def test_cancelled_abort_preserves_clean_and_effect_started_outcomes(self):
        store = IdempotencyStore(
            sqlite_path=self.database_path,
            cancel_drain_timeout_seconds=0.01,
        )
        clean_payload = {"query": "clean abort after cancellation"}
        clean = await store.begin("abort-clean-key", "operation", clean_payload)

        await self._cancel_after_sync_commit(
            store,
            "_abort_sync",
            lambda: store.abort(clean.lease),
        )

        clean_retry = await store.begin(
            "abort-clean-key",
            "operation",
            {"query": "changed payload is allowed after clean abort"},
        )
        self.assertFalse(clean_retry.replayed)
        await store.abort(clean_retry.lease)

        effect_payload = {"query": "effect abort after cancellation"}
        effect = await store.begin("abort-effect-key", "operation", effect_payload)
        await store.mark_effect_started(effect.lease, "stateful_tool")

        await self._cancel_after_sync_commit(
            store,
            "_abort_sync",
            lambda: store.abort(effect.lease),
        )

        self.assertTrue(await store.has_effect_started("abort-effect-key"))
        with self.assertRaises(IdempotencyConflictError) as raised:
            await store.begin("abort-effect-key", "operation", effect_payload)
        self.assertEqual(raised.exception.reason, "ambiguous")

    async def test_cancelled_late_begin_aborts_exact_old_owner_not_takeover(self):
        now = [0.0]
        first = IdempotencyStore(
            sqlite_path=self.database_path,
            lease_seconds=10,
            cancel_drain_timeout_seconds=0.01,
            clock=lambda: now[0],
        )
        second = IdempotencyStore(
            sqlite_path=self.database_path,
            lease_seconds=10,
            cancel_drain_timeout_seconds=0.01,
            clock=lambda: now[0],
        )
        committed = threading.Event()
        allow_return = threading.Event()
        observed = {}
        aborted_owners = []
        original_begin = first._begin_sync
        original_abort_sync = first._abort_sync

        def commit_then_wait(*args):
            decision = original_begin(*args)
            observed["decision"] = decision
            committed.set()
            if not allow_return.wait(timeout=5):
                raise TimeoutError("test did not release begin worker")
            return decision

        def capture_abort(key, owner_token):
            aborted_owners.append((key, owner_token))
            return original_abort_sync(key, owner_token)

        with (
            patch.object(
                first,
                "_begin_sync",
                side_effect=commit_then_wait,
            ),
            patch.object(first, "_abort_sync", side_effect=capture_abort),
        ):
            task = asyncio.create_task(first.begin("late-key", "operation", {"x": 1}))
            await self._wait_for_thread_event(committed)

            started_at = time.monotonic()
            task.cancel("client disconnected")
            await asyncio.sleep(0)
            task.cancel("shutdown cancellation")
            try:
                done, _ = await asyncio.wait({task}, timeout=0.5)
                completed_before_release = task in done
                elapsed = time.monotonic() - started_at
                background_before_release = len(first._background_workers)

                now[0] = 11.0
                takeover = await second.begin("late-key", "operation", {"x": 1})
                old_lease = observed["decision"].lease
                self.assertNotEqual(old_lease.owner_token, takeover.lease.owner_token)
            finally:
                allow_return.set()
                await asyncio.gather(task, return_exceptions=True)
                await self._drain_background(first)

            self.assertTrue(
                completed_before_release,
                "cancelled begin ignored its configured drain deadline",
            )
            with self.assertRaises(asyncio.CancelledError) as raised:
                task.result()
            self.assertEqual(raised.exception.args, ("client disconnected",))
            self.assertLess(elapsed, 0.5)
            self.assertEqual(background_before_release, 1)

        self.assertEqual(
            aborted_owners,
            [("late-key", observed["decision"].lease.owner_token)],
        )
        with self.assertRaises(IdempotencyConflictError) as blocked:
            await first.begin("late-key", "operation", {"x": 1})
        self.assertEqual(blocked.exception.reason, "in_progress")
        await second.complete(takeover.lease, {"done": True})

    async def test_begin_cleanup_uses_one_deadline_and_preserves_cancellation(self):
        store = IdempotencyStore(
            sqlite_path=self.database_path,
            cancel_drain_timeout_seconds=1,
        )
        started = threading.Event()
        allow_return = threading.Event()
        observed_deadlines = []
        original_drain = store._drain_cancelled_worker

        def delayed_decision(key, _operation, _fingerprint, owner, recovery):
            started.set()
            if not allow_return.wait(timeout=5):
                raise TimeoutError("test did not release begin worker")
            return BeginDecision(
                replayed=False,
                lease=ReceiptLease(key, owner, recovery, 100.0),
            )

        async def observe_deadline(worker, **kwargs):
            observed_deadlines.append(kwargs.get("deadline"))
            return await original_drain(worker, **kwargs)

        with (
            patch.object(
                store,
                "_begin_sync",
                side_effect=delayed_decision,
            ),
            patch.object(
                store,
                "_abort_sync",
                side_effect=RuntimeError("cleanup failed"),
            ),
            patch.object(
                store,
                "_drain_cancelled_worker",
                side_effect=observe_deadline,
            ),
            patch("services.idempotency.logger.error"),
        ):
            task = asyncio.create_task(store.begin("deadline-key", "operation", {}))
            await self._wait_for_thread_event(started)
            task.cancel("client disconnected")
            allow_return.set()
            with self.assertRaises(asyncio.CancelledError) as raised:
                await asyncio.wait_for(task, timeout=1)

        self.assertEqual(raised.exception.args, ("client disconnected",))
        self.assertEqual(len(observed_deadlines), 2)
        self.assertIsNotNone(observed_deadlines[0])
        self.assertEqual(observed_deadlines[0], observed_deadlines[1])
        self.assertFalse(store._background_workers)

    async def test_mismatched_late_begin_decision_never_aborts_foreign_lease(self):
        store = IdempotencyStore(
            sqlite_path=self.database_path,
            cancel_drain_timeout_seconds=0.01,
        )
        started = threading.Event()
        allow_return = threading.Event()
        foreign = ReceiptLease("foreign-key", "foreign-owner", "recovery", 100.0)
        observed = {}
        abort_sync = MagicMock(return_value=False)

        def foreign_decision(_key, _operation, _fingerprint, owner, _recovery):
            observed["owner"] = owner
            started.set()
            if not allow_return.wait(timeout=5):
                raise TimeoutError("test did not release begin worker")
            return BeginDecision(replayed=False, lease=foreign)

        with (
            patch.object(
                store,
                "_begin_sync",
                side_effect=foreign_decision,
            ),
            patch.object(store, "_abort_sync", abort_sync),
        ):
            task = asyncio.create_task(store.begin("expected-key", "operation", {}))
            await self._wait_for_thread_event(started)
            task.cancel()
            try:
                done, _ = await asyncio.wait({task}, timeout=0.5)
                completed_before_release = task in done
            finally:
                allow_return.set()
                await asyncio.gather(task, return_exceptions=True)
                await self._drain_background(store)

        self.assertTrue(
            completed_before_release,
            "cancelled begin ignored its configured drain deadline",
        )
        with self.assertRaises(asyncio.CancelledError):
            task.result()

        abort_sync.assert_called_once_with("expected-key", observed["owner"])

    async def test_cancelled_begin_cleans_commit_after_worker_ack_failure(self):
        store = IdempotencyStore(
            sqlite_path=self.database_path,
            cancel_drain_timeout_seconds=0.01,
        )
        committed = threading.Event()
        allow_failure = threading.Event()
        original_begin = store._begin_sync

        def commit_then_fail(*args):
            original_begin(*args)
            committed.set()
            if not allow_failure.wait(timeout=5):
                raise TimeoutError("test did not release begin worker")
            raise RuntimeError("commit acknowledgement failed")

        with (
            patch.object(
                store,
                "_begin_sync",
                side_effect=commit_then_fail,
            ),
            patch("services.idempotency.logger.error") as log_error,
        ):
            task = asyncio.create_task(store.begin("ack-loss-key", "operation", {"query": "first"}))
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
            "cancelled begin ignored its configured drain deadline",
        )
        with self.assertRaises(asyncio.CancelledError) as raised:
            task.result()
        self.assertEqual(raised.exception.args, ("client disconnected",))
        self.assertEqual(background_before_release, 1)

        log_error.assert_called()
        retry = await store.begin(
            "ack-loss-key",
            "operation",
            {"query": "changed payload"},
        )
        self.assertFalse(retry.replayed)
        await store.abort(retry.lease)

    async def test_outer_abort_cleanup_is_bounded_and_conservative(self):
        store = IdempotencyStore(
            sqlite_path=self.database_path,
            cancel_drain_timeout_seconds=0.01,
        )
        lease = ReceiptLease("abort-key", "owner", "recovery", 100.0)
        started = asyncio.Event()
        allow_finish = asyncio.Event()

        async def blocked_abort(_lease):
            started.set()
            await allow_finish.wait()
            raise RuntimeError("late abort failure")

        with (
            patch.object(store, "abort", side_effect=blocked_abort),
            patch("services.idempotency.logger.error") as log_error,
        ):
            cleanup = asyncio.create_task(abort_idempotency_claim(store, lease))
            await asyncio.wait_for(started.wait(), timeout=1)
            started_at = time.monotonic()
            try:
                done, _ = await asyncio.wait({cleanup}, timeout=0.5)
                completed_before_release = cleanup in done
                elapsed = time.monotonic() - started_at
                background_before_release = len(store._background_workers)
            finally:
                allow_finish.set()
                await asyncio.gather(cleanup, return_exceptions=True)
                await self._drain_background(store)

            self.assertTrue(
                completed_before_release,
                "outer abort cleanup ignored its configured drain deadline",
            )
            self.assertTrue(cleanup.result())
            self.assertLess(elapsed, 0.5)
            self.assertEqual(background_before_release, 1)

        log_error.assert_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
