"""Bounded PostgreSQL and cancellation contracts for Learning Paths."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from models.learning_path import LearningPath, LearningStage
from services.learning_path_store import LearningPathStore


def _fingerprint(value: dict) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _path(document_id: str, title: str = "Learn binary search") -> LearningPath:
    return LearningPath(
        document_id=document_id,
        title=title,
        total_stages=2,
        stages=[
            LearningStage(
                stage=1,
                title="Understand the invariant",
                topics=["ordering", "boundaries"],
                description="Explain why one half can be discarded.",
                estimated_minutes=20,
            ),
            LearningStage(
                stage=2,
                title="Implement and test",
                topics=["loop invariant", "edge cases"],
                description="Implement the search and test boundary cases.",
                estimated_minutes=30,
            ),
        ],
    )


class TestLearningPathStoreRuntimeBounds(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = str(Path(self.temp_dir.name) / "learning-paths.sqlite3")

    async def asyncTearDown(self) -> None:
        self.temp_dir.cleanup()

    def _store(
        self,
        *,
        cancel_drain_timeout_seconds: float = 0.01,
    ) -> LearningPathStore:
        return LearningPathStore(
            sqlite_path=self.database_path,
            cancel_drain_timeout_seconds=cancel_drain_timeout_seconds,
            clock=lambda: 1_000.0,
        )

    async def _wait_for_thread_event(self, event: threading.Event) -> None:
        self.assertTrue(await asyncio.to_thread(event.wait, 2))

    async def _drain_background(self, store: LearningPathStore) -> None:
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
            "sqlite_busy_timeout_ms",
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
                        LearningPathStore(
                            sqlite_path=self.database_path,
                            **{parameter_name: invalid_value},
                        )

    def test_from_environment_reads_explicit_storage_budgets(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://example/studyloop",
                "LEARNING_PATH_PG_CONNECT_TIMEOUT_SECONDS": "7",
                "LEARNING_PATH_PG_LOCK_TIMEOUT_MS": "1234",
                "LEARNING_PATH_PG_STATEMENT_TIMEOUT_MS": "5678",
                "LEARNING_PATH_PG_TCP_USER_TIMEOUT_MS": "9876",
                "LEARNING_PATH_SCHEMA_INIT_WAIT_TIMEOUT_SECONDS": "9.5",
                "LEARNING_PATH_CANCEL_DRAIN_TIMEOUT_SECONDS": "2.5",
            },
            clear=True,
        ):
            store = LearningPathStore.from_environment()

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
        store = LearningPathStore(
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
        store = LearningPathStore(
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
        store = LearningPathStore(database_url=database_url)

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
                "PostgreSQL Learning Path DATABASE_URL is invalid",
            ) as invalid:
                store._connect()

        self.assertNotIn(secret, str(invalid.exception))
        self.assertIsNone(invalid.exception.__cause__)

    async def test_schema_lock_timeout_and_failure_remain_retryable(self) -> None:
        store = LearningPathStore(
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
            "Learning Path schema initialization timed out",
        ):
            initialization.result()
        self.assertFalse(store._schema_ready)
        await asyncio.to_thread(store._ensure_schema)
        self.assertTrue(store._schema_ready)

        failed = LearningPathStore(sqlite_path=str(Path(self.temp_dir.name) / "failed.sqlite3"))
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
            raise RuntimeError("late Learning Path worker failure")

        with (
            patch.object(store, "_get_sync", side_effect=blocked_worker),
            patch("services.learning_path_store.logger.error") as log_error,
        ):
            task = asyncio.create_task(store.get(f"lp_{'0' * 32}"))
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

    async def test_cancelled_create_preserves_commit_after_drain_deadline(self) -> None:
        store = self._store()
        committed = threading.Event()
        allow_return = threading.Event()
        original_create = store._create_sync
        key = "learning-path-runtime-create"
        request_hash = _fingerprint({"intent": "cancelled create"})

        def commit_then_wait(*args):
            result = original_create(*args)
            committed.set()
            if not allow_return.wait(timeout=5):
                raise TimeoutError("test did not release create worker")
            return result

        with patch.object(store, "_create_sync", side_effect=commit_then_wait):
            task = asyncio.create_task(
                store.create(
                    "runtime-user",
                    "notes.md",
                    _path("notes.md"),
                    idempotency_key=key,
                    request_fingerprint=request_hash,
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
            "cancelled create ignored its configured drain deadline",
        )
        with self.assertRaises(asyncio.CancelledError) as raised:
            task.result()
        self.assertEqual(raised.exception.args, ("client disconnected",))
        self.assertEqual(background_before_release, 1)
        restored = await store.find_by_creation(
            key,
            "runtime-user",
            "notes.md",
            request_hash,
        )
        self.assertIsNotNone(restored)
        replay = await store.create(
            "runtime-user",
            "notes.md",
            _path("notes.md", "must not replace canonical"),
            idempotency_key=key,
            request_fingerprint=request_hash,
        )
        self.assertEqual(replay, restored)

    async def test_cancelled_create_recovers_commit_acknowledgement_loss(self) -> None:
        store = self._store()
        committed = threading.Event()
        allow_failure = threading.Event()
        original_create = store._create_sync
        key = "learning-path-runtime-ack-loss"
        request_hash = _fingerprint({"intent": "commit acknowledgement loss"})

        def commit_then_fail(*args):
            original_create(*args)
            committed.set()
            if not allow_failure.wait(timeout=5):
                raise TimeoutError("test did not release create worker")
            raise RuntimeError("create commit acknowledgement failed")

        with (
            patch.object(store, "_create_sync", side_effect=commit_then_fail),
            patch("services.learning_path_store.logger.error") as log_error,
        ):
            task = asyncio.create_task(
                store.create(
                    "runtime-user",
                    "notes.md",
                    _path("notes.md"),
                    idempotency_key=key,
                    request_fingerprint=request_hash,
                )
            )
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

        self.assertTrue(completed_before_release)
        with self.assertRaises(asyncio.CancelledError) as raised:
            task.result()
        self.assertEqual(raised.exception.args, ("client disconnected",))
        self.assertEqual(background_before_release, 1)
        log_error.assert_called()
        replay = await store.create(
            "runtime-user",
            "notes.md",
            _path("notes.md", "must replay committed candidate"),
            idempotency_key=key,
            request_fingerprint=request_hash,
        )
        self.assertEqual(replay.path.title, "Learn binary search")

    async def test_cancelled_stage_completion_preserves_committed_receipt(self) -> None:
        store = self._store()
        record = await store.create(
            "runtime-user",
            "notes.md",
            _path("notes.md"),
            idempotency_key="learning-path-runtime-stage-path",
            request_fingerprint=_fingerprint({"intent": "stage path"}),
        )
        committed = threading.Event()
        allow_return = threading.Event()
        original_complete = store._complete_stage_sync
        grade_hash = _fingerprint({"quiz": "runtime-stage"})

        def commit_then_wait(*args):
            result = original_complete(*args)
            committed.set()
            if not allow_return.wait(timeout=5):
                raise TimeoutError("test did not release completion worker")
            return result

        with patch.object(
            store,
            "_complete_stage_sync",
            side_effect=commit_then_wait,
        ):
            task = asyncio.create_task(
                store.complete_stage(
                    record.path_id,
                    1,
                    "runtime-stage-quiz",
                    user_id="runtime-user",
                    document_id="notes.md",
                    grading_report_hash=grade_hash,
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
            "cancelled completion ignored its configured drain deadline",
        )
        with self.assertRaises(asyncio.CancelledError) as raised:
            task.result()
        self.assertEqual(raised.exception.args, ("client disconnected",))
        self.assertEqual(background_before_release, 1)
        replay = await store.complete_stage(
            record.path_id,
            1,
            "runtime-stage-quiz",
            user_id="runtime-user",
            document_id="notes.md",
            grading_report_hash=grade_hash,
        )
        self.assertEqual(replay.completed_through, 1)
        self.assertEqual(replay.progress_revision, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
