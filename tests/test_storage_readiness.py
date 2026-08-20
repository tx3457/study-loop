"""Storage-readiness service and HTTP contract tests."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import stat
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

import routers.health as health_router
import services.memory as memory
import services.vectorstore as vectorstore
from main import app
from services.storage_readiness import (
    StorageReadinessChecker,
    _probe_local_state_paths,
    _probe_postgres_connection,
    _selected_local_state_paths,
)


def _ready_payload() -> dict:
    return {
        "name": "StudyLoop",
        "status": "ready",
        "code": "storage_ready",
        "checks": {
            "chroma": {
                "status": "ready",
                "required": True,
                "code": "chroma_ready",
            },
            "postgres": {
                "status": "not_configured",
                "required": False,
                "code": "postgres_not_configured",
            },
            "local_state": {
                "status": "ready",
                "required": True,
                "code": "local_state_ready",
            },
        },
    }


def _unready_payload() -> dict:
    payload = _ready_payload()
    payload["status"] = "unready"
    payload["code"] = "storage_unavailable"
    payload["checks"]["chroma"] = {
        "status": "unavailable",
        "required": True,
        "code": "chroma_unavailable",
    }
    return payload


class TestStorageReadinessChecker(unittest.IsolatedAsyncioTestCase):
    async def test_local_mode_requires_chroma_without_importing_postgres(self):
        chroma_probe = Mock()
        memory_probe = Mock()
        local_state_probe = Mock()
        postgres_probe = AsyncMock()
        checker = StorageReadinessChecker(
            chroma_probe=chroma_probe,
            learner_memory_probe=memory_probe,
            local_state_probe=local_state_probe,
            database_url_loader=lambda: None,
            postgres_probe=postgres_probe,
        )

        result = await checker.check()

        self.assertEqual(result, _ready_payload())
        chroma_probe.assert_called_once_with()
        memory_probe.assert_not_called()
        local_state_probe.assert_called_once_with()
        postgres_probe.assert_not_awaited()

    async def test_postgres_mode_checks_new_and_live_store_connections(self):
        calls = []

        async def postgres_probe(url, connect_timeout, statement_timeout):
            calls.append((url, connect_timeout, statement_timeout))

        memory_probe = Mock()
        local_state_probe = Mock()
        checker = StorageReadinessChecker(
            chroma_probe=Mock(),
            learner_memory_probe=memory_probe,
            local_state_probe=local_state_probe,
            database_url_loader=lambda: "postgresql://private@db/studyloop",
            postgres_probe=postgres_probe,
        )

        result = await checker.check()

        self.assertEqual(result["status"], "ready")
        self.assertEqual(
            result["checks"]["postgres"],
            {
                "status": "ready",
                "required": True,
                "code": "postgres_ready",
            },
        )
        self.assertEqual(
            calls,
            [("postgresql://private@db/studyloop", 2, 1_500)],
        )
        memory_probe.assert_called_once_with()
        local_state_probe.assert_not_called()

    async def test_existing_learner_store_failure_keeps_postgres_unready(self):
        secret = "postgresql://user:password@internal-db/studyloop"

        def memory_probe():
            raise RuntimeError(secret)

        async def postgres_probe(*_args):
            return None

        checker = StorageReadinessChecker(
            chroma_probe=Mock(),
            learner_memory_probe=memory_probe,
            database_url_loader=lambda: secret,
            postgres_probe=postgres_probe,
            cache_ttl_seconds=0,
        )

        with self.assertLogs(
            "services.storage_readiness", level="WARNING"
        ) as logs:
            result = await checker.check()

        serialized = json.dumps(result, ensure_ascii=False) + "\n".join(logs.output)
        self.assertEqual(result["status"], "unready")
        self.assertEqual(
            result["checks"]["postgres"]["code"],
            "postgres_unavailable",
        )
        self.assertNotIn(secret, serialized)
        self.assertIn("error_type=RuntimeError", serialized)

    async def test_chroma_failure_is_sanitized(self):
        secret = "D:/private/chroma/password.txt"

        def chroma_probe():
            raise OSError(secret)

        checker = StorageReadinessChecker(
            chroma_probe=chroma_probe,
            local_state_probe=Mock(),
            database_url_loader=lambda: None,
            cache_ttl_seconds=0,
        )

        with self.assertLogs(
            "services.storage_readiness", level="WARNING"
        ) as logs:
            result = await checker.check()

        serialized = json.dumps(result, ensure_ascii=False) + "\n".join(logs.output)
        self.assertEqual(result["status"], "unready")
        self.assertNotIn(secret, serialized)
        self.assertIn("component=chroma error_type=OSError", serialized)

    async def test_hung_chroma_probe_is_retained_instead_of_restarted(self):
        started = threading.Event()
        release = threading.Event()
        calls = 0
        calls_lock = threading.Lock()

        def chroma_probe():
            nonlocal calls
            with calls_lock:
                calls += 1
            started.set()
            release.wait(timeout=2)

        checker = StorageReadinessChecker(
            chroma_probe=chroma_probe,
            local_state_probe=Mock(),
            database_url_loader=lambda: None,
            chroma_timeout_seconds=0.01,
            cache_ttl_seconds=0,
        )
        try:
            first = await checker.check()
            self.assertTrue(started.is_set())
            results = [first]
            for _ in range(5):
                results.append(await checker.check())

            self.assertEqual(
                [result["status"] for result in results],
                ["unready"] * 6,
            )
            self.assertEqual(calls, 1)
            retained = checker._chroma._future
            self.assertIsNotNone(retained)
            self.assertEqual(len(retained._done_callbacks), 1)

            release.set()
            async def wait_until_worker_is_cleared():
                while checker._chroma._future is not None:
                    await asyncio.sleep(0)

            await asyncio.wait_for(
                wait_until_worker_is_cleared(), timeout=0.2
            )
            recovered = await checker.check()
            self.assertEqual(recovered["status"], "ready")
            self.assertEqual(calls, 2)
        finally:
            release.set()

    async def test_cached_payload_is_deep_copied(self):
        chroma_probe = Mock()
        checker = StorageReadinessChecker(
            chroma_probe=chroma_probe,
            local_state_probe=Mock(),
            database_url_loader=lambda: None,
            cache_ttl_seconds=60,
        )

        first = await checker.check()
        first["checks"]["chroma"]["code"] = "caller_mutation"
        second = await checker.check()

        self.assertEqual(second["checks"]["chroma"]["code"], "chroma_ready")
        chroma_probe.assert_called_once_with()

    async def test_local_state_failure_makes_local_mode_unready(self):
        secret = "D:/private/state.sqlite"

        def local_state_probe():
            raise sqlite3.OperationalError(secret)

        checker = StorageReadinessChecker(
            chroma_probe=Mock(),
            local_state_probe=local_state_probe,
            database_url_loader=lambda: None,
            cache_ttl_seconds=0,
        )

        with self.assertLogs(
            "services.storage_readiness", level="WARNING"
        ) as logs:
            result = await checker.check()

        serialized = json.dumps(result, ensure_ascii=False) + "\n".join(logs.output)
        self.assertEqual(result["status"], "unready")
        self.assertEqual(
            result["checks"]["local_state"]["code"],
            "local_state_unavailable",
        )
        self.assertNotIn(secret, serialized)

    async def test_postgres_internal_budget_cancels_a_hung_probe(self):
        cancelled = asyncio.Event()
        calls = 0

        async def postgres_probe(*_args):
            nonlocal calls
            calls += 1
            if calls == 1:
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        checker = StorageReadinessChecker(
            chroma_probe=Mock(),
            learner_memory_probe=Mock(),
            local_state_probe=Mock(),
            database_url_loader=lambda: "postgresql://private@db/studyloop",
            postgres_probe=postgres_probe,
            postgres_timeout_seconds=0.01,
            cache_ttl_seconds=0,
        )

        result = await asyncio.wait_for(checker.check(), timeout=0.2)

        self.assertEqual(result["status"], "unready")
        await asyncio.wait_for(cancelled.wait(), timeout=0.2)
        recovered = await asyncio.wait_for(checker.check(), timeout=0.2)
        self.assertEqual(recovered["status"], "ready")
        self.assertEqual(calls, 2)

    async def test_postgres_cancel_cleanup_cannot_exceed_http_budget(self):
        cancel_seen = asyncio.Event()
        cleanup_release = asyncio.Event()
        calls = 0

        async def postgres_probe(*_args):
            nonlocal calls
            calls += 1
            if calls == 1:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancel_seen.set()
                    await cleanup_release.wait()
                    raise

        checker = StorageReadinessChecker(
            chroma_probe=Mock(),
            learner_memory_probe=Mock(),
            local_state_probe=Mock(),
            database_url_loader=lambda: "postgresql://private@db/studyloop",
            postgres_probe=postgres_probe,
            postgres_timeout_seconds=0.01,
            cache_ttl_seconds=0,
        )

        first = await asyncio.wait_for(checker.check(), timeout=0.2)
        self.assertEqual(first["status"], "unready")
        await asyncio.wait_for(cancel_seen.wait(), timeout=0.2)

        second = await asyncio.wait_for(checker.check(), timeout=0.2)
        self.assertEqual(second["status"], "unready")
        self.assertEqual(calls, 1)

        pending_task = checker._loop_state().postgres._task
        self.assertIsNotNone(pending_task)
        cleanup_release.set()
        await asyncio.wait_for(asyncio.shield(pending_task), timeout=0.2)
        await asyncio.sleep(0)

        recovered = await asyncio.wait_for(checker.check(), timeout=0.2)
        self.assertEqual(recovered["status"], "ready")
        self.assertEqual(calls, 2)

    async def test_caller_cancellation_retains_one_sync_worker(self):
        started = threading.Event()
        release = threading.Event()
        calls = 0

        def chroma_probe():
            nonlocal calls
            calls += 1
            started.set()
            release.wait(timeout=2)

        checker = StorageReadinessChecker(
            chroma_probe=chroma_probe,
            local_state_probe=Mock(),
            database_url_loader=lambda: None,
            chroma_timeout_seconds=1,
            cache_ttl_seconds=0,
        )
        first = asyncio.create_task(checker.check())
        try:
            while not started.is_set():
                await asyncio.sleep(0)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first

            second = asyncio.create_task(checker.check())
            await asyncio.sleep(0.02)
            self.assertEqual(calls, 1)
            release.set()
            result = await asyncio.wait_for(second, timeout=0.2)
            self.assertEqual(result["status"], "ready")
        finally:
            release.set()

    async def test_postgres_probe_uses_driver_timeouts_and_closes(self):
        calls = []

        class FakeCursor:
            async def fetchone(self):
                return (1,)

        class FakeConnection:
            def __init__(self):
                self.closed = False

            async def execute(self, query):
                calls.append(("execute", query))
                return FakeCursor()

            async def close(self):
                self.closed = True

        connection = FakeConnection()

        class FakeAsyncConnection:
            @staticmethod
            async def connect(url, **kwargs):
                calls.append(("connect", url, kwargs))
                return connection

        fake_psycopg = SimpleNamespace(AsyncConnection=FakeAsyncConnection)
        with patch.dict(sys.modules, {"psycopg": fake_psycopg}):
            await _probe_postgres_connection(
                "postgresql://private@db/studyloop",
                connect_timeout_seconds=2,
                statement_timeout_ms=1_500,
            )

        self.assertEqual(
            calls[0],
            (
                "connect",
                "postgresql://private@db/studyloop",
                {
                    "autocommit": True,
                    "connect_timeout": 2,
                    "options": "-c statement_timeout=1500ms",
                },
            ),
        )
        self.assertEqual(calls[1], ("execute", "SELECT 1"))
        self.assertTrue(connection.closed)


class TestStorageReadinessSyncBoundaries(unittest.TestCase):
    def test_top_level_singleflight_is_scoped_to_each_event_loop(self):
        checker = StorageReadinessChecker(
            chroma_probe=Mock(),
            local_state_probe=Mock(),
            database_url_loader=lambda: None,
            cache_ttl_seconds=0,
        )

        async def two_checks():
            results = await asyncio.gather(checker.check(), checker.check())
            self.assertEqual([item["status"] for item in results], ["ready", "ready"])

        asyncio.run(two_checks())
        asyncio.run(two_checks())
        self.assertEqual(len(checker._loop_states), 1)

    def test_retained_chroma_worker_can_cross_event_loops(self):
        started = threading.Event()
        release = threading.Event()
        calls = 0

        def chroma_probe():
            nonlocal calls
            calls += 1
            started.set()
            release.wait(timeout=2)

        checker = StorageReadinessChecker(
            chroma_probe=chroma_probe,
            local_state_probe=Mock(),
            database_url_loader=lambda: None,
            chroma_timeout_seconds=0.01,
            cache_ttl_seconds=0,
        )
        try:
            first = asyncio.run(checker.check())
            second = asyncio.run(checker.check())

            self.assertTrue(started.is_set())
            self.assertEqual(first["status"], "unready")
            self.assertEqual(second["status"], "unready")
            self.assertEqual(calls, 1)
        finally:
            release.set()

    def test_local_state_probe_rejects_an_unopenable_sqlite_parent(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            blocker = Path(directory) / "not-a-directory"
            blocker.write_text("occupied", encoding="utf-8")
            invalid_db = blocker / "state.sqlite3"
            snapshot = Path(directory) / "memory.json"

            with self.assertRaises((OSError, sqlite3.Error)):
                _probe_local_state_paths((invalid_db,), snapshot)

    def test_local_state_probe_opens_sqlite_and_cleans_snapshot_probe(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "nested" / "state.sqlite3"
            snapshot = root / "memory" / "snapshot.json"

            _probe_local_state_paths((database,), snapshot)

            self.assertTrue(database.is_file())
            connection = sqlite3.connect(database)
            try:
                leftover = connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE name LIKE 'studyloop_readiness_%'"
                ).fetchall()
            finally:
                connection.close()
            self.assertEqual(leftover, [])
            self.assertEqual(
                list(snapshot.parent.glob(".studyloop-ready-*")), []
            )

    def test_local_state_probe_rejects_read_only_sqlite_database(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            snapshot = root / "memory.json"
            connection = sqlite3.connect(database)
            try:
                connection.execute("CREATE TABLE durable_state (value INTEGER)")
                connection.commit()
            finally:
                connection.close()
            database.chmod(stat.S_IREAD)
            try:
                with self.assertRaises(sqlite3.Error):
                    _probe_local_state_paths((database,), snapshot)
            finally:
                database.chmod(stat.S_IREAD | stat.S_IWRITE)

    def test_local_state_probe_rejects_snapshot_sqlite_path_collision(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "shared-state"

            with self.assertRaisesRegex(RuntimeError, "conflicts"):
                _probe_local_state_paths((state_path,), state_path)

    def test_local_state_probe_rejects_read_only_existing_snapshot(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            snapshot = root / "memory.json"
            snapshot.write_text("{}", encoding="utf-8")
            snapshot.chmod(stat.S_IREAD)
            try:
                if os.access(snapshot, os.W_OK):
                    self.skipTest("platform can still write a read-only file")
                with self.assertRaises(PermissionError):
                    _probe_local_state_paths((database,), snapshot)
            finally:
                snapshot.chmod(stat.S_IREAD | stat.S_IWRITE)

    def test_local_state_paths_follow_live_feature_overrides(self):
        import tempfile

        import routers.autonomous as autonomous_router
        import services.adaptive_sessions as adaptive_module
        import services.idempotency as idempotency_module
        import services.learning_path_store as learning_path_module
        import services.memory_persist as memory_persist
        import services.quiz_sessions as quiz_module

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {
                root / "receipt.sqlite3",
                root / "path.sqlite3",
                root / "quiz.sqlite3",
                root / "adaptive.sqlite3",
                root / "autonomous.sqlite3",
            }
            with (
                patch.object(
                    idempotency_module.request_idempotency,
                    "_sqlite_path",
                    str(root / "receipt.sqlite3"),
                ),
                patch.object(
                    learning_path_module.learning_path_store,
                    "_sqlite_path",
                    str(root / "path.sqlite3"),
                ),
                patch.object(
                    quiz_module.quiz_sessions,
                    "_sqlite_path",
                    str(root / "quiz.sqlite3"),
                ),
                patch.object(
                    adaptive_module.adaptive_sessions,
                    "_sqlite_path",
                    str(root / "adaptive.sqlite3"),
                ),
                patch.object(
                    autonomous_router.autonomous_sessions,
                    "_sqlite_path",
                    str(root / "autonomous.sqlite3"),
                ),
                patch.object(
                    memory_persist,
                    "snapshot_path",
                    return_value=str(root / "memory.json"),
                ),
            ):
                sqlite_paths, snapshot = _selected_local_state_paths()

        self.assertEqual(set(sqlite_paths), expected)
        self.assertEqual(snapshot, root / "memory.json")


class TestStorageReadinessApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)

    def test_ready_and_unready_use_one_stable_response_shape(self):
        checker = SimpleNamespace(
            check=AsyncMock(side_effect=[_ready_payload(), _unready_payload()])
        )
        provider_checker = SimpleNamespace(check=AsyncMock())

        with (
            patch.object(health_router, "storage_readiness_checker", checker),
            patch.object(health_router, "provider_health_checker", provider_checker),
        ):
            ready = self.client.get("/health/ready")
            unready = self.client.get("/health/ready")

        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json(), _ready_payload())
        self.assertEqual(unready.status_code, 503)
        self.assertEqual(unready.json(), _unready_payload())
        self.assertRegex(ready.headers["X-Request-ID"], r"^req_[0-9a-f]{32}$")
        provider_checker.check.assert_not_awaited()

    def test_liveness_does_not_invoke_storage_checker(self):
        checker = SimpleNamespace(check=AsyncMock())
        with patch.object(health_router, "storage_readiness_checker", checker):
            response = self.client.get("/health/live")

        self.assertEqual(response.status_code, 200)
        checker.check.assert_not_awaited()

    def test_openapi_declares_readiness_200_and_503_envelopes(self):
        operation = app.openapi()["paths"]["/health/ready"]["get"]

        self.assertIn("200", operation["responses"])
        self.assertIn("503", operation["responses"])
        success_schema = operation["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        failure_schema = operation["responses"]["503"]["content"][
            "application/json"
        ]["schema"]
        self.assertEqual(success_schema, failure_schema)

    def test_vectorstore_probe_reads_catalog_without_creating_collection(self):
        client = Mock()
        with patch.object(vectorstore, "chromadb_client", client):
            vectorstore.probe_vectorstore_readiness()

        client.count_collections.assert_called_once_with()
        client.create_collection.assert_not_called()

    def test_learner_memory_probe_reads_current_store(self):
        current_store = Mock()
        current_store.search.return_value = []
        with patch.object(memory, "store", current_store):
            memory.probe_learner_memory_readiness()

        current_store.search.assert_called_once_with(
            ("system", "readiness", "probe"), limit=1
        )
