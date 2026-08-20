"""Optional live PostgreSQL contracts for immutable Learning Path records."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import unittest
import uuid
from unittest.mock import patch

from models.learning_path import LearningPath
import services.learning_path_store as store_module
from services.learning_path_store import LearningPathStore


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

_ADMIN_CONNECT_TIMEOUT_SECONDS = 5
_ADMIN_LOCK_TIMEOUT_MS = 2_000
_ADMIN_STATEMENT_TIMEOUT_MS = 5_000
_ADMIN_TCP_USER_TIMEOUT_MS = 30_000
_FAST_LOCK_TIMEOUT_MS = 100
_FAST_STATEMENT_TIMEOUT_MS = 100
_TIMEOUT_ASSERTION_SECONDS = 6.0


def _path(title: str) -> LearningPath:
    return LearningPath.model_validate(
        {
            "document_id": "notes.md",
            "title": title,
            "total_stages": 1,
            "stages": [
                {
                    "stage": 1,
                    "title": "PostgreSQL 阶段",
                    "topics": ["持久化"],
                    "description": "验证跨连接恢复与唯一键并发。",
                    "estimated_minutes": 15,
                }
            ],
        }
    )


def _fingerprint() -> str:
    canonical = json.dumps(
        {"operation": "postgres-contract", "document_id": "notes.md"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@unittest.skipUnless(TEST_DATABASE_URL, "TEST_DATABASE_URL is not configured")
class TestPostgresLearningPathStore(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # Keep both derived index names below PostgreSQL's 63-byte identifier
        # limit so this contract really creates and exercises both indexes.
        self.table_name = f"sl_lp_{uuid.uuid4().hex[:20]}"
        self.schema_lock_id = uuid.uuid4().int & ((1 << 63) - 1)
        self.table_patch = patch.object(store_module, "_TABLE", self.table_name)
        self.schema_lock_patch = patch.object(
            store_module,
            "_POSTGRES_SCHEMA_LOCK_ID",
            self.schema_lock_id,
        )
        self.table_patch.start()
        self.schema_lock_patch.start()
        self.store = self._new_store()

    def tearDown(self) -> None:
        from psycopg import sql

        try:
            with self._admin_connect() as connection:
                connection.execute(
                    sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                        sql.Identifier(f"{self.table_name}_stage_completions")
                    )
                )
                connection.execute(
                    sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                        sql.Identifier(self.table_name)
                    )
                )
        finally:
            try:
                self.schema_lock_patch.stop()
            finally:
                self.table_patch.stop()

    @staticmethod
    def _admin_connect():
        import psycopg
        from psycopg.conninfo import conninfo_to_dict

        connection_parameters = conninfo_to_dict(TEST_DATABASE_URL)
        if "options" in connection_parameters:
            existing_options = (connection_parameters.get("options") or "").strip()
        else:
            existing_options = os.getenv("PGOPTIONS", "").strip()
        bounded_options = (
            f"-c lock_timeout={_ADMIN_LOCK_TIMEOUT_MS}ms "
            f"-c statement_timeout={_ADMIN_STATEMENT_TIMEOUT_MS}ms"
        )
        return psycopg.connect(
            TEST_DATABASE_URL,
            connect_timeout=_ADMIN_CONNECT_TIMEOUT_SECONDS,
            tcp_user_timeout=_ADMIN_TCP_USER_TIMEOUT_MS,
            options=f"{existing_options} {bounded_options}".strip(),
        )

    @staticmethod
    def _new_store(
        *,
        postgres_lock_timeout_ms: int = 5_000,
        postgres_statement_timeout_ms: int = 15_000,
    ) -> LearningPathStore:
        return LearningPathStore(
            database_url=TEST_DATABASE_URL,
            postgres_connect_timeout_seconds=5,
            postgres_lock_timeout_ms=postgres_lock_timeout_ms,
            postgres_statement_timeout_ms=postgres_statement_timeout_ms,
            postgres_tcp_user_timeout_ms=30_000,
        )

    async def _assert_finishes_with_database_error_while_blocked(
        self,
        operation,
        expected_exception: type[BaseException],
        release_blocker,
    ) -> None:
        task = asyncio.create_task(operation)
        completed_while_blocked = False
        exception = None
        try:
            done, _ = await asyncio.wait(
                {task},
                timeout=_TIMEOUT_ASSERTION_SECONDS,
            )
            completed_while_blocked = task in done
            exception = task.exception() if completed_while_blocked else None
        finally:
            try:
                release_blocker()
            finally:
                await asyncio.gather(task, return_exceptions=True)

        self.assertTrue(
            completed_while_blocked,
            "database operation ignored its configured timeout",
        )
        self.assertIsInstance(exception, expected_exception)

    async def test_reopen_and_current_work_across_connections(self) -> None:
        key = f"learning-path-{uuid.uuid4().hex}"
        created = await self.store.create(
            "default_user",
            "notes.md",
            _path("PostgreSQL 路径"),
            idempotency_key=key,
            request_fingerprint=_fingerprint(),
        )

        reopened = self._new_store()
        self.assertEqual(await reopened.get(created.path_id), created)
        self.assertEqual(
            await reopened.get_current("default_user", "notes.md"),
            created,
        )
        self.assertEqual(
            await reopened.find_by_creation(
                key,
                "default_user",
                "notes.md",
                _fingerprint(),
            ),
            created,
        )

    async def test_source_time_orders_current_and_path_id_breaks_ties(self) -> None:
        path_ids = iter(
            [
                "lp_" + "a" * 32,
                "lp_" + "b" * 32,
                "lp_" + "c" * 32,
                "lp_" + "d" * 32,
            ]
        )
        store = LearningPathStore(
            database_url=TEST_DATABASE_URL,
            postgres_connect_timeout_seconds=5,
            postgres_lock_timeout_ms=5_000,
            postgres_statement_timeout_ms=15_000,
            path_id_factory=lambda: next(path_ids),
        )
        newer = await store.create(
            "default_user",
            "notes.md",
            _path("Newer source"),
            idempotency_key=f"learning-path-{uuid.uuid4().hex}",
            request_fingerprint=hashlib.sha256(b"newer-source").hexdigest(),
            source_created_at=200.0,
        )
        await store.create(
            "default_user",
            "notes.md",
            _path("Delayed older source"),
            idempotency_key=f"learning-path-{uuid.uuid4().hex}",
            request_fingerprint=hashlib.sha256(b"older-source").hexdigest(),
            source_created_at=100.0,
        )
        self.assertEqual(await store.get_current("default_user"), newer)

        tied_first = await store.create(
            "default_user",
            "notes.md",
            _path("Tie C"),
            idempotency_key=f"learning-path-{uuid.uuid4().hex}",
            request_fingerprint=hashlib.sha256(b"tie-c").hexdigest(),
            source_created_at=300.0,
        )
        tied_second = await store.create(
            "default_user",
            "notes.md",
            _path("Tie D"),
            idempotency_key=f"learning-path-{uuid.uuid4().hex}",
            request_fingerprint=hashlib.sha256(b"tie-d").hexdigest(),
            source_created_at=300.0,
        )
        self.assertLess(tied_first.path_id, tied_second.path_id)
        self.assertEqual(await self._new_store().get_current("default_user"), tied_second)

    async def test_concurrent_same_key_selects_one_canonical_record(self) -> None:
        key = f"learning-path-{uuid.uuid4().hex}"
        first, second = await asyncio.gather(
            self.store.create(
                "default_user",
                "notes.md",
                _path("候选一"),
                idempotency_key=key,
                request_fingerprint=_fingerprint(),
            ),
            self._new_store().create(
                "default_user",
                "notes.md",
                _path("候选二"),
                idempotency_key=key,
                request_fingerprint=_fingerprint(),
            ),
        )
        self.assertEqual(first, second)
        self.assertIn(first.path.title, {"候选一", "候选二"})

    async def test_concurrent_stage_completion_advances_once(self) -> None:
        created = await self.store.create(
            "default_user",
            "notes.md",
            _path("PostgreSQL progress"),
            idempotency_key=f"learning-path-{uuid.uuid4().hex}",
            request_fingerprint=_fingerprint(),
        )
        first, second = await asyncio.gather(
            self.store.complete_stage(
                created.path_id,
                1,
                f"quiz-{uuid.uuid4().hex}",
                user_id="default_user",
                document_id="notes.md",
                grading_report_hash=hashlib.sha256(b"grade-a").hexdigest(),
            ),
            self._new_store().complete_stage(
                created.path_id,
                1,
                f"quiz-{uuid.uuid4().hex}",
                user_id="default_user",
                document_id="notes.md",
                grading_report_hash=hashlib.sha256(b"grade-b").hexdigest(),
            ),
        )
        self.assertEqual(first.completed_through, 1)
        self.assertEqual(second.completed_through, 1)
        reopened = self._new_store()
        progress = await reopened.get(created.path_id)
        self.assertEqual(progress.completed_through, 1)
        self.assertEqual(progress.progress_revision, 2)

    async def test_schema_advisory_lock_times_out_then_initialization_retries(
        self,
    ) -> None:
        import psycopg

        store = self._new_store(
            postgres_lock_timeout_ms=_FAST_LOCK_TIMEOUT_MS,
            postgres_statement_timeout_ms=2_000,
        )
        with self._admin_connect() as blocker:
            blocker.execute(
                "SELECT pg_advisory_xact_lock(%s)",
                (self.schema_lock_id,),
            )
            await self._assert_finishes_with_database_error_while_blocked(
                store.get(f"lp_{uuid.uuid4().hex}"),
                psycopg.errors.LockNotAvailable,
                blocker.rollback,
            )

        self.assertIsNone(await store.get(f"lp_{uuid.uuid4().hex}"))

    async def test_uncommitted_creation_key_times_out_then_create_retries(
        self,
    ) -> None:
        import psycopg
        from psycopg import sql

        store = self._new_store(
            postgres_lock_timeout_ms=_FAST_LOCK_TIMEOUT_MS,
            postgres_statement_timeout_ms=2_000,
        )
        self.assertIsNone(await store.get(f"lp_{uuid.uuid4().hex}"))

        key = f"learning-path-{uuid.uuid4().hex}"
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        request_fingerprint = hashlib.sha256(b"unique-index-lock").hexdigest()
        blocked_path_id = f"lp_{uuid.uuid4().hex}"
        blocked_payload = store._serialize_path(_path("未提交创建项"), "notes.md")
        blocked_created_at = 1_000.0
        blocked_immutable_hash = store._immutable_hash(
            blocked_path_id,
            "default_user",
            "notes.md",
            blocked_payload,
            key_hash,
            request_fingerprint,
            blocked_created_at,
        )

        with self._admin_connect() as blocker:
            blocker.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (
                        path_id, schema_version, user_id, document_id,
                        payload_json, status, immutable_hash,
                        idempotency_key_hash, request_fingerprint, created_at
                    ) VALUES (%s, 1, %s, %s, %s, 'ready', %s, %s, %s, %s)
                    """
                ).format(sql.Identifier(self.table_name)),
                (
                    blocked_path_id,
                    "default_user",
                    "notes.md",
                    blocked_payload,
                    blocked_immutable_hash,
                    key_hash,
                    request_fingerprint,
                    blocked_created_at,
                ),
            )
            await self._assert_finishes_with_database_error_while_blocked(
                store.create(
                    "default_user",
                    "notes.md",
                    _path("创建锁重试"),
                    idempotency_key=key,
                    request_fingerprint=request_fingerprint,
                ),
                psycopg.errors.LockNotAvailable,
                blocker.rollback,
            )

        created = await store.create(
            "default_user",
            "notes.md",
            _path("创建锁重试"),
            idempotency_key=key,
            request_fingerprint=request_fingerprint,
        )
        self.assertEqual(created.path.title, "创建锁重试")
        self.assertEqual(
            await store.find_by_creation(
                key,
                "default_user",
                "notes.md",
                request_fingerprint,
            ),
            created,
        )

    async def test_path_row_lock_times_out_then_stage_completion_retries(
        self,
    ) -> None:
        import psycopg
        from psycopg import sql

        store = self._new_store(
            postgres_lock_timeout_ms=_FAST_LOCK_TIMEOUT_MS,
            postgres_statement_timeout_ms=2_000,
        )
        created = await store.create(
            "default_user",
            "notes.md",
            _path("目标行锁"),
            idempotency_key=f"learning-path-{uuid.uuid4().hex}",
            request_fingerprint=hashlib.sha256(b"path-row-lock").hexdigest(),
        )
        quiz_session_id = f"quiz-{uuid.uuid4().hex}"
        grading_report_hash = hashlib.sha256(b"path-row-lock-grade").hexdigest()

        with self._admin_connect() as blocker:
            row = blocker.execute(
                sql.SQL("SELECT path_id FROM {} WHERE path_id = %s FOR UPDATE").format(
                    sql.Identifier(self.table_name)
                ),
                (created.path_id,),
            ).fetchone()
            self.assertEqual(row, (created.path_id,))
            await self._assert_finishes_with_database_error_while_blocked(
                store.complete_stage(
                    created.path_id,
                    1,
                    quiz_session_id,
                    user_id="default_user",
                    document_id="notes.md",
                    grading_report_hash=grading_report_hash,
                ),
                psycopg.errors.LockNotAvailable,
                blocker.rollback,
            )

        progress = await store.complete_stage(
            created.path_id,
            1,
            quiz_session_id,
            user_id="default_user",
            document_id="notes.md",
            grading_report_hash=grading_report_hash,
        )
        self.assertEqual(progress.completed_through, 1)
        self.assertEqual(progress.progress_revision, 2)

    async def test_create_insert_table_lock_times_out_then_create_retries(
        self,
    ) -> None:
        import psycopg
        from psycopg import sql

        store = self._new_store(
            postgres_lock_timeout_ms=_FAST_LOCK_TIMEOUT_MS,
            postgres_statement_timeout_ms=2_000,
        )
        self.assertIsNone(await store.get(f"lp_{uuid.uuid4().hex}"))
        key = f"learning-path-{uuid.uuid4().hex}"
        request_fingerprint = hashlib.sha256(b"create-table-lock").hexdigest()

        with self._admin_connect() as blocker:
            # SHARE permits create's plain preflight SELECT but conflicts with
            # the ROW EXCLUSIVE lock PostgreSQL takes for its actual INSERT.
            blocker.execute(
                sql.SQL("LOCK TABLE {} IN SHARE MODE").format(sql.Identifier(self.table_name))
            )
            await self._assert_finishes_with_database_error_while_blocked(
                store.create(
                    "default_user",
                    "notes.md",
                    _path("建表锁重试"),
                    idempotency_key=key,
                    request_fingerprint=request_fingerprint,
                ),
                psycopg.errors.LockNotAvailable,
                blocker.rollback,
            )

        created = await store.create(
            "default_user",
            "notes.md",
            _path("建表锁重试"),
            idempotency_key=key,
            request_fingerprint=request_fingerprint,
        )
        self.assertEqual(created.path.title, "建表锁重试")

    async def test_statement_timeout_cancels_pg_sleep_then_store_retries(self) -> None:
        import psycopg

        store = self._new_store(
            postgres_lock_timeout_ms=2_000,
            postgres_statement_timeout_ms=_FAST_STATEMENT_TIMEOUT_MS,
        )

        def execute_slow_query() -> None:
            with store._transaction() as connection:
                connection.execute("SELECT pg_sleep(1)").fetchone()

        task = asyncio.create_task(asyncio.to_thread(execute_slow_query))
        done, _ = await asyncio.wait(
            {task},
            timeout=_TIMEOUT_ASSERTION_SECONDS,
        )
        completed = task in done
        exception = task.exception() if completed else None
        await asyncio.gather(task, return_exceptions=True)

        self.assertTrue(completed, "pg_sleep ignored the configured statement timeout")
        self.assertIsInstance(exception, psycopg.errors.QueryCanceled)
        self.assertIsNone(await store.get(f"lp_{uuid.uuid4().hex}"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
