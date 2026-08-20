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
        self.table_patch = patch.object(store_module, "_TABLE", self.table_name)
        self.table_patch.start()
        self.store = self._new_store()

    def tearDown(self) -> None:
        import psycopg
        from psycopg import sql

        try:
            with psycopg.connect(
                TEST_DATABASE_URL,
                connect_timeout=5,
            ) as connection:
                connection.execute(
                    sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                        sql.Identifier(self.table_name)
                    )
                )
        finally:
            self.table_patch.stop()

    @staticmethod
    def _new_store() -> LearningPathStore:
        return LearningPathStore(
            database_url=TEST_DATABASE_URL,
            postgres_connect_timeout_seconds=5,
            postgres_lock_timeout_ms=5_000,
            postgres_statement_timeout_ms=15_000,
        )

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
