"""Durability, idempotency, and corruption boundaries for Learning Paths."""

from __future__ import annotations

import asyncio
import hashlib
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

from models.learning_path import LearningPath, LearningStage
from services.learning_path_store import (
    LearningPathCorruptError,
    LearningPathCreationConflictError,
    LearningPathPayloadTooLargeError,
    LearningPathStageConflictError,
    LearningPathStore,
)


def _fingerprint(value: dict) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _path(document_id: str, title: str = "掌握二分查找") -> LearningPath:
    return LearningPath(
        document_id=document_id,
        title=title,
        total_stages=2,
        stages=[
            LearningStage(
                stage=1,
                title="理解有序区间",
                topics=["单调性", "区间边界"],
                description="先确认问题为什么可以排除一半候选区间。",
                estimated_minutes=20,
            ),
            LearningStage(
                stage=2,
                title="实现与验证",
                topics=["循环不变量", "边界测试"],
                description="实现查找并覆盖空数组、单元素和不存在目标等边界。",
                estimated_minutes=30,
            ),
        ],
    )


class TestLearningPathStore(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "learning-paths.sqlite3")
        self.now = [1_000.0]
        self.store = LearningPathStore(
            sqlite_path=self.db_path,
            clock=lambda: self.now[0],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_create_reopen_and_pre_generation_replay(self) -> None:
        key = "learning-path-create-key-1"
        request_hash = _fingerprint(
            {"user_id": "u-1", "document_id": "notes.md", "intent": "binary search"}
        )
        created = await self.store.create(
            "u-1",
            "notes.md",
            _path("notes.md"),
            idempotency_key=key,
            request_fingerprint=request_hash,
        )

        self.assertRegex(created.path_id, r"^lp_[0-9a-f]{32}$")
        self.assertEqual(created.schema_version, 1)
        self.assertEqual(created.user_id, "u-1")
        self.assertEqual(created.document_id, "notes.md")
        self.assertEqual(created.created_at, 1_000.0)
        self.assertEqual(created.path.title, "掌握二分查找")
        self.assertEqual(await self.store.get(created.path_id), created)
        self.assertIsNone(await self.store.get("learning-path-invalid"))

        reopened = LearningPathStore(sqlite_path=self.db_path)
        self.assertEqual(await reopened.get(created.path_id), created)
        self.assertEqual(
            await reopened.find_by_creation(
                key,
                "u-1",
                "notes.md",
                request_hash,
            ),
            created,
        )

        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute(
                """
                SELECT idempotency_key_hash, request_fingerprint, status
                FROM studyloop_learning_paths
                """
            ).fetchone()
            columns = {
                item[1]
                for item in connection.execute(
                    "PRAGMA table_info(studyloop_learning_paths)"
                ).fetchall()
            }
        self.assertEqual(row[0], hashlib.sha256(key.encode("utf-8")).hexdigest())
        self.assertNotEqual(row[0], key)
        self.assertEqual(row[1], request_hash)
        self.assertEqual(row[2], "ready")
        self.assertNotIn("revision", columns)
        self.assertNotIn("updated_at", columns)

    async def test_same_creation_replays_canonical_path_and_mismatch_conflicts(self) -> None:
        key = "learning-path-create-key-2"
        request_hash = _fingerprint({"intent": "first"})
        first = await self.store.create(
            "u-1",
            "notes.md",
            _path("notes.md", "第一个规范结果"),
            idempotency_key=key,
            request_fingerprint=request_hash,
        )
        replay = await self.store.create(
            "u-1",
            "notes.md",
            _path("notes.md", "并发生成但不应覆盖"),
            idempotency_key=key,
            request_fingerprint=request_hash,
        )
        self.assertEqual(replay, first)
        self.assertEqual(replay.path.title, "第一个规范结果")
        invalid_candidate_replay = await self.store.create(
            "u-1",
            "notes.md",
            _path("other.md", "候选绑定无效也不得遮蔽规范结果"),
            idempotency_key=key,
            request_fingerprint=request_hash,
        )
        self.assertEqual(invalid_candidate_replay, first)

        mismatches = [
            ("u-1", "notes.md", _fingerprint({"intent": "different"})),
            ("u-2", "notes.md", request_hash),
            ("u-1", "other.md", request_hash),
        ]
        for user_id, document_id, fingerprint in mismatches:
            with self.subTest(user_id=user_id, document_id=document_id):
                with self.assertRaises(LearningPathCreationConflictError) as raised:
                    await self.store.find_by_creation(
                        key,
                        user_id,
                        document_id,
                        fingerprint,
                    )
                self.assertEqual(raised.exception.reason, "payload_mismatch")

        with closing(sqlite3.connect(self.db_path)) as connection:
            count = connection.execute("SELECT COUNT(*) FROM studyloop_learning_paths").fetchone()[
                0
            ]
        self.assertEqual(count, 1)

    async def test_current_is_scoped_by_user_and_optional_document(self) -> None:
        first = await self.store.create(
            "u-1",
            "a.md",
            _path("a.md", "A1"),
            idempotency_key="current-key-a1",
            request_fingerprint=_fingerprint({"request": "a1"}),
        )
        self.now[0] = 1_001.0
        second = await self.store.create(
            "u-1",
            "b.md",
            _path("b.md", "B1"),
            idempotency_key="current-key-b1",
            request_fingerprint=_fingerprint({"request": "b1"}),
        )
        self.now[0] = 1_002.0
        latest = await self.store.create(
            "u-1",
            "a.md",
            _path("a.md", "A2"),
            idempotency_key="current-key-a2",
            request_fingerprint=_fingerprint({"request": "a2"}),
        )

        self.assertEqual(await self.store.get_current("u-1"), latest)
        self.assertEqual(await self.store.get_current("u-1", "a.md"), latest)
        self.assertEqual(await self.store.get_current("u-1", "b.md"), second)
        self.assertNotEqual(first.path_id, latest.path_id)
        self.assertIsNone(await self.store.get_current("missing-user"))
        self.assertIsNone(await self.store.get_current("u-1", "missing.md"))

    async def test_delayed_source_creation_does_not_replace_a_newer_current_path(
        self,
    ) -> None:
        self.now[0] = 2_000.0
        newer = await self.store.create(
            "u-1",
            "notes.md",
            _path("notes.md", "Newer source path"),
            idempotency_key="source-order-newer",
            request_fingerprint=_fingerprint({"source": "newer"}),
            source_created_at=200.0,
        )

        # This Adaptive result is published later in physical time, but its
        # source terminal state predates the already-created Web path.
        self.now[0] = 3_000.0
        older = await self.store.create(
            "u-1",
            "notes.md",
            _path("notes.md", "Older delayed source path"),
            idempotency_key="source-order-older",
            request_fingerprint=_fingerprint({"source": "older"}),
            source_created_at=100.0,
        )

        self.assertEqual(newer.created_at, 200.0)
        self.assertEqual(older.created_at, 100.0)
        self.assertEqual(await self.store.get_current("u-1"), newer)
        self.assertEqual(
            await self.store.get_current("u-1", "notes.md"),
            newer,
        )

    async def test_equal_source_times_use_path_id_as_a_stable_tie_breaker(
        self,
    ) -> None:
        path_ids = iter(["lp_" + "a" * 32, "lp_" + "b" * 32])
        deterministic = LearningPathStore(
            sqlite_path=self.db_path,
            clock=lambda: self.now[0],
            path_id_factory=lambda: next(path_ids),
        )
        first = await deterministic.create(
            "u-1",
            "notes.md",
            _path("notes.md", "Tie A"),
            idempotency_key="source-tie-a",
            request_fingerprint=_fingerprint({"source": "tie-a"}),
            source_created_at=500.0,
        )
        second = await deterministic.create(
            "u-1",
            "notes.md",
            _path("notes.md", "Tie B"),
            idempotency_key="source-tie-b",
            request_fingerprint=_fingerprint({"source": "tie-b"}),
            source_created_at=500.0,
        )

        self.assertLess(first.path_id, second.path_id)
        self.assertEqual(await deterministic.get_current("u-1"), second)
        reopened = LearningPathStore(sqlite_path=self.db_path)
        self.assertEqual(await reopened.get_current("u-1"), second)

    async def test_source_created_at_is_strict_and_replay_keeps_canonical_time(
        self,
    ) -> None:
        key = "source-time-replay"
        fingerprint = _fingerprint({"source": "replay"})
        first = await self.store.create(
            "u-1",
            "notes.md",
            _path("notes.md", "Canonical source time"),
            idempotency_key=key,
            request_fingerprint=fingerprint,
            source_created_at=123.5,
        )
        replay = await self.store.create(
            "u-1",
            "notes.md",
            _path("notes.md", "Ignored replay candidate"),
            idempotency_key=key,
            request_fingerprint=fingerprint,
            source_created_at=999.0,
        )
        self.assertEqual(replay, first)
        self.assertEqual(replay.created_at, 123.5)

        cases = [True, "123", -1.0, float("nan"), float("inf")]
        for index, value in enumerate(cases):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    await self.store.create(
                        "u-1",
                        "notes.md",
                        _path("notes.md"),
                        idempotency_key=f"invalid-source-time-{index}",
                        request_fingerprint=_fingerprint({"invalid": index}),
                        source_created_at=value,
                    )

        self.now[0] = 777.0
        web = await self.store.create(
            "u-1",
            "notes.md",
            _path("notes.md", "Web clock path"),
            idempotency_key="source-time-web-default",
            request_fingerprint=_fingerprint({"source": "web"}),
        )
        self.assertEqual(web.created_at, 777.0)

    async def test_two_store_instances_concurrently_choose_one_canonical_record(self) -> None:
        peer = LearningPathStore(sqlite_path=self.db_path, clock=lambda: self.now[0])
        key = "learning-path-concurrent-key"
        request_hash = _fingerprint({"intent": "same request"})

        first, second = await asyncio.gather(
            self.store.create(
                "u-1",
                "notes.md",
                _path("notes.md", "candidate one"),
                idempotency_key=key,
                request_fingerprint=request_hash,
            ),
            peer.create(
                "u-1",
                "notes.md",
                _path("notes.md", "candidate two"),
                idempotency_key=key,
                request_fingerprint=request_hash,
            ),
        )

        self.assertEqual(first, second)
        self.assertIn(first.path.title, {"candidate one", "candidate two"})
        with closing(sqlite3.connect(self.db_path)) as connection:
            count = connection.execute("SELECT COUNT(*) FROM studyloop_learning_paths").fetchone()[
                0
            ]
        self.assertEqual(count, 1)

    async def test_stage_completion_is_sequential_durable_and_idempotent(self) -> None:
        record = await self.store.create(
            "u-1",
            "notes.md",
            _path("notes.md"),
            idempotency_key="learning-path-progress-key",
            request_fingerprint=_fingerprint({"intent": "progress"}),
        )
        self.assertEqual(record.completed_through, 0)
        self.assertEqual(record.progress_revision, 1)

        grade_one = _fingerprint({"session": "quiz-1", "score": 0.5})
        first = await self.store.complete_stage(
            record.path_id,
            1,
            "quiz-1",
            user_id="u-1",
            document_id="notes.md",
            grading_report_hash=grade_one,
        )
        replay = await self.store.complete_stage(
            record.path_id,
            1,
            "quiz-1",
            user_id="u-1",
            document_id="notes.md",
            grading_report_hash=grade_one,
        )
        loser = await self.store.complete_stage(
            record.path_id,
            1,
            "quiz-1-loser",
            user_id="u-1",
            document_id="notes.md",
            grading_report_hash=_fingerprint({"session": "quiz-1-loser"}),
        )
        self.assertEqual(first.completed_through, 1)
        self.assertEqual(first.progress_revision, 2)
        self.assertEqual(replay, first)
        self.assertEqual(loser, first)

        self.now[0] = 1_001.0
        second = await self.store.complete_stage(
            record.path_id,
            2,
            "quiz-2",
            user_id="u-1",
            document_id="notes.md",
            grading_report_hash=_fingerprint({"session": "quiz-2", "score": 1.0}),
        )
        self.assertEqual(second.completed_through, 2)
        self.assertEqual(second.progress_revision, 3)
        reopened = LearningPathStore(sqlite_path=self.db_path)
        self.assertEqual(await reopened.get(record.path_id), second)

        with closing(sqlite3.connect(self.db_path)) as connection:
            count = connection.execute(
                """
                SELECT COUNT(*)
                FROM studyloop_learning_paths_stage_completions
                """
            ).fetchone()[0]
        self.assertEqual(count, 2)

    async def test_stage_completion_rejects_locked_or_mismatched_bindings(self) -> None:
        record = await self.store.create(
            "u-1",
            "notes.md",
            _path("notes.md"),
            idempotency_key="learning-path-progress-conflict-key",
            request_fingerprint=_fingerprint({"intent": "progress conflicts"}),
        )
        grade_hash = _fingerprint({"session": "quiz-conflict"})

        cases = [
            (2, "quiz-locked", "u-1", "notes.md", "stage_locked"),
            (1, "quiz-user", "u-2", "notes.md", "path_binding_mismatch"),
            (1, "quiz-doc", "u-1", "other.md", "path_binding_mismatch"),
        ]
        for stage, session_id, user_id, document_id, reason in cases:
            with self.subTest(reason=reason, session_id=session_id):
                with self.assertRaises(LearningPathStageConflictError) as raised:
                    await self.store.complete_stage(
                        record.path_id,
                        stage,
                        session_id,
                        user_id=user_id,
                        document_id=document_id,
                        grading_report_hash=grade_hash,
                    )
                self.assertEqual(raised.exception.reason, reason)

        await self.store.complete_stage(
            record.path_id,
            1,
            "quiz-conflict",
            user_id="u-1",
            document_id="notes.md",
            grading_report_hash=grade_hash,
        )
        with self.assertRaises(LearningPathStageConflictError) as altered:
            await self.store.complete_stage(
                record.path_id,
                1,
                "quiz-conflict",
                user_id="u-1",
                document_id="notes.md",
                grading_report_hash=_fingerprint({"session": "quiz-conflict", "v": 2}),
            )
        self.assertEqual(altered.exception.reason, "completion_binding_mismatch")

    async def test_two_store_instances_complete_one_ready_stage_once(self) -> None:
        record = await self.store.create(
            "u-1",
            "notes.md",
            _path("notes.md"),
            idempotency_key="learning-path-progress-race-key",
            request_fingerprint=_fingerprint({"intent": "progress race"}),
        )
        peer = LearningPathStore(sqlite_path=self.db_path, clock=lambda: self.now[0])
        first, second = await asyncio.gather(
            self.store.complete_stage(
                record.path_id,
                1,
                "quiz-race-a",
                user_id="u-1",
                document_id="notes.md",
                grading_report_hash=_fingerprint({"quiz": "a"}),
            ),
            peer.complete_stage(
                record.path_id,
                1,
                "quiz-race-b",
                user_id="u-1",
                document_id="notes.md",
                grading_report_hash=_fingerprint({"quiz": "b"}),
            ),
        )
        self.assertEqual(first.completed_through, 1)
        self.assertEqual(second.completed_through, 1)
        with closing(sqlite3.connect(self.db_path)) as connection:
            count = connection.execute(
                """
                SELECT COUNT(*)
                FROM studyloop_learning_paths_stage_completions
                """
            ).fetchone()[0]
        self.assertEqual(count, 1)

    async def test_completion_hash_and_contiguous_prefix_fail_closed(self) -> None:
        record = await self.store.create(
            "u-1",
            "notes.md",
            _path("notes.md"),
            idempotency_key="learning-path-progress-corrupt-key",
            request_fingerprint=_fingerprint({"intent": "progress corrupt"}),
        )
        await self.store.complete_stage(
            record.path_id,
            1,
            "quiz-corrupt",
            user_id="u-1",
            document_id="notes.md",
            grading_report_hash=_fingerprint({"quiz": "corrupt"}),
        )
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                """
                UPDATE studyloop_learning_paths_stage_completions
                SET immutable_hash = ? WHERE path_id = ? AND stage_id = 1
                """,
                ("0" * 64, record.path_id),
            )
            connection.commit()
        with self.assertRaises(LearningPathCorruptError):
            await self.store.get(record.path_id)

    async def test_cancellation_drains_committed_create_for_safe_retry(self) -> None:
        committed = threading.Event()
        release = threading.Event()
        original_create = self.store._create_sync
        key = "learning-path-cancel-key"
        request_hash = _fingerprint({"intent": "cancel after commit"})

        def pause_after_commit(*args):
            result = original_create(*args)
            committed.set()
            release.wait(timeout=5)
            return result

        with patch.object(self.store, "_create_sync", side_effect=pause_after_commit):
            task = asyncio.create_task(
                self.store.create(
                    "u-1",
                    "notes.md",
                    _path("notes.md"),
                    idempotency_key=key,
                    request_fingerprint=request_hash,
                )
            )
            self.assertTrue(await asyncio.to_thread(committed.wait, 2))
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        restored = await self.store.find_by_creation(
            key,
            "u-1",
            "notes.md",
            request_hash,
        )
        self.assertIsNotNone(restored)
        replay = await self.store.create(
            "u-1",
            "notes.md",
            _path("notes.md", "must not replace the committed row"),
            idempotency_key=key,
            request_fingerprint=request_hash,
        )
        self.assertEqual(replay, restored)

    async def test_cancellation_drains_committed_stage_for_safe_retry(self) -> None:
        record = await self.store.create(
            "u-1",
            "notes.md",
            _path("notes.md"),
            idempotency_key="learning-path-stage-cancel-key",
            request_fingerprint=_fingerprint({"intent": "cancel stage"}),
        )
        committed = threading.Event()
        release = threading.Event()
        original_complete = self.store._complete_stage_sync
        grade_hash = _fingerprint({"session": "quiz-stage-cancel"})

        def pause_after_commit(*args):
            result = original_complete(*args)
            committed.set()
            release.wait(timeout=5)
            return result

        with patch.object(
            self.store,
            "_complete_stage_sync",
            side_effect=pause_after_commit,
        ):
            task = asyncio.create_task(
                self.store.complete_stage(
                    record.path_id,
                    1,
                    "quiz-stage-cancel",
                    user_id="u-1",
                    document_id="notes.md",
                    grading_report_hash=grade_hash,
                )
            )
            self.assertTrue(await asyncio.to_thread(committed.wait, 2))
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        replay = await self.store.complete_stage(
            record.path_id,
            1,
            "quiz-stage-cancel",
            user_id="u-1",
            document_id="notes.md",
            grading_report_hash=grade_hash,
        )
        self.assertEqual(replay.completed_through, 1)
        self.assertEqual(replay.progress_revision, 2)

    async def test_json_version_hash_and_status_corruption_fail_closed(self) -> None:
        record = await self.store.create(
            "u-1",
            "notes.md",
            _path("notes.md"),
            idempotency_key="learning-path-corrupt-key",
            request_fingerprint=_fingerprint({"intent": "corruption"}),
        )
        with closing(sqlite3.connect(self.db_path)) as connection:
            original = connection.execute(
                """
                SELECT payload_json, schema_version, immutable_hash, status,
                       idempotency_key_hash, request_fingerprint, created_at
                FROM studyloop_learning_paths WHERE path_id = ?
                """,
                (record.path_id,),
            ).fetchone()

        corruptions = [
            ("payload_json", "{"),
            ("schema_version", 2),
            ("immutable_hash", "0" * 64),
            ("status", "draft"),
            ("idempotency_key_hash", "1" * 64),
            ("request_fingerprint", "2" * 64),
            ("created_at", original[6] + 1),
        ]
        originals = dict(
            zip(
                (
                    "payload_json",
                    "schema_version",
                    "immutable_hash",
                    "status",
                    "idempotency_key_hash",
                    "request_fingerprint",
                    "created_at",
                ),
                original,
            )
        )
        for column, corrupt_value in corruptions:
            with self.subTest(column=column):
                with closing(sqlite3.connect(self.db_path)) as connection:
                    connection.execute("PRAGMA ignore_check_constraints = ON")
                    connection.execute(
                        f"UPDATE studyloop_learning_paths SET {column} = ? WHERE path_id = ?",
                        (corrupt_value, record.path_id),
                    )
                    connection.commit()
                with self.assertRaises(LearningPathCorruptError):
                    await self.store.get(record.path_id)
                with closing(sqlite3.connect(self.db_path)) as connection:
                    connection.execute("PRAGMA ignore_check_constraints = ON")
                    connection.execute(
                        f"UPDATE studyloop_learning_paths SET {column} = ? WHERE path_id = ?",
                        (originals[column], record.path_id),
                    )
                    connection.commit()

    async def test_corrupt_latest_record_does_not_fall_back_to_an_older_path(self) -> None:
        await self.store.create(
            "u-1",
            "notes.md",
            _path("notes.md", "older"),
            idempotency_key="learning-path-older-key",
            request_fingerprint=_fingerprint({"request": "older"}),
        )
        self.now[0] = 1_001.0
        latest = await self.store.create(
            "u-1",
            "notes.md",
            _path("notes.md", "latest"),
            idempotency_key="learning-path-latest-key",
            request_fingerprint=_fingerprint({"request": "latest"}),
        )
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "UPDATE studyloop_learning_paths SET payload_json = ? WHERE path_id = ?",
                ("{}", latest.path_id),
            )
            connection.commit()
        with self.assertRaises(LearningPathCorruptError):
            await self.store.get_current("u-1", "notes.md")

    async def test_payload_limit_and_input_validation_happen_before_database_write(self) -> None:
        tiny_path = str(Path(self.temp_dir.name) / "tiny.sqlite3")
        tiny = LearningPathStore(sqlite_path=tiny_path, max_payload_bytes=100)
        with self.assertRaises(LearningPathPayloadTooLargeError):
            await tiny.create(
                "u-1",
                "notes.md",
                _path("notes.md"),
                idempotency_key="learning-path-tiny-key",
                request_fingerprint=_fingerprint({"intent": "too large"}),
            )
        self.assertFalse(Path(tiny_path).exists())

        with self.assertRaises(ValueError):
            await self.store.create(
                "u-1",
                "other.md",
                _path("notes.md"),
                idempotency_key="learning-path-doc-mismatch",
                request_fingerprint=_fingerprint({"intent": "mismatch"}),
            )
        with self.assertRaises(ValueError):
            await self.store.find_by_creation("key", "u-1", "notes.md", "not-a-hash")

    async def test_environment_precedence_and_connection_timeouts(self) -> None:
        local_path = str(Path(self.temp_dir.name) / "learning.sqlite3")
        fallback_path = str(Path(self.temp_dir.name) / "fallback.sqlite3")
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "",
                "LEARNING_PATH_DB_PATH": local_path,
                "IDEMPOTENCY_DB_PATH": fallback_path,
            },
        ):
            local = LearningPathStore.from_environment()
        self.assertEqual(local._sqlite_path, local_path)
        self.assertIsNone(local._database_url)

        captured: dict[str, object] = {}

        class Connection:
            def close(self) -> None:
                captured["closed"] = True

        def connect(database_url: str, **kwargs):
            captured["database_url"] = database_url
            captured.update(kwargs)
            return Connection()

        fake_psycopg = types.SimpleNamespace(connect=connect)
        fake_conninfo = types.SimpleNamespace(conninfo_to_dict=lambda _database_url: {})
        postgres = LearningPathStore(
            database_url="postgresql://example.invalid/studyloop",
            postgres_connect_timeout_seconds=7,
            postgres_lock_timeout_ms=1_234,
            postgres_statement_timeout_ms=5_678,
            postgres_tcp_user_timeout_ms=9_876,
        )
        with (
            patch.dict(
                sys.modules,
                {"psycopg": fake_psycopg, "psycopg.conninfo": fake_conninfo},
            ),
            patch.dict(os.environ, {"PGOPTIONS": ""}, clear=True),
        ):
            connection = postgres._connect()
            connection.close()

        self.assertEqual(
            captured["database_url"],
            "postgresql://example.invalid/studyloop",
        )
        self.assertEqual(captured["connect_timeout"], 7)
        self.assertEqual(captured["tcp_user_timeout"], 9_876)
        self.assertEqual(
            captured["options"],
            "-c lock_timeout=1234ms -c statement_timeout=5678ms",
        )
        self.assertTrue(captured["closed"])

        with closing(self.store._connect()) as sqlite_connection:
            busy_timeout = sqlite_connection.execute("PRAGMA busy_timeout").fetchone()[0]
            foreign_keys = sqlite_connection.execute("PRAGMA foreign_keys").fetchone()[0]
        self.assertEqual(busy_timeout, 10_000)
        self.assertEqual(foreign_keys, 1)


if __name__ == "__main__":
    unittest.main()
