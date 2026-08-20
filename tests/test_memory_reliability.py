import asyncio
import concurrent.futures
import json
import os
import sys
import tempfile
import threading
import types
import unittest
import uuid
from unittest.mock import AsyncMock, patch

from models.grader import GradingReport, QuestionGrade
from models.quiz import Question
from models.session import QuizSession


def _report(session_id: str, score: float = 1.0) -> GradingReport:
    is_correct = score >= 0.5
    return GradingReport(
        session_id=session_id,
        total=1,
        correct=1 if is_correct else 0,
        score=score,
        grades=[
            QuestionGrade(
                index=0,
                question="测试题",
                user_answer="正确" if is_correct else "错误",
                correct_answer="正确",
                is_correct=is_correct,
                knowledge_gap=None if is_correct else "测试盲点",
            )
        ],
    )


def _clear_user(memory, user_id: str) -> None:
    for bank in memory.SEMANTIC_BANKS + memory.EPISODIC_BANKS:
        namespace = ("users", user_id, bank)
        for item in memory._search_all_items(namespace):
            memory.store.delete(namespace, item.key)


class TestSessionArchive(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.user_id = f"archive-user-{uuid.uuid4()}"

    async def asyncTearDown(self):
        import services.memory as memory

        _clear_user(memory, self.user_id)

    async def test_archives_keep_exact_total_and_recent_raw_sessions(self):
        import services.memory as memory

        for index in range(25):
            await memory.write_episodic_memory(
                self.user_id,
                _report(f"session-{index}", score=(index % 10) / 10),
                "archive.md",
            )
            await memory.maybe_archive_session_briefs(self.user_id)

        profile = await memory.get_user_profile(self.user_id)
        history = await memory.get_user_sessions(self.user_id)
        archives = [item for item in history if item.get("type") == "archive"]
        raw = [item for item in history if item.get("type") != "archive"]

        self.assertEqual(profile["total_sessions"], 25)
        self.assertEqual(profile["average_correct_rate"], 0.4)
        self.assertLessEqual(len(raw), memory.ARCHIVE_THRESHOLD)
        self.assertGreater(len(archives), 1)
        self.assertEqual(
            len(raw) + sum(item["session_count"] for item in archives),
            25,
        )
        archived_ids = [
            session_id
            for item in archives
            for session_id in item["session_ids"]
        ]
        self.assertEqual(len(archived_ids), len(set(archived_ids)))
        self.assertTrue(all(item["date"] for item in archives))

        from services.memory_context import build_returning_context

        returning = await build_returning_context(self.user_id, "archive.md")
        self.assertEqual(returning["session_count"], 25)

    async def test_session_brief_is_immutable_for_request_retries(self):
        import services.memory as memory

        await memory.write_episodic_memory(
            self.user_id,
            _report("immutable-session", score=0.0),
            "archive.md",
        )
        await memory.write_episodic_memory(
            self.user_id,
            _report("immutable-session", score=1.0),
            "archive.md",
        )

        history = await memory.get_user_sessions(self.user_id)
        matching = [
            item for item in history if item.get("session_id") == "immutable-session"
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["correct_rate"], 0.0)

    async def test_semantic_retry_does_not_apply_mastery_ema_twice(self):
        import services.memory as memory

        await memory.update_mastery(self.user_id, "archive.md", 1.0)
        report = _report("partial-semantic", score=0.0)
        with patch.object(
            memory,
            "_append_weak_points_sync",
            side_effect=RuntimeError("weak-point write failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "weak-point write failed"):
                await memory.update_semantic_memory(
                    self.user_id,
                    report,
                    "archive.md",
                )

        self.assertEqual(await memory.get_mastery(self.user_id, "archive.md"), 0.6)
        await memory.update_semantic_memory(
            self.user_id,
            report,
            "archive.md",
        )

        self.assertEqual(await memory.get_mastery(self.user_id, "archive.md"), 0.6)
        self.assertEqual(
            await memory.get_weak_points(self.user_id, "archive.md"),
            ["测试盲点"],
        )
        self.assertNotIn(
            "_applied_sessions",
            await memory.get_mastery(self.user_id, None),
        )

    async def test_optional_audit_failure_does_not_rollback_core_memory(self):
        import services.memory as memory

        core_written = False

        def mark_core_written():
            nonlocal core_written
            core_written = True

        async def fail_audit():
            raise RuntimeError("audit unavailable")

        await memory.commit_learning_memory(
            self.user_id,
            _report("audit-failure", score=1.0),
            "archive.md",
            after_write=fail_audit,
            on_core_written=mark_core_written,
        )

        self.assertTrue(core_written)
        self.assertEqual(await memory.get_mastery(self.user_id, "archive.md"), 1.0)
        self.assertEqual(
            (await memory.get_user_profile(self.user_id))["total_sessions"],
            1,
        )

    async def test_late_retry_of_archived_session_does_not_double_count(self):
        import services.memory as memory

        for index in range(12):
            await memory.write_episodic_memory(
                self.user_id,
                _report(f"session-{index}"),
                "archive.md",
            )
            await memory.maybe_archive_session_briefs(self.user_id)
        history = await memory.get_user_sessions(self.user_id)
        archived_id = next(
            item["session_ids"][0]
            for item in history
            if item.get("type") == "archive"
        )

        await memory.write_episodic_memory(
            self.user_id,
            _report(archived_id),
            "archive.md",
        )
        await memory.maybe_archive_session_briefs(self.user_id)

        profile = await memory.get_user_profile(self.user_id)
        self.assertEqual(profile["total_sessions"], 12)

    async def test_partial_archive_delete_is_hidden_and_cleaned_on_next_pass(self):
        import services.memory as memory

        for index in range(12):
            await memory.write_episodic_memory(
                self.user_id,
                _report(f"session-{index}"),
                "archive.md",
            )
            await memory.maybe_archive_session_briefs(self.user_id)
        history = await memory.get_user_sessions(self.user_id)
        archived_id = next(
            item["session_ids"][0]
            for item in history
            if item.get("type") == "archive"
        )
        namespace = ("users", self.user_id, "session_briefs")
        memory.store.put(
            namespace,
            archived_id,
            {
                "session_id": archived_id,
                "document_id": "archive.md",
                "date": "2026-01-01",
                "correct_rate": 1.0,
                "total_questions": 1,
                "knowledge_gaps": [],
            },
        )

        profile = await memory.get_user_profile(self.user_id)
        visible_history = await memory.get_user_sessions(self.user_id)
        self.assertEqual(profile["total_sessions"], 12)
        self.assertFalse(
            any(
                item.get("type") != "archive"
                and item.get("session_id") == archived_id
                for item in visible_history
            )
        )

        await memory.maybe_archive_session_briefs(self.user_id)
        self.assertIsNone(memory.store.get(namespace, archived_id))


class TestSnapshotCompleteness(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_failure_does_not_publish_core_written_marker(self):
        import services.memory as memory

        marker_calls: list[str] = []
        with (
            patch.object(memory, "DATABASE_URL", None),
            patch.object(memory, "write_episodic_memory", AsyncMock()),
            patch.object(memory, "update_semantic_memory", AsyncMock()),
            patch.object(memory, "maybe_archive_session_briefs", AsyncMock()),
            patch.object(
                memory,
                "persist_memory_snapshot",
                AsyncMock(return_value=False),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "snapshot was not persisted"):
                await memory.commit_learning_memory(
                    "snapshot-failure-user",
                    _report("snapshot-failure-session"),
                    "snapshot.md",
                    on_core_written=lambda: marker_calls.append("published"),
                )

        self.assertEqual(marker_calls, [])

    async def test_cancelled_request_waits_for_memory_commit_to_finish(self):
        import services.memory as memory

        started = asyncio.Event()
        release = asyncio.Event()
        completed = asyncio.Event()

        async def operation():
            started.set()
            await release.wait()
            completed.set()

        task = asyncio.create_task(memory._complete_memory_commit(operation()))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())

        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(completed.is_set())

    async def test_cancelled_request_has_bounded_commit_drain(self):
        import services.memory as memory

        started = asyncio.Event()
        release = asyncio.Event()

        async def operation():
            started.set()
            await release.wait()

        with patch.object(
            memory,
            "_MEMORY_COMMIT_CANCEL_DRAIN_TIMEOUT_SECONDS",
            0.01,
        ):
            task = asyncio.create_task(
                memory._complete_memory_commit(operation())
            )
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertEqual(len(memory._BACKGROUND_MEMORY_COMMITS), 1)
        release.set()
        for _ in range(10):
            if not memory._BACKGROUND_MEMORY_COMMITS:
                break
            await asyncio.sleep(0)
        self.assertFalse(memory._BACKGROUND_MEMORY_COMMITS)

    async def test_store_values_do_not_share_mutable_aliases_with_callers(self):
        import services.memory as memory

        user_id = f"snapshot-alias-{uuid.uuid4()}"
        payload = {"nested": {"items": ["original"]}}
        try:
            await memory.write_bank_state(user_id, "preferences", payload)
            payload["nested"]["items"].append("caller-write")

            first_read = await memory.read_bank_state(user_id, "preferences")
            first_read["nested"]["items"].append("reader-write")
            second_read = await memory.read_bank_state(user_id, "preferences")

            self.assertEqual(second_read, {"nested": {"items": ["original"]}})
        finally:
            _clear_user(memory, user_id)

    async def test_snapshot_pages_all_namespaces_and_items(self):
        import services.memory as memory
        from services.memory_persist import save_snapshot

        prefix = f"snapshot-page-{uuid.uuid4()}"
        users = [f"{prefix}-{index}" for index in range(105)]
        event_user = f"{prefix}-events"
        try:
            for user_id in users:
                memory.store.put(
                    ("users", user_id, "preferences"),
                    "current",
                    {"marker": user_id},
                )
            for index in range(105):
                memory.store.put(
                    ("users", event_user, "error_log"),
                    f"error-{index}",
                    {"marker": index},
                )

            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "memory.json")
                self.assertTrue(save_snapshot(path))
                with open(path, encoding="utf-8") as handle:
                    payload = json.load(handle)

            saved_preferences = {
                record["ns"][1]
                for record in payload["items"]
                if len(record["ns"]) == 3
                and record["ns"][0] == "users"
                and record["ns"][1] in users
                and record["ns"][-1] == "preferences"
            }
            saved_events = [
                record
                for record in payload["items"]
                if record["ns"] == ["users", event_user, "error_log"]
            ]
            self.assertTrue(set(users).issubset(saved_preferences))
            self.assertEqual(len(saved_events), 105)
        finally:
            for user_id in users:
                _clear_user(memory, user_id)
            _clear_user(memory, event_user)

    async def test_completed_quiz_flushes_full_memory_to_local_snapshot(self):
        import services.memory as memory
        from services.session import _write_back_profile

        user_id = f"snapshot-quiz-{uuid.uuid4()}"
        session_id = f"quiz-{uuid.uuid4()}"
        session = QuizSession(
            session_id=session_id,
            document_id="snapshot.md",
            user_id=user_id,
            questions=[
                Question(
                    question="答案是什么？",
                    options=["错误", "正确"],
                    answer="正确",
                    explanation="选择正确。",
                    source="chunk-1",
                    type="choice",
                )
            ],
            user_answers=["正确"],
            status="completed",
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "memory.json")
                with patch.dict(
                    os.environ,
                    {"DATABASE_URL": "", "MEMORY_SNAPSHOT_PATH": path},
                ):
                    await _write_back_profile(session, session_id)

                self.assertTrue(os.path.exists(path))
                with open(path, encoding="utf-8") as handle:
                    payload = json.load(handle)

            user_records = [
                record
                for record in payload["items"]
                if record["ns"][0:2] == ["users", user_id]
            ]
            self.assertTrue(
                any(
                    record["ns"][-1] == "session_briefs"
                    and record["key"] == session_id
                    for record in user_records
                )
            )
            self.assertTrue(
                any(record["ns"][-1] == "mastery" for record in user_records)
            )
        finally:
            _clear_user(memory, user_id)

    async def test_completed_adaptive_turn_flushes_memory_to_local_snapshot(self):
        import routers.adaptive as adaptive
        import services.memory as memory
        from models.adaptive import AdaptiveTurn, NextStepDecision
        from models.adaptive_session import (
            AdaptivePendingSubmit,
            AdaptiveSessionAggregate,
        )
        from services.adaptive_sessions import AdaptiveSessionStore

        user_id = f"snapshot-adaptive-{uuid.uuid4()}"
        adaptive_session_id = f"adapt_{uuid.uuid4().hex}"
        quiz_session_id = f"adaptive:{adaptive_session_id}:turn:1"
        report = _report(quiz_session_id, score=1.0)
        decision = NextStepDecision(
            action="continue",
            topic="持久化",
            count=1,
            reason="继续练习",
        )
        quiz = QuizSession(
            session_id=quiz_session_id,
            document_id="adaptive.md",
            user_id=user_id,
            questions=[
                Question(
                    question="测试题",
                    options=["错误", "正确"],
                    answer="正确",
                    explanation="选择正确。",
                    source="chunk-1",
                    type="choice",
                )
            ],
            user_answers=["正确"],
            status="completed",
            question_grades={0: report.grades[0]},
            grading_report=report,
        )
        artifact = adaptive._artifact(
            session_id=adaptive_session_id,
            turn=1,
            decision=decision,
            trajectory=[
                AdaptiveTurn(
                    turn=1,
                    action="continue",
                    topic="持久化",
                    difficulty_score=0.5,
                )
            ],
            mastery=0.2,
            report=report,
            quiz=quiz,
        )
        aggregate = AdaptiveSessionAggregate(
            adaptive_session_id=adaptive_session_id,
            user_id=user_id,
            document_id="adaptive.md",
            goal="验证持久化",
            current_quiz=quiz,
            last_report=report,
            current_artifact=artifact,
            pending=AdaptivePendingSubmit(
                request_hash="0" * 64,
                turn=1,
                revision=1,
                answers=["正确"],
            ),
        )

        try:
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "adaptive-memory.json")
                store = AdaptiveSessionStore(
                    sqlite_path=os.path.join(directory, "adaptive.sqlite3")
                )
                with (
                    patch.dict(
                        os.environ,
                        {"DATABASE_URL": "", "MEMORY_SNAPSHOT_PATH": path},
                    ),
                    patch.object(adaptive, "adaptive_sessions", store),
                ):
                    await store.create(aggregate)
                    claim = await store.claim(
                        adaptive_session_id,
                        "memory-snapshot-test",
                    )
                    self.assertTrue(claim.claimed)
                    record = await adaptive._write_checkpointed_memory(
                        claim.record.aggregate,
                        claim.record,
                        claim.token,
                        report,
                    )
                    await store.release(adaptive_session_id, claim.token)

                with open(path, encoding="utf-8") as handle:
                    payload = json.load(handle)

            banks = {
                record["ns"][-1]
                for record in payload["items"]
                if record["ns"][0:2] == ["users", user_id]
            }
            self.assertTrue(
                {"session_briefs", "mastery", "decision_log"}.issubset(banks)
            )
            self.assertTrue(record.aggregate.current_quiz.profile_written)
        finally:
            _clear_user(memory, user_id)


class TestPostgresArchiveLock(unittest.TestCase):
    def test_same_user_archives_never_enter_critical_section_together(self):
        import services.memory as memory

        advisory_lock = threading.Lock()
        entered = threading.Event()
        release = threading.Event()

        class Cursor:
            def __init__(self, acquired):
                self.acquired = acquired

            def fetchone(self):
                return (self.acquired,)

        class Connection:
            def __init__(self):
                self.acquired = False

            def __enter__(self):
                return self

            def execute(self, statement, params):
                self.acquired = advisory_lock.acquire(blocking=False)
                return Cursor(self.acquired)

            def __exit__(self, exc_type, exc, traceback):
                if self.acquired:
                    advisory_lock.release()

        def archive(_user_id):
            entered.set()
            self.assertTrue(release.wait(timeout=2))

        fake_psycopg = types.SimpleNamespace(
            connect=lambda *args, **kwargs: Connection()
        )
        with (
            patch.dict(sys.modules, {"psycopg": fake_psycopg}),
            patch.object(
                memory,
                "_bounded_postgres_conninfo",
                side_effect=lambda database_url: database_url,
            ),
            patch.object(memory, "_archive_session_briefs_sync", side_effect=archive) as run,
            concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor,
        ):
            holder = executor.submit(
                memory._archive_session_briefs_with_postgres_lock,
                "same-user",
                "postgresql://memory-test",
            )
            self.assertTrue(entered.wait(timeout=2))
            waiter = executor.submit(
                memory._archive_session_briefs_with_postgres_lock,
                "same-user",
                "postgresql://memory-test",
            )
            self.assertFalse(waiter.done())
            release.set()
            self.assertTrue(holder.result(timeout=2))
            self.assertTrue(waiter.result(timeout=2))

        self.assertEqual(run.call_count, 2)
        run.assert_any_call("same-user")

    def test_archive_lock_wait_is_bounded_and_closes_connection(self):
        import services.memory as memory

        events = []

        class Cursor:
            def fetchone(self):
                return (False,)

        class Connection:
            def __enter__(self):
                events.append("enter")
                return self

            def execute(self, statement, params):
                events.append("try-lock")
                return Cursor()

            def __exit__(self, exc_type, exc, traceback):
                events.append(("exit", exc_type))

        fake_psycopg = types.SimpleNamespace(
            connect=lambda *args, **kwargs: Connection()
        )
        with (
            patch.dict(sys.modules, {"psycopg": fake_psycopg}),
            patch.object(
                memory,
                "_bounded_postgres_conninfo",
                side_effect=lambda database_url: database_url,
            ),
            patch.dict(
                os.environ,
                {"MEMORY_SESSION_ARCHIVE_LOCK_TIMEOUT_SECONDS": "1"},
            ),
            patch.object(memory.time, "monotonic", side_effect=(10.0, 11.0)),
        ):
            with self.assertRaises(memory.PostgresSessionArchiveLockTimeoutError):
                memory._run_with_postgres_session_archive_lock(
                    "busy-user",
                    "postgresql://memory-test",
                    lambda: events.append("operation"),
                )

        self.assertEqual(
            events,
            ["enter", "try-lock", ("exit", memory.PostgresSessionArchiveLockTimeoutError)],
        )

    def test_archive_lock_id_is_stable_per_user_and_separate_from_schema_lock(self):
        import services.memory as memory

        first = memory._session_archive_lock_id("user-a")
        self.assertEqual(first, memory._session_archive_lock_id("user-a"))
        self.assertNotEqual(first, memory._session_archive_lock_id("user-b"))
        self.assertLess(first, 0)
        self.assertNotEqual(first, memory._POSTGRES_STORE_SETUP_LOCK_ID)


if __name__ == "__main__":
    unittest.main()
