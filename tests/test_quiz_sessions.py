import asyncio
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from models.quiz import Question
from models.session import (
    AnswerResult,
    LearningPathQuizSource,
    QuizSession,
    QuizSessionAggregate,
)
from services.quiz_sessions import (
    QuizSessionCapacityError,
    QuizSessionClaim,
    QuizSessionCorruptError,
    QuizSessionPayloadTooLargeError,
    QuizSessionStartConflictError,
    QuizSessionStore,
)


def _question(label: str = "A") -> Question:
    return Question(
        question=f"Question {label}",
        options=["A. Alpha", "B. Beta"],
        answer="A",
        explanation="Alpha is correct",
        source=f"chunk-{label}",
        type="choice",
    )


def _aggregate(
    session_id: str,
    *,
    completed: bool = False,
    origin: str = "standard",
) -> QuizSessionAggregate:
    answers = ["A"] if completed else []
    session = QuizSession(
        session_id=session_id,
        document_id="notes.md",
        user_id="user-1",
        questions=[_question(session_id)],
        user_answers=answers,
        status="completed" if completed else "active",
    )
    return QuizSessionAggregate(
        origin=origin,
        session=session,
        last_answer_index=0 if completed else None,
        last_answer_result=(
            AnswerResult(
                correct=True,
                correct_answer="A",
                explanation="Alpha is correct",
                is_last=True,
            )
            if completed
            else None
        ),
    )


class TestQuizSessionStore(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "quiz.sqlite3")
        self.now = [1000.0]
        self.store = QuizSessionStore(
            sqlite_path=self.db_path,
            ttl_seconds=60,
            operation_lease_seconds=10,
            max_count=3,
            clock=lambda: self.now[0],
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    async def _assert_event_is_set(
        self,
        event: threading.Event,
        *,
        timeout: float = 2,
    ) -> None:
        self.assertTrue(
            await asyncio.wait_for(
                asyncio.to_thread(event.wait, timeout),
                timeout=timeout + 0.5,
            )
        )

    async def _drain_background_tasks(
        self,
        tasks: tuple[asyncio.Task, ...],
        *,
        timeout: float = 2,
    ) -> None:
        self.assertTrue(tasks)
        callbacks_completed = asyncio.Event()
        remaining = len(tasks)

        def record_callback(_completed: asyncio.Task) -> None:
            nonlocal remaining
            remaining -= 1
            if remaining == 0:
                callbacks_completed.set()

        for task in tasks:
            task.add_done_callback(record_callback)
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout,
        )
        await asyncio.wait_for(callbacks_completed.wait(), timeout=timeout)

    async def test_reopens_payload_and_replays_start_key(self):
        request = {"document_id": "notes.md", "count": 1}
        created = await self.store.create(
            _aggregate("s-1"),
            start_key="start-key-123",
            start_request=request,
        )
        self.assertTrue(created.created)
        self.assertEqual(created.record.revision, 1)

        peer = QuizSessionStore(
            sqlite_path=self.db_path,
            ttl_seconds=60,
            operation_lease_seconds=10,
            max_count=3,
            clock=lambda: self.now[0],
        )
        replay = await peer.find_start("start-key-123", request)
        self.assertIsNotNone(replay)
        self.assertEqual(replay.aggregate.session.session_id, "s-1")
        self.assertEqual(replay.aggregate.session.questions[0].answer, "A")

        decision = await peer.create(
            _aggregate("s-2"),
            start_key="start-key-123",
            start_request=request,
        )
        self.assertFalse(decision.created)
        self.assertEqual(decision.record.aggregate.session.session_id, "s-1")

        with self.assertRaises(QuizSessionStartConflictError):
            await peer.find_start(
                "start-key-123",
                {"document_id": "other.md", "count": 1},
            )

    def test_optional_path_binding_preserves_legacy_immutable_hash(self):
        aggregate = _aggregate("legacy-hash")
        session = aggregate.session
        canonical = json.dumps(
            {
                "session_id": session.session_id,
                "document_id": session.document_id,
                "user_id": session.user_id,
                "questions": [question.model_dump(mode="json") for question in session.questions],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        legacy_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.assertEqual(self.store._immutable_hash(aggregate), legacy_hash)

        aggregate.learning_path_source = LearningPathQuizSource(
            learning_path_id="lp_00000000000000000000000000000000",
            stage_id=1,
        )
        self.assertNotEqual(self.store._immutable_hash(aggregate), legacy_hash)

    async def test_claim_checkpoint_complete_and_fencing_takeover(self):
        await self.store.create(_aggregate("s-1"))
        first = await self.store.claim("s-1", "grade")
        self.assertTrue(first.claimed)

        peer = QuizSessionStore(
            sqlite_path=self.db_path,
            ttl_seconds=60,
            operation_lease_seconds=10,
            max_count=3,
            clock=lambda: self.now[0],
        )
        blocked = await peer.claim("s-1", "answer")
        self.assertFalse(blocked.claimed)
        self.assertEqual(blocked.reason, "in_progress")

        aggregate = first.record.aggregate.model_copy(deep=True)
        aggregate.session.user_answers.append("A")
        aggregate.session.status = "completed"
        aggregate.last_answer_index = 0
        aggregate.last_answer_result = AnswerResult(
            correct=True,
            correct_answer="A",
            explanation="Alpha is correct",
            is_last=True,
        )
        checkpoint = await self.store.checkpoint("s-1", first.token, aggregate)
        self.assertEqual(checkpoint.revision, 2)
        self.assertTrue(checkpoint.busy)

        self.now[0] += 11
        takeover = await peer.claim("s-1", "report")
        self.assertTrue(takeover.claimed)
        self.assertIsNone(await self.store.complete("s-1", first.token, aggregate))

        completed = await peer.complete("s-1", takeover.token, aggregate)
        self.assertEqual(completed.revision, 3)
        self.assertFalse(completed.busy)
        self.assertEqual(completed.expires_at, self.now[0] + 60)

    async def test_cancelled_claim_releases_ownership_not_returned_to_caller(self):
        await self.store.create(_aggregate("s-cancelled-claim"))
        entered = threading.Event()
        finish = threading.Event()
        original_claim = self.store._claim_sync

        def commit_then_wait(*args):
            decision = original_claim(*args)
            entered.set()
            finish.wait(timeout=2)
            return decision

        with patch.object(self.store, "_claim_sync", side_effect=commit_then_wait):
            task = asyncio.create_task(self.store.claim("s-cancelled-claim", "answer"))
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            task.cancel()
            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        restored = await self.store.inspect("s-cancelled-claim")
        self.assertIsNotNone(restored)
        self.assertFalse(restored.busy)
        retry = await self.store.claim("s-cancelled-claim", "answer")
        self.assertTrue(retry.claimed)
        self.assertTrue(await self.store.release("s-cancelled-claim", retry.token))

    async def test_cancelled_inspect_returns_within_budget_and_drains_in_background(
        self,
    ):
        store = QuizSessionStore(
            sqlite_path=self.db_path,
            cancel_drain_timeout_seconds=0.01,
            clock=lambda: self.now[0],
        )
        started = threading.Event()
        allow_finish = threading.Event()
        finished = threading.Event()

        def blocked_inspect(*args):
            started.set()
            try:
                if not allow_finish.wait(timeout=5):
                    raise TimeoutError("test did not release inspect worker")
                return None
            finally:
                finished.set()

        with patch.object(store, "_inspect_sync", side_effect=blocked_inspect):
            task = asyncio.create_task(store.inspect("s-blocked-inspect"))
            await self._assert_event_is_set(started)

            task.cancel("client disconnected")
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)

            background = tuple(store._background_workers)
            self.assertEqual(len(background), 1)
            allow_finish.set()
            await self._assert_event_is_set(finished)
            await self._drain_background_tasks(background)

        self.assertFalse(store._background_workers)

    async def test_cancelled_late_claim_releases_the_original_generated_token(self):
        store = QuizSessionStore(
            sqlite_path=self.db_path,
            cancel_drain_timeout_seconds=0.01,
            clock=lambda: self.now[0],
        )
        claim_started = threading.Event()
        allow_claim_commit = threading.Event()
        release_called = threading.Event()
        observed: dict[str, object] = {}

        def delayed_claim(session_id, operation, token, now):
            observed["claim"] = (session_id, operation, token, now)
            claim_started.set()
            if not allow_claim_commit.wait(timeout=5):
                raise TimeoutError("test did not release claim worker")
            return QuizSessionClaim(claimed=True, token=token)

        def record_release(session_id, token, now):
            observed["release"] = (session_id, token, now)
            release_called.set()
            return True

        with (
            patch.object(store, "_claim_sync", side_effect=delayed_claim),
            patch.object(store, "_release_sync", side_effect=record_release),
        ):
            task = asyncio.create_task(store.claim("s-late-claim", "answer"))
            await self._assert_event_is_set(claim_started)

            task.cancel("client disconnected")
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)

            background = tuple(store._background_workers)
            self.assertEqual(len(background), 1)
            allow_claim_commit.set()
            await self._assert_event_is_set(release_called)
            await self._drain_background_tasks(background)

        claim_session_id, _, claim_token, _ = observed["claim"]
        release_session_id, release_token, _ = observed["release"]
        self.assertEqual(claim_session_id, "s-late-claim")
        self.assertEqual(release_session_id, claim_session_id)
        self.assertEqual(release_token, claim_token)
        self.assertFalse(store._background_workers)

    async def test_claim_release_failure_does_not_replace_original_cancellation(self):
        store = QuizSessionStore(
            sqlite_path=self.db_path,
            cancel_drain_timeout_seconds=1,
            clock=lambda: self.now[0],
        )
        claim_started = threading.Event()
        allow_claim_commit = threading.Event()
        release_attempted = threading.Event()
        cancellation_handler_entered = asyncio.Event()
        original_drain = store._drain_cancelled_worker

        def delayed_claim(session_id, operation, token, now):
            claim_started.set()
            if not allow_claim_commit.wait(timeout=5):
                raise TimeoutError("test did not release claim worker")
            return QuizSessionClaim(claimed=True, token=token)

        def fail_release(*args):
            release_attempted.set()
            raise RuntimeError("release failed")

        async def observe_cancel_drain(worker, **kwargs):
            cancellation_handler_entered.set()
            return await original_drain(worker, **kwargs)

        with (
            patch.object(store, "_claim_sync", side_effect=delayed_claim),
            patch.object(store, "_release_sync", side_effect=fail_release),
            patch.object(
                store,
                "_drain_cancelled_worker",
                side_effect=observe_cancel_drain,
            ),
            patch("services.quiz_sessions.logger.error"),
        ):
            task = asyncio.create_task(store.claim("s-release-fails", "answer"))
            await self._assert_event_is_set(claim_started)

            task.cancel("client disconnected")
            await asyncio.wait_for(cancellation_handler_entered.wait(), timeout=2)
            allow_claim_commit.set()

            with self.assertRaises(asyncio.CancelledError) as raised:
                await asyncio.wait_for(task, timeout=2)

        self.assertEqual(raised.exception.args, ("client disconnected",))
        await self._assert_event_is_set(release_attempted)
        self.assertTrue(task.cancelled())
        self.assertFalse(store._background_workers)

    async def test_immutable_question_change_cannot_commit(self):
        await self.store.create(_aggregate("s-1"))
        claim = await self.store.claim("s-1", "answer")
        aggregate = claim.record.aggregate.model_copy(deep=True)
        aggregate.session.questions[0].answer = "B"
        self.assertIsNone(await self.store.complete("s-1", claim.token, aggregate))
        self.assertTrue(await self.store.release("s-1", claim.token))

        stored = await self.store.inspect("s-1")
        self.assertEqual(stored.aggregate.session.questions[0].answer, "A")

    async def test_ttl_and_capacity_never_evict_live_active_session(self):
        limited = QuizSessionStore(
            sqlite_path=self.db_path,
            ttl_seconds=60,
            operation_lease_seconds=10,
            max_count=2,
            clock=lambda: self.now[0],
        )
        await limited.create(_aggregate("active-1"))
        await limited.create(_aggregate("active-2"))
        with self.assertRaises(QuizSessionCapacityError):
            await limited.create(_aggregate("active-3"))

        self.now[0] += 61
        expired = await limited.inspect("active-1")
        self.assertTrue(expired.expired)
        claim = await limited.claim("active-1", "answer")
        self.assertFalse(claim.claimed)
        self.assertEqual(claim.reason, "expired")

        created = await limited.create(_aggregate("active-3"))
        self.assertTrue(created.created)
        self.assertIsNone(await limited.inspect("active-1"))

    async def test_operation_cannot_checkpoint_or_revive_session_after_ttl(self):
        await self.store.create(_aggregate("expires-during-operation"))
        self.now[0] = 1059.0
        claim = await self.store.claim("expires-during-operation", "grade")
        self.assertTrue(claim.claimed)

        # The operation lease is still live at 1061, while the session TTL
        # ended at 1060. This isolates the session-expiry fence.
        self.now[0] = 1061.0
        aggregate = claim.record.aggregate.model_copy(deep=True)
        self.assertIsNone(
            await self.store.checkpoint(
                "expires-during-operation",
                claim.token,
                aggregate,
            )
        )
        self.assertIsNone(
            await self.store.complete(
                "expires-during-operation",
                claim.token,
                aggregate,
            )
        )
        self.assertTrue(await self.store.release("expires-during-operation", claim.token))

        expired = await self.store.inspect("expires-during-operation")
        self.assertIsNotNone(expired)
        self.assertTrue(expired.expired)
        self.assertEqual(expired.expires_at, 1060.0)

    async def test_oldest_completed_session_is_capacity_evictable(self):
        limited = QuizSessionStore(
            sqlite_path=self.db_path,
            max_count=2,
            clock=lambda: self.now[0],
        )
        await limited.create(_aggregate("done", completed=True))
        self.now[0] += 1
        await limited.create(_aggregate("active"))
        self.now[0] += 1
        await limited.create(_aggregate("new"))
        self.assertIsNone(await limited.inspect("done"))
        self.assertIsNotNone(await limited.inspect("active"))

    async def test_corrupt_payload_and_unknown_schema_fail_closed(self):
        await self.store.create(_aggregate("s-1"))
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "UPDATE studyloop_quiz_sessions SET payload_json = ? WHERE session_id = ?",
                (json.dumps({"schema_version": 1, "session": {}}), "s-1"),
            )
            connection.commit()
        with self.assertRaises(QuizSessionCorruptError):
            await self.store.inspect("s-1")

        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("PRAGMA ignore_check_constraints = ON")
            connection.execute(
                "UPDATE studyloop_quiz_sessions SET schema_version = 2 WHERE session_id = ?",
                ("s-1",),
            )
            connection.commit()
        with self.assertRaises(QuizSessionCorruptError):
            await self.store.inspect("s-1")

    async def test_payload_size_is_checked_before_database_write(self):
        tiny = QuizSessionStore(
            sqlite_path=self.db_path,
            max_payload_bytes=200,
            clock=lambda: self.now[0],
        )
        with self.assertRaises(QuizSessionPayloadTooLargeError):
            await tiny.create(_aggregate("too-large"))
        self.assertFalse(Path(self.db_path).exists())

    def test_storage_budgets_must_be_positive_and_finite(self):
        parameter_names = (
            "postgres_connect_timeout_seconds",
            "postgres_lock_timeout_ms",
            "postgres_statement_timeout_ms",
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
                        QuizSessionStore(
                            sqlite_path=self.db_path,
                            **{parameter_name: invalid_value},
                        )

    def test_from_environment_reads_explicit_storage_budgets(self):
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://example/studyloop",
                "QUIZ_SESSION_PG_CONNECT_TIMEOUT_SECONDS": "7",
                "QUIZ_SESSION_PG_LOCK_TIMEOUT_MS": "1234",
                "QUIZ_SESSION_PG_STATEMENT_TIMEOUT_MS": "5678",
                "QUIZ_SESSION_SCHEMA_INIT_WAIT_TIMEOUT_SECONDS": "9.5",
                "QUIZ_SESSION_CANCEL_DRAIN_TIMEOUT_SECONDS": "2.5",
            },
            clear=True,
        ):
            store = QuizSessionStore.from_environment()

        self.assertEqual(store._database_url, "postgresql://example/studyloop")
        self.assertEqual(store._postgres_connect_timeout_seconds, 7)
        self.assertEqual(store._postgres_lock_timeout_ms, 1234)
        self.assertEqual(store._postgres_statement_timeout_ms, 5678)
        self.assertEqual(store._schema_init_wait_timeout_seconds, 9.5)
        self.assertEqual(store._cancel_drain_timeout_seconds, 2.5)

    def test_postgres_connection_preserves_options_and_adds_timeouts(self):
        connect = MagicMock(return_value=SimpleNamespace())
        conninfo_to_dict = MagicMock(return_value={"options": "-csearch_path=tenant_schema"})
        fake_psycopg = ModuleType("psycopg")
        fake_psycopg.connect = connect
        fake_conninfo = ModuleType("psycopg.conninfo")
        fake_conninfo.conninfo_to_dict = conninfo_to_dict
        database_url = "postgresql://example/studyloop?options=-csearch_path%3Dtenant_schema"
        store = QuizSessionStore(
            database_url=database_url,
            postgres_connect_timeout_seconds=7,
            postgres_lock_timeout_ms=1234,
            postgres_statement_timeout_ms=5678,
        )

        with (
            patch.dict(
                sys.modules,
                {"psycopg": fake_psycopg, "psycopg.conninfo": fake_conninfo},
            ),
            patch.dict(os.environ, {"PGOPTIONS": "-csearch_path=ignored_env_schema"}),
        ):
            connection = store._connect()

        self.assertIs(connection, connect.return_value)
        conninfo_to_dict.assert_called_once_with(database_url)
        connect.assert_called_once_with(
            database_url,
            connect_timeout=7,
            options=(
                "-csearch_path=tenant_schema -c lock_timeout=1234ms -c statement_timeout=5678ms"
            ),
        )

    def test_postgres_connection_inherits_pgoptions_when_dsn_has_none(self):
        connect = MagicMock(return_value=SimpleNamespace())
        conninfo_to_dict = MagicMock(return_value={"dbname": "studyloop"})
        fake_psycopg = ModuleType("psycopg")
        fake_psycopg.connect = connect
        fake_conninfo = ModuleType("psycopg.conninfo")
        fake_conninfo.conninfo_to_dict = conninfo_to_dict
        database_url = "postgresql://example/studyloop"
        store = QuizSessionStore(
            database_url=database_url,
            postgres_connect_timeout_seconds=7,
            postgres_lock_timeout_ms=1234,
            postgres_statement_timeout_ms=5678,
        )

        with (
            patch.dict(
                sys.modules,
                {"psycopg": fake_psycopg, "psycopg.conninfo": fake_conninfo},
            ),
            patch.dict(os.environ, {"PGOPTIONS": "-csearch_path=environment_schema"}),
        ):
            connection = store._connect()

        self.assertIs(connection, connect.return_value)
        conninfo_to_dict.assert_called_once_with(database_url)
        connect.assert_called_once_with(
            database_url,
            connect_timeout=7,
            options=(
                "-csearch_path=environment_schema "
                "-c lock_timeout=1234ms -c statement_timeout=5678ms"
            ),
        )

    def test_schema_python_lock_timeout_leaves_initialization_retryable(self):
        store = QuizSessionStore(
            sqlite_path=self.db_path,
            schema_init_wait_timeout_seconds=0.01,
            clock=lambda: self.now[0],
        )
        self.assertTrue(store._schema_lock.acquire(blocking=False))
        try:
            with self.assertRaisesRegex(
                TimeoutError,
                "quiz session schema initialization timed out",
            ):
                store._ensure_schema()
        finally:
            store._schema_lock.release()

        self.assertFalse(store._schema_ready)
        store._ensure_schema()
        self.assertTrue(store._schema_ready)


class TestQuizSessionAggregate(unittest.TestCase):
    def test_rejects_inconsistent_status_and_feedback(self):
        session = QuizSession(
            session_id="broken",
            document_id="notes.md",
            user_id="u",
            questions=[_question()],
            user_answers=["A"],
            status="active",
        )
        with self.assertRaises(ValueError):
            QuizSessionAggregate(session=session)

        session.status = "completed"
        with self.assertRaises(ValueError):
            QuizSessionAggregate(session=session)


if __name__ == "__main__":
    unittest.main()
