"""Durability, fencing, and retry contracts for Adaptive HTTP routes."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from main import app
from models.adaptive import NextStepDecision
from models.grader import QuestionGrade
from models.learning_path import LearningPath
from models.quiz import Question, QuizResponse
import routers.adaptive as adaptive_router
import routers.learning_path as learning_path_router
from services.adaptive_sessions import AdaptiveSessionStore
from services.grader import grade_quiz_session as real_grade_quiz_session
from services.learning_path_store import LearningPathStore


def _decision(action: str = "continue") -> NextStepDecision:
    return NextStepDecision(
        action=action,
        topic="sorting",
        count=1,
        reason=f"test-{action}",
    )


def _quiz() -> QuizResponse:
    return QuizResponse(
        questions=[
            Question(
                question="Is merge sort stable?",
                options=["A", "B"],
                answer="A",
                explanation="Equal elements can retain their order.",
                source="private-source",
                type="choice",
            )
        ]
    )


def _path(document_id: str = "notes.md", title: str = "Sorting path") -> LearningPath:
    return LearningPath.model_validate(
        {
            "document_id": document_id,
            "title": title,
            "total_stages": 1,
            "stages": [
                {
                    "stage": 1,
                    "title": "Foundations",
                    "topics": ["merge sort"],
                    "description": "Learn stable divide-and-conquer sorting.",
                    "estimated_minutes": 20,
                }
            ],
        }
    )


class TestAdaptiveDurableRoutes(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database_path = str(Path(self.tempdir.name) / "adaptive.sqlite3")
        self.store = AdaptiveSessionStore(sqlite_path=self.database_path)
        self.path_store = LearningPathStore(sqlite_path=self.database_path)
        self.store_patch = patch.object(adaptive_router, "adaptive_sessions", self.store)
        self.path_store_patch = patch.object(
            adaptive_router, "learning_path_store", self.path_store
        )
        self.store_patch.start()
        self.path_store_patch.start()

        async def commit_memory(*args, after_write=None, on_core_written=None, **kwargs) -> None:
            if on_core_written is not None:
                on_core_written()
            if after_write is not None:
                await after_write()

        self.generate = AsyncMock(return_value=_quiz())
        self.decide = AsyncMock(return_value=_decision())
        self.mastery = AsyncMock(return_value=0.2)
        self.memory = AsyncMock(side_effect=commit_memory)
        self.append = AsyncMock()
        self.patches = [
            patch.object(adaptive_router, "ensure_document_available", AsyncMock()),
            patch.object(
                adaptive_router,
                "check_injection",
                AsyncMock(return_value=(False, "")),
            ),
            patch.object(adaptive_router, "generate_question", self.generate),
            patch.object(adaptive_router, "decide_next_step", self.decide),
            patch.object(adaptive_router, "get_mastery", self.mastery),
            patch.object(adaptive_router, "get_weak_points", AsyncMock(return_value=[])),
            patch.object(adaptive_router, "commit_learning_memory", self.memory),
            patch.object(adaptive_router, "append_decision", self.append),
        ]
        for active_patch in self.patches:
            active_patch.start()
        self.client = TestClient(app, raise_server_exceptions=False)

    def tearDown(self) -> None:
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.path_store_patch.stop()
        self.store_patch.stop()
        self.tempdir.cleanup()

    def _start(self, key: str | None = None) -> dict:
        headers = {"Idempotency-Key": key} if key else None
        response = self.client.post(
            "/agent/adaptive/start",
            json={
                "user_id": "user-1",
                "document_id": "notes.md",
                "goal": "learn sorting",
            },
            headers=headers,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    @staticmethod
    def _submit_payload(start: dict) -> dict:
        return {
            "adaptive_session_id": start["adaptive_session_id"],
            "turn": start["turn"],
            "revision": start["revision"],
            "answers": ["A"],
        }

    def test_start_key_replays_and_rejects_payload_mismatch(self) -> None:
        first = self._start("adaptive-start-key-1")
        replay = self._start("adaptive-start-key-1")

        self.assertEqual(replay, first)
        self.assertEqual(self.generate.await_count, 1)
        self.assertEqual(self.decide.await_count, 1)

        mismatch = self.client.post(
            "/agent/adaptive/start",
            json={
                "user_id": "user-1",
                "document_id": "notes.md",
                "goal": "a different goal",
            },
            headers={"Idempotency-Key": "adaptive-start-key-1"},
        )
        self.assertEqual(mismatch.status_code, 409, mismatch.text)
        self.assertEqual(mismatch.json()["code"], "idempotency_conflict")
        self.assertEqual(mismatch.json()["reason"], "payload_mismatch")
        self.assertEqual(self.generate.await_count, 1)

    def test_snapshot_is_safe_and_exposes_storage_metadata(self) -> None:
        start = self._start()
        response = self.client.get(f"/agent/adaptive/{start['adaptive_session_id']}")

        self.assertEqual(response.status_code, 200, response.text)
        snapshot = response.json()
        self.assertEqual(snapshot["revision"], start["revision"])
        self.assertFalse(snapshot["busy"])
        self.assertGreater(snapshot["expires_at"], 0)
        self.assertEqual(
            set(snapshot["questions"][0]),
            {"index", "question", "options", "type"},
        )
        self.assertNotIn('"explanation"', response.text)
        self.assertNotIn('"source"', response.text)
        self.assertNotIn('"answer":', response.text)

    def test_start_rejects_whitespace_padded_question_over_public_limit(self) -> None:
        padded = _quiz()
        padded.questions[0].question = " " * 100 + "q" * 4_000
        self.generate.return_value = padded

        response = self.client.post(
            "/agent/adaptive/start",
            json={
                "user_id": "user-1",
                "document_id": "notes.md",
                "goal": "learn sorting",
            },
        )

        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(self.generate.await_count, 1)

    def test_opening_finish_completes_without_generating_a_quiz(self) -> None:
        self.decide.return_value = _decision("finish")
        start = self._start()

        self.assertTrue(start["done"])
        self.assertEqual(start["terminate_reason"], "agent_finish")
        self.assertEqual(start["questions"], [])
        self.assertEqual(self.generate.await_count, 0)

    def test_opening_switch_to_plan_returns_a_terminal_path(self) -> None:
        fake_path = _path()
        self.mastery.return_value = 0.99
        self.decide.return_value = _decision("switch_to_plan")
        generate_path = AsyncMock(return_value=fake_path)
        original_create = self.path_store.create

        async def create_after_terminal(*args, **kwargs):
            stored = await self.store.find_start(
                "adaptive-opening-path-1",
                {
                    "user_id": "user-1",
                    "document_id": "notes.md",
                    "goal": "learn sorting",
                },
            )
            self.assertIsNotNone(stored)
            self.assertEqual(stored.aggregate.status, "completed")
            self.assertFalse(stored.busy)
            return await original_create(*args, **kwargs)

        with (
            patch.object(
                adaptive_router,
                "generate_learning_path",
                generate_path,
            ),
            patch.object(
                self.path_store,
                "create",
                side_effect=create_after_terminal,
            ),
        ):
            start = self._start("adaptive-opening-path-1")
            replay = self._start("adaptive-opening-path-1")

        self.assertTrue(start["done"])
        self.assertEqual(start["terminate_reason"], "switch_to_plan")
        self.assertEqual(start["learning_path"], fake_path.model_dump(mode="json"))
        self.assertRegex(start["learning_path_id"], r"^lp_[0-9a-f]{32}$")
        self.assertEqual(replay, start)
        self.assertEqual(generate_path.await_count, 1)
        self.assertEqual(self.generate.await_count, 0)
        published = self.path_store._get_sync(start["learning_path_id"])
        self.assertEqual(published.user_id, "user-1")
        with patch.object(
            learning_path_router,
            "learning_path_store",
            self.path_store,
        ):
            fetched = self.client.get(f"/learning-paths/{start['learning_path_id']}")
        self.assertEqual(fetched.status_code, 200, fetched.text)
        self.assertEqual(fetched.json()["user_id"], "user-1")

    def test_switch_to_plan_precedes_mastery_termination(self) -> None:
        fake_path = _path(title="Review path")
        self.mastery.side_effect = [0.2, 0.99]
        self.decide.side_effect = [_decision(), _decision("switch_to_plan")]
        start = self._start()
        payload = self._submit_payload(start)
        headers = {"Idempotency-Key": "adaptive-submit-path-1"}
        generate_path = AsyncMock(return_value=fake_path)

        with patch.object(
            adaptive_router,
            "generate_learning_path",
            generate_path,
        ):
            switched = self.client.post(
                "/agent/adaptive/submit",
                json=payload,
                headers=headers,
            )
            replay = self.client.post(
                "/agent/adaptive/submit",
                json=payload,
                headers=headers,
            )

        self.assertEqual(switched.status_code, 200, switched.text)
        self.assertTrue(switched.json()["done"])
        self.assertEqual(switched.json()["terminate_reason"], "switch_to_plan")
        self.assertEqual(
            switched.json()["learning_path"],
            fake_path.model_dump(mode="json"),
        )
        self.assertRegex(
            switched.json()["learning_path_id"],
            r"^lp_[0-9a-f]{32}$",
        )
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(replay.json(), switched.json())
        self.assertEqual(generate_path.await_count, 1)
        self.assertEqual(self.memory.await_count, 1)

    def test_completed_submit_replays_after_sqlite_reopen(self) -> None:
        self.decide.side_effect = [_decision(), _decision("finish")]
        start = self._start()
        payload = self._submit_payload(start)
        headers = {"Idempotency-Key": "adaptive-submit-replay-1"}

        first = self.client.post("/agent/adaptive/submit", json=payload, headers=headers)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertTrue(first.json()["done"])

        reopened = AdaptiveSessionStore(sqlite_path=self.database_path)
        with patch.object(adaptive_router, "adaptive_sessions", reopened):
            replay = self.client.post("/agent/adaptive/submit", json=payload, headers=headers)

        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(replay.json(), first.json())
        self.assertEqual(self.memory.await_count, 1)
        self.assertEqual(self.append.await_count, 1)
        self.assertEqual(self.decide.await_count, 2)

    def test_receipt_race_returns_settled_storage_metadata(self) -> None:
        self.decide.side_effect = [_decision(), _decision("finish")]
        start = self._start()
        stale_record = self.store._inspect_sync(start["adaptive_session_id"])
        payload = self._submit_payload(start)
        headers = {"Idempotency-Key": "adaptive-receipt-race-1"}
        first = self.client.post("/agent/adaptive/submit", json=payload, headers=headers)
        self.assertEqual(first.status_code, 200, first.text)

        original_inspect = adaptive_router._inspect_live
        inspect_count = 0

        async def stale_optimistic_inspect(session_id: str):
            nonlocal inspect_count
            inspect_count += 1
            if inspect_count == 1:
                return stale_record, stale_record.aggregate.model_copy(deep=True)
            return await original_inspect(session_id)

        with patch.object(
            adaptive_router,
            "_inspect_live",
            side_effect=stale_optimistic_inspect,
        ):
            replay = self.client.post("/agent/adaptive/submit", json=payload, headers=headers)

        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(replay.json(), first.json())
        self.assertFalse(replay.json()["busy"])
        self.assertGreater(replay.json()["revision"], stale_record.revision)

    def test_partial_grade_checkpoint_survives_failure_and_reopen(self) -> None:
        self.decide.side_effect = [_decision(), _decision("finish")]
        start = self._start()
        payload = self._submit_payload(start)
        headers = {"Idempotency-Key": "adaptive-partial-grade-1"}
        grade_calls = 0

        async def flaky_grade(session, *, checkpoint=None):
            nonlocal grade_calls
            grade_calls += 1
            if grade_calls == 1:
                question = session.questions[0]
                session.question_grades[0] = QuestionGrade(
                    index=0,
                    question=question.question,
                    user_answer=session.user_answers[0],
                    correct_answer=question.answer,
                    is_correct=True,
                )
                await checkpoint()
                raise RuntimeError("grader disconnected after checkpoint")
            return await real_grade_quiz_session(session, checkpoint=checkpoint)

        with patch.object(adaptive_router, "grade_quiz_session", side_effect=flaky_grade):
            failed = self.client.post("/agent/adaptive/submit", json=payload, headers=headers)
            self.assertEqual(failed.status_code, 500, failed.text)

            record = self.store._inspect_sync(start["adaptive_session_id"])
            self.assertIsNotNone(record.aggregate.pending)
            self.assertEqual(record.aggregate.current_quiz.user_answers, ["A"])
            self.assertIn(0, record.aggregate.current_quiz.question_grades)
            self.assertIsNone(record.aggregate.current_quiz.grading_report)
            self.assertFalse(record.busy)

            reopened = AdaptiveSessionStore(sqlite_path=self.database_path)
            with patch.object(adaptive_router, "adaptive_sessions", reopened):
                recovered = self.client.post(
                    "/agent/adaptive/submit", json=payload, headers=headers
                )

        self.assertEqual(recovered.status_code, 200, recovered.text)
        self.assertTrue(recovered.json()["done"])
        self.assertEqual(grade_calls, 2)
        self.assertEqual(self.memory.await_count, 1)

    def test_decision_checkpoint_resumes_without_repeating_side_effects(self) -> None:
        self.generate.side_effect = [
            _quiz(),
            RuntimeError("question provider disconnected"),
            _quiz(),
        ]
        start = self._start()
        payload = self._submit_payload(start)
        headers = {"Idempotency-Key": "adaptive-stage-recovery-1"}

        failed = self.client.post("/agent/adaptive/submit", json=payload, headers=headers)
        self.assertEqual(failed.status_code, 500, failed.text)
        record = self.store._inspect_sync(start["adaptive_session_id"])
        self.assertIsNotNone(record.aggregate.pending.next_decision)
        self.assertTrue(record.aggregate.current_quiz.profile_written)
        self.assertFalse(record.busy)

        recovered = self.client.post("/agent/adaptive/submit", json=payload, headers=headers)
        self.assertEqual(recovered.status_code, 200, recovered.text)
        self.assertEqual(recovered.json()["turn"], 2)
        self.assertEqual(self.memory.await_count, 1)
        self.assertEqual(self.append.await_count, 1)
        self.assertEqual(self.mastery.await_count, 2)
        self.assertEqual(self.decide.await_count, 2)

    def test_snapshot_failure_does_not_persist_profile_marker(self) -> None:
        self.decide.side_effect = [_decision(), _decision("finish")]

        async def fail_after_core_write(
            *args, after_write=None, on_core_written=None, **kwargs
        ) -> None:
            if on_core_written is not None:
                on_core_written()
            raise RuntimeError("learner-memory snapshot was not persisted")

        self.memory.side_effect = fail_after_core_write
        start = self._start()
        payload = self._submit_payload(start)
        headers = {"Idempotency-Key": "adaptive-snapshot-retry-1"}

        failed = self.client.post("/agent/adaptive/submit", json=payload, headers=headers)
        self.assertEqual(failed.status_code, 500, failed.text)
        record = self.store._inspect_sync(start["adaptive_session_id"])
        self.assertIsNotNone(record.aggregate.pending)
        self.assertFalse(record.aggregate.current_quiz.profile_written)
        self.assertFalse(record.busy)

        async def successful_commit(
            *args, after_write=None, on_core_written=None, **kwargs
        ) -> None:
            if on_core_written is not None:
                on_core_written()
            if after_write is not None:
                await after_write()

        self.memory.side_effect = successful_commit
        recovered = self.client.post("/agent/adaptive/submit", json=payload, headers=headers)
        self.assertEqual(recovered.status_code, 200, recovered.text)
        self.assertTrue(recovered.json()["done"])
        self.assertEqual(self.memory.await_count, 2)

    def test_swallowed_decision_audit_failure_keeps_memory_stage_retryable(self) -> None:
        self.decide.side_effect = [_decision(), _decision("finish")]
        self.append.side_effect = [RuntimeError("audit store unavailable"), None]

        async def fail_soft_memory(*args, after_write=None, on_core_written=None, **kwargs) -> None:
            if on_core_written is not None:
                on_core_written()
            if after_write is not None:
                try:
                    await after_write()
                except Exception:
                    pass

        self.memory.side_effect = fail_soft_memory
        start = self._start()
        payload = self._submit_payload(start)
        headers = {"Idempotency-Key": "adaptive-decision-audit-1"}

        failed = self.client.post("/agent/adaptive/submit", json=payload, headers=headers)
        self.assertEqual(failed.status_code, 500, failed.text)
        record = self.store._inspect_sync(start["adaptive_session_id"])
        self.assertIsNotNone(record.aggregate.pending)
        self.assertFalse(record.aggregate.current_quiz.profile_written)
        self.assertFalse(record.busy)

        recovered = self.client.post("/agent/adaptive/submit", json=payload, headers=headers)
        self.assertEqual(recovered.status_code, 200, recovered.text)
        self.assertTrue(recovered.json()["done"])
        self.assertEqual(self.memory.await_count, 2)
        self.assertEqual(self.append.await_count, 2)

    def test_stale_and_busy_are_stable_conflicts(self) -> None:
        start = self._start()
        stale_payload = self._submit_payload(start)
        stale_payload["revision"] += 1
        stale = self.client.post("/agent/adaptive/submit", json=stale_payload)
        self.assertEqual(stale.status_code, 409, stale.text)
        self.assertEqual(stale.json()["code"], "adaptive_session_stale")

        claim = self.store._claim_sync(
            start["adaptive_session_id"], "external-test", "test-claim-token"
        )
        self.assertTrue(claim.claimed)
        busy = self.client.post("/agent/adaptive/submit", json=self._submit_payload(start))
        self.assertEqual(busy.status_code, 409, busy.text)
        self.assertEqual(busy.json()["code"], "adaptive_session_busy")
        self.store._release_sync(start["adaptive_session_id"], "test-claim-token")

    def test_submit_key_payload_mismatch_does_not_rerun_work(self) -> None:
        self.decide.side_effect = [_decision(), _decision("finish")]
        start = self._start()
        payload = self._submit_payload(start)
        headers = {"Idempotency-Key": "adaptive-submit-binding-1"}
        first = self.client.post("/agent/adaptive/submit", json=payload, headers=headers)
        self.assertEqual(first.status_code, 200, first.text)

        mismatch_payload = dict(payload)
        mismatch_payload["answers"] = ["B"]
        mismatch = self.client.post(
            "/agent/adaptive/submit", json=mismatch_payload, headers=headers
        )
        self.assertEqual(mismatch.status_code, 409, mismatch.text)
        self.assertEqual(mismatch.json()["code"], "idempotency_conflict")
        self.assertEqual(mismatch.json()["reason"], "payload_mismatch")
        self.assertEqual(self.memory.await_count, 1)

    def test_missing_session_is_stable_404(self) -> None:
        response = self.client.get("/agent/adaptive/adapt_missing")
        self.assertEqual(response.status_code, 404, response.text)
        self.assertEqual(response.json()["code"], "adaptive_session_not_found")

    def test_expired_session_is_stable_410(self) -> None:
        now = [100.0]
        expiring_store = AdaptiveSessionStore(
            sqlite_path=str(Path(self.tempdir.name) / "expiring.sqlite3"),
            ttl_seconds=1,
            clock=lambda: now[0],
        )
        with patch.object(adaptive_router, "adaptive_sessions", expiring_store):
            start = self._start()
            now[0] = 102.0
            response = self.client.get(f"/agent/adaptive/{start['adaptive_session_id']}")

        self.assertEqual(response.status_code, 410, response.text)
        self.assertEqual(response.json()["code"], "adaptive_session_expired")

    def test_corrupt_session_is_redacted_503(self) -> None:
        start = self._start()
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute(
                """
                UPDATE studyloop_adaptive_sessions
                SET payload_json = ?
                WHERE session_id = ?
                """,
                ('{"schema_version":1}', start["adaptive_session_id"]),
            )
            connection.commit()
        finally:
            connection.close()
        response = self.client.get(f"/agent/adaptive/{start['adaptive_session_id']}")

        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json()["code"], "adaptive_session_corrupt")
        self.assertNotIn("schema_version", response.text)


if __name__ == "__main__":
    unittest.main()
