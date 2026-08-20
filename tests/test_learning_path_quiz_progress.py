"""End-to-end contracts between durable Quiz grading and path progress."""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from chromadb.errors import NotFoundError
from fastapi.testclient import TestClient

from main import app
from models.grader import AIFeedback, GradingReport, QuestionGrade
from models.learning_path import LearningPath, LearningStage
from models.quiz import Question
from models.report import LearningReport
from models.session import (
    LearningPathQuizSource,
    QuizSession,
    QuizSessionAggregate,
)
import routers.session as session_router
from services.learning_path_store import LearningPathStore
from services.quiz_sessions import QuizSessionStore
from services.session import answer_result_for_index
import services.grader as grader_service


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _path() -> LearningPath:
    return LearningPath(
        document_id="notes.md",
        title="Durable path",
        total_stages=2,
        stages=[
            LearningStage(
                stage=1,
                title="Stage one",
                topics=["foundation"],
                description="Authoritative stage one objective",
                estimated_minutes=10,
            ),
            LearningStage(
                stage=2,
                title="Stage two",
                topics=["application"],
                description="Authoritative stage two objective",
                estimated_minutes=10,
            ),
        ],
    )


def _question() -> Question:
    return Question(
        question="Choose A",
        options=["A. yes", "B. no"],
        answer="A",
        explanation="A is correct",
        source="notes.md",
        type="choice",
    )


def _aggregate(
    session_id: str,
    source: LearningPathQuizSource,
    *,
    active: bool = False,
    canonical_grade: bool = False,
) -> QuizSessionAggregate:
    question = _question()
    answers = [] if active else ["A"]
    session = QuizSession(
        session_id=session_id,
        document_id="notes.md",
        user_id="default_user",
        questions=[question],
        user_answers=answers,
        status="active" if active else "completed",
    )
    if canonical_grade:
        grade = QuestionGrade(
            index=0,
            question=question.question,
            user_answer="A",
            correct_answer=question.answer,
            is_correct=True,
        )
        session.question_grades = {0: grade}
        session.grading_report = GradingReport(
            session_id=session_id,
            total=1,
            correct=1,
            score=1.0,
            grades=[grade],
        )
        session.profile_written = True
    return QuizSessionAggregate(
        session=session,
        last_answer_index=None if active else 0,
        last_answer_result=(
            None if active else answer_result_for_index(session, 0)
        ),
        learning_path_source=source,
    )


class TestLearningPathQuizProgress(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.path_store = LearningPathStore(
            sqlite_path=str(root / "learning-paths.sqlite3")
        )
        self.quiz_store = QuizSessionStore(
            sqlite_path=str(root / "quiz-sessions.sqlite3")
        )
        self.patches = [
            patch.object(session_router, "learning_path_store", self.path_store),
            patch.object(session_router, "quiz_sessions", self.quiz_store),
        ]
        for active_patch in self.patches:
            active_patch.start()
        self.client = TestClient(app, raise_server_exceptions=False)
        self.path_record = asyncio.run(
            self.path_store.create(
                "default_user",
                "notes.md",
                _path(),
                idempotency_key="learning-path-progress-resource",
                request_fingerprint=_fingerprint("path request"),
            )
        )
        self.source = LearningPathQuizSource(
            learning_path_id=self.path_record.path_id,
            stage_id=1,
        )

    def tearDown(self) -> None:
        self.client.close()
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.tempdir.cleanup()

    def _start_body(self, *, stage_id: int = 1) -> dict:
        return {
            "document_id": "notes.md",
            "description": "malicious client objective",
            "count": 1,
            "difficulty": "medium",
            "type": "choice",
            "user_id": "default_user",
            "learning_path_source": {
                "learning_path_id": self.path_record.path_id,
                "stage_id": stage_id,
            },
        }

    def _seed(self, aggregate: QuizSessionAggregate) -> None:
        asyncio.run(self.quiz_store.create(aggregate))

    @staticmethod
    def _memory_writer():
        async def commit_memory(*args, on_core_written=None, **kwargs):
            if on_core_written is not None:
                on_core_written()

        return commit_memory

    def test_bound_start_is_replay_first_and_uses_server_stage_objective(self) -> None:
        captured = []

        async def prepare(req):
            captured.append(req)
            return _aggregate("bound-start", self.source, active=True).session

        headers = {"Idempotency-Key": "learning-path-stage-start-key"}
        with patch.object(session_router, "prepare_session", side_effect=prepare):
            created = self.client.post(
                "/session/start",
                json=self._start_body(),
                headers=headers,
            )
            replay = self.client.post(
                "/session/start",
                json=self._start_body(),
                headers=headers,
            )

        self.assertEqual(created.status_code, 200, created.text)
        self.assertEqual(replay.json(), created.json())
        self.assertEqual(len(captured), 1)
        self.assertEqual(
            captured[0].description,
            "Authoritative stage one objective",
        )
        self.assertEqual(created.json()["learning_path_source"], self.source.model_dump())
        stored = asyncio.run(self.quiz_store.inspect("bound-start"))
        self.assertEqual(stored.aggregate.learning_path_source, self.source)

        missing_key = self.client.post("/session/start", json=self._start_body())
        self.assertEqual(missing_key.status_code, 400, missing_key.text)

        with patch.object(
            session_router,
            "prepare_session",
            AsyncMock(side_effect=AssertionError("locked stage reached provider")),
        ):
            locked = self.client.post(
                "/session/start",
                json=self._start_body(stage_id=2),
                headers={"Idempotency-Key": "learning-path-stage-locked-key"},
            )
        self.assertEqual(locked.status_code, 409, locked.text)
        self.assertEqual(locked.json()["reason"], "stage_locked")

    def test_existing_start_replays_after_material_disappears(self) -> None:
        async def prepare(_req):
            return _aggregate("deleted-material-start", self.source, active=True).session

        headers = {"Idempotency-Key": "learning-path-deleted-replay-key"}
        with patch.object(session_router, "prepare_session", side_effect=prepare):
            created = self.client.post(
                "/session/start",
                json=self._start_body(),
                headers=headers,
            )
        self.assertEqual(created.status_code, 200, created.text)

        missing = AsyncMock(side_effect=NotFoundError("deleted"))
        with patch.object(session_router, "prepare_session", missing):
            replay = self.client.post(
                "/session/start",
                json=self._start_body(),
                headers=headers,
            )
            fresh = self.client.post(
                "/session/start",
                json=self._start_body(),
                headers={"Idempotency-Key": "learning-path-deleted-fresh-key"},
            )
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(replay.json(), created.json())
        self.assertEqual(fresh.status_code, 404, fresh.text)
        self.assertEqual(missing.await_count, 1)

    def test_only_the_first_incomplete_stage_can_start(self) -> None:
        asyncio.run(
            self.path_store.complete_stage(
                self.path_record.path_id,
                1,
                "completed-stage-one-quiz",
                user_id="default_user",
                document_id="notes.md",
                grading_report_hash=_fingerprint("canonical stage one grade"),
            )
        )
        stage_two_source = LearningPathQuizSource(
            learning_path_id=self.path_record.path_id,
            stage_id=2,
        )
        captured = []

        async def prepare(req):
            captured.append(req)
            return _aggregate(
                "ready-stage-two",
                stage_two_source,
                active=True,
            ).session

        with patch.object(session_router, "prepare_session", side_effect=prepare):
            completed = self.client.post(
                "/session/start",
                json=self._start_body(stage_id=1),
                headers={"Idempotency-Key": "completed-stage-one-start"},
            )
            ready = self.client.post(
                "/session/start",
                json=self._start_body(stage_id=2),
                headers={"Idempotency-Key": "ready-stage-two-start"},
            )

        self.assertEqual(completed.status_code, 409, completed.text)
        self.assertEqual(completed.json()["reason"], "stage_completed")
        self.assertEqual(ready.status_code, 200, ready.text)
        self.assertEqual(captured[0].description, "Authoritative stage two objective")
        self.assertEqual(ready.json()["learning_path_source"]["stage_id"], 2)

    def test_provider_failure_rechecks_a_concurrently_committed_start(self) -> None:
        headers = {"Idempotency-Key": "learning-path-provider-race-key"}
        request_body = self._start_body()

        async def commit_then_fail(_req):
            aggregate = _aggregate(
                "learning-path-provider-race",
                self.source,
                active=True,
            )
            await self.quiz_store.create(
                aggregate,
                start_key=headers["Idempotency-Key"],
                start_request=request_body,
            )
            raise RuntimeError("provider response was lost")

        with patch.object(
            session_router,
            "prepare_session",
            side_effect=commit_then_fail,
        ):
            response = self.client.post(
                "/session/start",
                json=request_body,
                headers=headers,
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["session_id"], "learning-path-provider-race")
        self.assertEqual(response.json()["learning_path_source"], self.source.model_dump())

    def test_create_ack_loss_replays_the_committed_session(self) -> None:
        headers = {"Idempotency-Key": "learning-path-create-ack-loss-key"}
        request_body = self._start_body()
        original_create = self.quiz_store.create

        async def prepare(_req):
            return _aggregate(
                "learning-path-create-ack-loss",
                self.source,
                active=True,
            ).session

        async def commit_then_lose_ack(*args, **kwargs):
            await original_create(*args, **kwargs)
            raise RuntimeError("create response was lost after commit")

        with (
            patch.object(session_router, "prepare_session", side_effect=prepare),
            patch.object(
                self.quiz_store,
                "create",
                side_effect=commit_then_lose_ack,
            ),
        ):
            response = self.client.post(
                "/session/start",
                json=request_body,
                headers=headers,
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["session_id"],
            "learning-path-create-ack-loss",
        )

    def test_oversized_loser_replays_a_concurrently_committed_start(self) -> None:
        headers = {"Idempotency-Key": "learning-path-oversized-loser-key"}
        request_body = self._start_body()

        async def seed_winner_then_return_oversized(_req):
            winner = _aggregate(
                "learning-path-size-winner",
                self.source,
                active=True,
            )
            await self.quiz_store.create(
                winner,
                start_key=headers["Idempotency-Key"],
                start_request=request_body,
            )
            loser = _aggregate(
                "learning-path-size-loser",
                self.source,
                active=True,
            )
            loser.session.questions[0].question = "x" * (2 * 1024 * 1024)
            return loser.session

        with patch.object(
            session_router,
            "prepare_session",
            side_effect=seed_winner_then_return_oversized,
        ):
            response = self.client.post(
                "/session/start",
                json=request_body,
                headers=headers,
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["session_id"], "learning-path-size-winner")

    def test_unbound_start_receipt_keeps_its_pre_binding_request_hash(self) -> None:
        legacy_request = {
            "document_id": "notes.md",
            "description": "legacy objective",
            "count": 1,
            "difficulty": "medium",
            "type": "choice",
            "user_id": "default_user",
        }
        legacy = _aggregate("legacy-unbound", self.source, active=True)
        legacy.learning_path_source = None
        asyncio.run(
            self.quiz_store.create(
                legacy,
                start_key="legacy-unbound-start-key",
                start_request=legacy_request,
            )
        )
        with patch.object(
            session_router,
            "prepare_session",
            AsyncMock(side_effect=AssertionError("legacy replay called provider")),
        ):
            replay = self.client.post(
                "/session/start",
                json=legacy_request,
                headers={"Idempotency-Key": "legacy-unbound-start-key"},
            )
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertIsNone(replay.json()["learning_path_source"])

    def test_bound_objective_answer_waits_for_canonical_grade_before_progress(self) -> None:
        aggregate = _aggregate("bound-objective", self.source, active=True)
        self._seed(aggregate)
        fallback = AsyncMock(side_effect=AssertionError("objective fallback was used"))
        with patch.object(session_router, "write_objective_profile", fallback):
            answered = self.client.post(
                "/session/bound-objective/answer",
                json={"answer": "A", "question_index": 0},
            )
        self.assertEqual(answered.status_code, 200, answered.text)
        fallback.assert_not_awaited()
        after_answer = asyncio.run(self.quiz_store.inspect("bound-objective"))
        self.assertFalse(after_answer.aggregate.session.profile_written)
        self.assertIsNone(after_answer.aggregate.learning_path_completion)
        before_grade = self.client.get("/session/bound-objective/result")
        self.assertEqual(before_grade.status_code, 200, before_grade.text)
        self.assertEqual(
            asyncio.run(self.path_store.get(self.path_record.path_id)).completed_through,
            0,
        )

        with (
            patch.object(
                grader_service,
                "_llm_grade",
                AsyncMock(return_value=AIFeedback(
                    is_correct=True,
                    feedback="correct",
                    knowledge_gap="",
                )),
            ),
            patch.object(
                session_router,
                "commit_learning_memory",
                side_effect=self._memory_writer(),
            ),
        ):
            graded = self.client.post("/session/bound-objective/grade")

        self.assertEqual(graded.status_code, 200, graded.text)
        self.assertEqual(graded.json()["learning_path_completion"]["stage_id"], 1)
        self.assertEqual(graded.json()["learning_path_completion"]["completed_through"], 1)
        persisted = asyncio.run(self.quiz_store.inspect("bound-objective"))
        self.assertTrue(persisted.aggregate.session.profile_written)
        self.assertIsNotNone(persisted.aggregate.learning_path_completion)
        path = asyncio.run(self.path_store.get(self.path_record.path_id))
        self.assertEqual(path.completed_through, 1)
        self.assertEqual(path.progress_revision, 2)

    def test_snapshot_repairs_progress_after_canonical_grade_checkpoint(self) -> None:
        aggregate = _aggregate(
            "bound-progress-repair",
            self.source,
            canonical_grade=True,
        )
        self._seed(aggregate)

        snapshot = self.client.get("/session/bound-progress-repair")

        self.assertEqual(snapshot.status_code, 200, snapshot.text)
        self.assertEqual(
            snapshot.json()["learning_path_completion"],
            {
                "learning_path_id": self.path_record.path_id,
                "stage_id": 1,
                "completed_through": 1,
                "revision": 2,
            },
        )
        persisted = asyncio.run(self.quiz_store.inspect("bound-progress-repair"))
        self.assertIsNotNone(persisted.aggregate.learning_path_completion)
        self.assertEqual(
            asyncio.run(self.path_store.get(self.path_record.path_id)).completed_through,
            1,
        )

    def test_snapshot_retries_after_progress_event_ack_is_lost(self) -> None:
        aggregate = _aggregate(
            "bound-progress-ack-loss",
            self.source,
            canonical_grade=True,
        )
        self._seed(aggregate)
        original_checkpoint = self.quiz_store.checkpoint
        failures = 0

        async def fail_first_checkpoint(*args, **kwargs):
            nonlocal failures
            failures += 1
            if failures == 1:
                raise RuntimeError("quiz checkpoint response was lost")
            return await original_checkpoint(*args, **kwargs)

        with patch.object(
            self.quiz_store,
            "checkpoint",
            side_effect=fail_first_checkpoint,
        ):
            first = self.client.get("/session/bound-progress-ack-loss")

        self.assertEqual(first.status_code, 200, first.text)
        self.assertIsNone(first.json()["learning_path_completion"])
        path_after_loss = asyncio.run(
            self.path_store.get(self.path_record.path_id)
        )
        self.assertEqual(path_after_loss.completed_through, 1)

        repaired = self.client.get("/session/bound-progress-ack-loss")
        self.assertEqual(repaired.status_code, 200, repaired.text)
        self.assertEqual(
            repaired.json()["learning_path_completion"]["completed_through"],
            1,
        )

    def test_snapshot_replays_ack_committed_before_response_loss(self) -> None:
        aggregate = _aggregate(
            "bound-progress-committed-ack-loss",
            self.source,
            canonical_grade=True,
        )
        self._seed(aggregate)
        original_checkpoint = self.quiz_store.checkpoint
        calls = 0

        async def commit_first_checkpoint_then_fail(*args, **kwargs):
            nonlocal calls
            calls += 1
            committed = await original_checkpoint(*args, **kwargs)
            if calls == 1:
                raise RuntimeError("checkpoint ack was lost after commit")
            return committed

        with patch.object(
            self.quiz_store,
            "checkpoint",
            side_effect=commit_first_checkpoint_then_fail,
        ):
            first = self.client.get(
                "/session/bound-progress-committed-ack-loss"
            )

        self.assertEqual(first.status_code, 200, first.text)
        self.assertIsNone(first.json()["learning_path_completion"])
        persisted = asyncio.run(
            self.quiz_store.inspect("bound-progress-committed-ack-loss")
        )
        self.assertIsNotNone(persisted.aggregate.learning_path_completion)

        replay = self.client.get("/session/bound-progress-committed-ack-loss")
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(
            replay.json()["learning_path_completion"]["completed_through"],
            1,
        )

    def test_report_failure_does_not_roll_back_stage_completion(self) -> None:
        aggregate = _aggregate(
            "bound-report-failure",
            self.source,
            canonical_grade=True,
        )
        self._seed(aggregate)

        async def invalid_report(*_args, **_kwargs):
            return LearningReport.model_validate({})

        with patch.object(
            session_router,
            "generate_report_for_quiz",
            side_effect=invalid_report,
        ):
            failed = self.client.post("/session/bound-report-failure/report")

        self.assertEqual(failed.status_code, 503, failed.text)
        persisted = asyncio.run(self.quiz_store.inspect("bound-report-failure"))
        self.assertIsNotNone(persisted.aggregate.learning_path_completion)
        path = asyncio.run(self.path_store.get(self.path_record.path_id))
        self.assertEqual(path.completed_through, 1)

        snapshot = self.client.get("/session/bound-report-failure")
        self.assertEqual(snapshot.status_code, 200, snapshot.text)
        self.assertEqual(
            snapshot.json()["learning_path_completion"]["stage_id"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
