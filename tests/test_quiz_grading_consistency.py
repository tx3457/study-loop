import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from agents import adapt_agent
from main import app
from models.grader import AIFeedback
from models.quiz import Question
from models.report import TopicMastery, _ReportCore
from models.session import (
    AnswerRequest,
    QuizSession,
    QuizSessionAggregate,
    SessionStartRequest,
)
from routers import session as session_router
from services import grader as grader_service
from services import report as report_service
from services import session as session_service
from services.quiz_sessions import QuizSessionStore


def _question(
    *,
    question_type: str = "short_answer",
    prompt: str = "Explain retrieval-augmented generation.",
) -> Question:
    return Question(
        question=prompt,
        options=None if question_type == "short_answer" else ["A. Retrieval", "B. Cache"],
        answer="Retrieval" if question_type != "short_answer" else "Retrieve evidence before answering.",
        explanation="Use retrieved evidence to ground the answer.",
        source="notes.md",
        type=question_type,
    )


def _completed_session(
    session_id: str,
    *,
    questions: list[Question] | None = None,
    answers: list[str] | None = None,
) -> QuizSession:
    questions = questions or [_question()]
    answers = answers or ["Find supporting material first, then answer from it."]
    return QuizSession(
        session_id=session_id,
        document_id="notes.md",
        user_id="user-1",
        questions=questions,
        user_answers=answers,
        status="completed",
    )


def _feedback(*, is_correct: bool = True, label: str = "feedback") -> AIFeedback:
    return AIFeedback(
        is_correct=is_correct,
        feedback=label,
        knowledge_gap="" if is_correct else "retrieval order",
    )


def _durable_aggregate(session: QuizSession) -> QuizSessionAggregate:
    last_index = len(session.user_answers) - 1
    return QuizSessionAggregate(
        session=session,
        last_answer_index=last_index,
        last_answer_result=session_service.answer_result_for_index(
            session,
            last_index,
        ),
    )


def _report_response():
    core = _ReportCore(
        topic_mastery=[
            TopicMastery(
                topic="RAG",
                mastery_pct=100,
                question_count=1,
                correct_count=1,
            )
        ],
        strengths=["RAG"],
        weaknesses=[],
        recommendations=["Keep practising."],
        summary="The answer uses retrieval before generation.",
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=core))]
    )


class TestQuizGradingConsistency(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_sessions = dict(session_service.sessions)
        session_service.sessions.clear()
        grader_service._grade_locks.clear()
        report_service._report_locks.clear()

    def tearDown(self):
        session_service.sessions.clear()
        session_service.sessions.update(self.original_sessions)
        grader_service._grade_locks.clear()
        report_service._report_locks.clear()

    async def test_short_answer_is_pending_until_semantic_grading(self):
        session_id = "short-pending"
        session_service.sessions[session_id] = QuizSession(
            session_id=session_id,
            document_id="notes.md",
            user_id="user-1",
            questions=[_question()],
            user_answers=[],
            status="active",
        )

        answer = await session_service.submit_answer(
            session_id,
            "Find supporting material first, then answer from it.",
            question_index=0,
        )
        result = await session_service.get_result(session_id)

        self.assertEqual(answer.evaluation_status, "pending_ai")
        self.assertIsNone(answer.correct)
        self.assertIsNone(answer.correct_answer)
        self.assertIsNone(answer.explanation)
        self.assertEqual(result.pending, 1)
        self.assertEqual(result.correct, 0)
        self.assertEqual(result.incorrect, 0)
        self.assertIsNone(result.score)
        self.assertEqual(result.details[0].evaluation_status, "pending_ai")

    async def test_concurrent_grades_share_one_semantic_evaluation(self):
        session_id = "grade-single-flight"
        session_service.sessions[session_id] = _completed_session(session_id)
        llm_grade = AsyncMock(return_value=_feedback())

        with patch.object(grader_service, "_llm_grade", llm_grade):
            first, second = await asyncio.gather(
                grader_service.grade_session(session_id),
                grader_service.grade_session(session_id),
            )

        self.assertEqual(first.model_dump(), second.model_dump())
        self.assertEqual(first.correct, 1)
        llm_grade.assert_awaited_once()

        final_result = await session_service.get_result(session_id)
        self.assertEqual(final_result.pending, 0)
        self.assertEqual(final_result.correct, 1)
        self.assertEqual(final_result.score, 1.0)

    async def test_feedback_model_cannot_flip_objective_correctness(self):
        session_id = "objective-authority"
        session_service.sessions[session_id] = _completed_session(
            session_id,
            questions=[_question(question_type="choice")],
            answers=["B. Cache"],
        )

        with patch.object(
            grader_service,
            "_llm_grade",
            AsyncMock(return_value=_feedback(is_correct=True)),
        ):
            grading = await grader_service.grade_session(session_id)

        self.assertEqual(grading.correct, 0)
        self.assertFalse(grading.grades[0].is_correct)
        self.assertEqual(grading.grades[0].ai_feedback, "feedback")

    async def test_partial_provider_failure_only_retries_missing_question(self):
        session_id = "partial-grading"
        session_service.sessions[session_id] = _completed_session(
            session_id,
            questions=[
                _question(prompt="Question one"),
                _question(prompt="Question two"),
            ],
            answers=["Answer one", "Answer two"],
        )
        llm_grade = AsyncMock(
            side_effect=[_feedback(label="first"), RuntimeError("provider down")]
        )

        with patch.object(grader_service, "_llm_grade", llm_grade):
            with self.assertRaisesRegex(RuntimeError, "provider down"):
                await grader_service.grade_session(session_id)
            llm_grade.side_effect = None
            llm_grade.return_value = _feedback(label="second")
            grading = await grader_service.grade_session(session_id)

        self.assertEqual(llm_grade.await_count, 3)
        self.assertEqual(
            [grade.ai_feedback for grade in grading.grades],
            ["first", "second"],
        )

    async def test_cached_report_still_rejects_noncanonical_grading(self):
        session_id = "cached-report-validation"
        session_service.sessions[session_id] = _completed_session(session_id)

        with patch.object(
            grader_service,
            "_llm_grade",
            AsyncMock(return_value=_feedback()),
        ):
            grading = await grader_service.grade_session(session_id)
        with patch.object(
            report_service,
            "llm_parse",
            AsyncMock(return_value=_report_response()),
        ):
            await report_service.generate_report(session_id, grading)

        mismatched = grading.model_copy(update={"score": 0.0})
        with self.assertRaisesRegex(ValueError, "does not match"):
            await report_service.generate_report(session_id, mismatched)

    async def test_agent_profile_write_uses_session_ownership(self):
        session_id = "authoritative-owner"
        session = _completed_session(session_id)
        session_service.sessions[session_id] = session
        llm_grade = AsyncMock(return_value=_feedback())

        with patch.object(grader_service, "_llm_grade", llm_grade):
            grading = await grader_service.grade_session(session_id)

        captured = {}

        async def commit_memory(user_id, report, document_id, **kwargs):
            captured.update(
                user_id=user_id,
                document_id=document_id,
                questions=kwargs.get("questions"),
            )
            await kwargs["after_write"]()
            kwargs["on_core_written"]()

        decision = AsyncMock()
        with (
            patch.object(adapt_agent, "commit_learning_memory", commit_memory),
            patch.object(adapt_agent, "append_decision", decision),
        ):
            await adapt_agent._write_profile(
                {
                    "user_id": "wrong-user",
                    "document_id": "wrong-document",
                    "grading_report": grading.model_dump(),
                }
            )

        self.assertEqual(captured["user_id"], "user-1")
        self.assertEqual(captured["document_id"], "notes.md")
        self.assertEqual(captured["questions"], session.questions)
        self.assertTrue(session.profile_written)
        self.assertEqual(decision.await_args.args[0], "user-1")
        self.assertEqual(
            decision.await_args.args[1]["decision_id"],
            f"adapt_writer:{session_id}",
        )

    def test_answer_contract_rejects_blank_and_unbounded_input(self):
        with self.assertRaises(ValidationError):
            AnswerRequest(answer="   ", question_index=0)
        with self.assertRaises(ValidationError):
            AnswerRequest(answer="x" * 4001, question_index=0)

    def test_session_start_contract_rejects_unbounded_or_unknown_modes(self):
        base = {
            "document_id": "notes.md",
            "description": "RAG",
            "user_id": "user-1",
        }
        for invalid in (
            {"count": 0},
            {"count": 11},
            {"difficulty": "impossible"},
            {"type": "essay"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                SessionStartRequest(**base, **invalid)


class TestDurableWebQuizGrading(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tempdir.name) / "quiz-sessions.sqlite3")
        self.client = TestClient(app, raise_server_exceptions=False)
        self.store_patch = None
        self._install_store(QuizSessionStore(sqlite_path=self.db_path))

    def tearDown(self):
        if self.store_patch is not None:
            self.store_patch.stop()
        self.client.close()
        self.tempdir.cleanup()

    def _install_store(self, store: QuizSessionStore) -> None:
        if self.store_patch is not None:
            self.store_patch.stop()
        self.store = store
        self.store_patch = patch.object(session_router, "quiz_sessions", store)
        self.store_patch.start()

    def _reopen_store(self) -> None:
        self._install_store(QuizSessionStore(sqlite_path=self.db_path))

    def _seed(self, session: QuizSession) -> None:
        asyncio.run(self.store.create(_durable_aggregate(session)))

    def _inspect(self, session_id: str):
        return asyncio.run(self.store.inspect(session_id))

    @staticmethod
    def _memory_writer(counter: dict[str, int]):
        async def commit_memory(*args, on_core_written=None, **kwargs):
            counter["calls"] += 1
            if on_core_written is not None:
                on_core_written()

        return commit_memory

    def test_partial_grade_checkpoint_survives_failure_and_store_reopen(self):
        session_id = "durable-partial-grading"
        self._seed(
            _completed_session(
                session_id,
                questions=[
                    _question(prompt="Question one"),
                    _question(prompt="Question two"),
                ],
                answers=["Answer one", "Answer two"],
            )
        )
        llm_grade = AsyncMock(
            side_effect=[_feedback(label="first"), RuntimeError("provider down")]
        )
        memory = {"calls": 0}

        with (
            patch.object(grader_service, "_llm_grade", llm_grade),
            patch.object(
                session_router,
                "commit_learning_memory",
                side_effect=self._memory_writer(memory),
            ),
        ):
            failed = self.client.post(f"/session/{session_id}/grade")
            self.assertEqual(failed.status_code, 500, failed.text)

            checkpoint = self._inspect(session_id)
            self.assertIsNotNone(checkpoint)
            self.assertFalse(checkpoint.busy)
            self.assertEqual(
                list(checkpoint.aggregate.session.question_grades),
                [0],
            )
            self.assertIsNone(checkpoint.aggregate.session.grading_report)

            self._reopen_store()
            llm_grade.side_effect = None
            llm_grade.return_value = _feedback(label="second")
            completed = self.client.post(f"/session/{session_id}/grade")

        self.assertEqual(completed.status_code, 200, completed.text)
        self.assertEqual(
            [grade["ai_feedback"] for grade in completed.json()["grades"]],
            ["first", "second"],
        )
        self.assertEqual(llm_grade.await_count, 3)
        self.assertEqual(memory["calls"], 1)

        persisted = self._inspect(session_id)
        self.assertIsNotNone(persisted.aggregate.session.grading_report)
        self.assertTrue(persisted.aggregate.session.profile_written)

    def test_memory_retry_after_reopen_reuses_canonical_grading(self):
        session_id = "durable-memory-retry"
        self._seed(_completed_session(session_id))
        llm_grade = AsyncMock(return_value=_feedback())
        successful_memory = {"calls": 0}

        with patch.object(grader_service, "_llm_grade", llm_grade):
            with patch.object(
                session_router,
                "commit_learning_memory",
                AsyncMock(side_effect=RuntimeError("memory unavailable")),
            ):
                failed = self.client.post(f"/session/{session_id}/grade")

            self.assertEqual(failed.status_code, 500, failed.text)
            checkpoint = self._inspect(session_id)
            self.assertIsNotNone(checkpoint.aggregate.session.grading_report)
            self.assertFalse(checkpoint.aggregate.session.profile_written)
            self.assertFalse(checkpoint.busy)

            self._reopen_store()
            with patch.object(
                session_router,
                "commit_learning_memory",
                side_effect=self._memory_writer(successful_memory),
            ):
                completed = self.client.post(f"/session/{session_id}/grade")

        self.assertEqual(completed.status_code, 200, completed.text)
        llm_grade.assert_awaited_once()
        self.assertEqual(successful_memory["calls"], 1)
        self.assertTrue(
            self._inspect(session_id).aggregate.session.profile_written
        )

    def test_report_retry_and_reopen_reuses_canonical_caches_and_memory(self):
        session_id = "durable-report-retry"
        self._seed(_completed_session(session_id))
        llm_grade = AsyncMock(return_value=_feedback())
        report_llm = AsyncMock(
            side_effect=[RuntimeError("report unavailable"), _report_response()]
        )
        memory = {"calls": 0}

        with (
            patch.object(grader_service, "_llm_grade", llm_grade),
            patch.object(report_service, "llm_parse", report_llm),
            patch.object(
                session_router,
                "commit_learning_memory",
                side_effect=self._memory_writer(memory),
            ),
        ):
            failed = self.client.post(f"/session/{session_id}/report")
            self.assertEqual(failed.status_code, 500, failed.text)

            checkpoint = self._inspect(session_id)
            self.assertIsNotNone(checkpoint.aggregate.session.grading_report)
            self.assertTrue(checkpoint.aggregate.session.profile_written)
            self.assertIsNone(checkpoint.aggregate.session.learning_report)
            self.assertFalse(checkpoint.busy)

            self._reopen_store()
            completed = self.client.post(f"/session/{session_id}/report")
            self.assertEqual(completed.status_code, 200, completed.text)

            self._reopen_store()
            cached = self.client.post(f"/session/{session_id}/report")

        self.assertEqual(cached.status_code, 200, cached.text)
        self.assertEqual(completed.json()["summary"], cached.json()["summary"])
        llm_grade.assert_awaited_once()
        self.assertEqual(report_llm.await_count, 2)
        self.assertEqual(memory["calls"], 1)

        persisted = self._inspect(session_id)
        self.assertIsNotNone(persisted.aggregate.session.grading_report)
        self.assertIsNotNone(persisted.aggregate.session.learning_report)
        self.assertTrue(persisted.aggregate.session.profile_written)

    def test_missing_durable_grade_is_http_404(self):
        response = self.client.post("/session/missing-session/grade")

        self.assertEqual(response.status_code, 404, response.text)
        self.assertEqual(response.json()["code"], "quiz_session_not_found")


if __name__ == "__main__":
    unittest.main()
