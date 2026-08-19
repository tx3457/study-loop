import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from agents import adapt_agent
from models.grader import AIFeedback
from models.quiz import Question
from models.report import TopicMastery, _ReportCore
from models.session import AnswerRequest, QuizSession, SessionStartRequest
from routers import session as session_router
from services import grader as grader_service
from services import report as report_service
from services import session as session_service


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
        session_router._answer_locks.clear()

    def tearDown(self):
        session_service.sessions.clear()
        session_service.sessions.update(self.original_sessions)
        grader_service._grade_locks.clear()
        report_service._report_locks.clear()
        session_router._answer_locks.clear()

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

    async def test_report_reuses_canonical_grade_and_writes_memory_once(self):
        session_id = "report-canonical"
        session_service.sessions[session_id] = _completed_session(session_id)
        llm_grade = AsyncMock(return_value=_feedback())
        report_llm = AsyncMock(return_value=_report_response())
        memory_calls = 0

        async def commit_memory(*args, on_core_written=None, **kwargs):
            nonlocal memory_calls
            memory_calls += 1
            if on_core_written is not None:
                on_core_written()

        with (
            patch.object(grader_service, "_llm_grade", llm_grade),
            patch.object(report_service, "llm_parse", report_llm),
            patch.object(
                session_router,
                "commit_learning_memory",
                side_effect=commit_memory,
            ),
        ):
            first = await session_router.report(session_id)
            second = await session_router.report(session_id)

        self.assertEqual(first.model_dump(), second.model_dump())
        self.assertEqual(first.overall_score, 1.0)
        llm_grade.assert_awaited_once()
        report_llm.assert_awaited_once()
        self.assertEqual(memory_calls, 1)

    async def test_memory_retry_reuses_successful_grading(self):
        session_id = "memory-retry"
        session_service.sessions[session_id] = _completed_session(session_id)
        llm_grade = AsyncMock(return_value=_feedback())
        successful_commits = 0

        async def successful_commit(*args, on_core_written=None, **kwargs):
            nonlocal successful_commits
            successful_commits += 1
            if on_core_written is not None:
                on_core_written()

        with patch.object(grader_service, "_llm_grade", llm_grade):
            with patch.object(
                session_router,
                "commit_learning_memory",
                AsyncMock(side_effect=RuntimeError("memory unavailable")),
            ):
                with self.assertRaisesRegex(RuntimeError, "memory unavailable"):
                    await session_router.grade(session_id)

            with patch.object(
                session_router,
                "commit_learning_memory",
                side_effect=successful_commit,
            ):
                grading = await session_router.grade(session_id)

        self.assertEqual(grading.correct, 1)
        llm_grade.assert_awaited_once()
        self.assertEqual(successful_commits, 1)

    async def test_report_retry_does_not_repeat_grading_or_memory(self):
        session_id = "report-retry"
        session_service.sessions[session_id] = _completed_session(session_id)
        llm_grade = AsyncMock(return_value=_feedback())
        report_llm = AsyncMock(
            side_effect=[RuntimeError("report unavailable"), _report_response()]
        )
        memory_calls = 0

        async def commit_memory(*args, on_core_written=None, **kwargs):
            nonlocal memory_calls
            memory_calls += 1
            if on_core_written is not None:
                on_core_written()

        with (
            patch.object(grader_service, "_llm_grade", llm_grade),
            patch.object(report_service, "llm_parse", report_llm),
            patch.object(
                session_router,
                "commit_learning_memory",
                side_effect=commit_memory,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "report unavailable"):
                await session_router.report(session_id)
            report = await session_router.report(session_id)

        self.assertEqual(report.overall_score, 1.0)
        llm_grade.assert_awaited_once()
        self.assertEqual(report_llm.await_count, 2)
        self.assertEqual(memory_calls, 1)

    async def test_concurrent_grade_and_report_share_canonical_state(self):
        session_id = "grade-report-concurrent"
        session_service.sessions[session_id] = _completed_session(session_id)
        llm_grade = AsyncMock(return_value=_feedback())
        report_llm = AsyncMock(return_value=_report_response())
        memory_calls = 0

        async def commit_memory(*args, on_core_written=None, **kwargs):
            nonlocal memory_calls
            memory_calls += 1
            if on_core_written is not None:
                on_core_written()

        with (
            patch.object(grader_service, "_llm_grade", llm_grade),
            patch.object(report_service, "llm_parse", report_llm),
            patch.object(
                session_router,
                "commit_learning_memory",
                side_effect=commit_memory,
            ),
        ):
            grading, learning_report = await asyncio.gather(
                session_router.grade(session_id),
                session_router.report(session_id),
            )

        self.assertEqual(grading.score, learning_report.overall_score)
        llm_grade.assert_awaited_once()
        report_llm.assert_awaited_once()
        self.assertEqual(memory_calls, 1)

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

    async def test_missing_grade_does_not_allocate_router_lock(self):
        with self.assertRaises(HTTPException) as raised:
            await session_router.grade("missing-session")
        self.assertEqual(raised.exception.status_code, 404)
        self.assertNotIn("missing-session", session_router._answer_locks)

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


if __name__ == "__main__":
    unittest.main()
