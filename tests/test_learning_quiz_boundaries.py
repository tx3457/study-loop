"""Regression coverage for document-grounded quiz and session boundaries."""

import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from chromadb.errors import InternalError, NotFoundError
from fastapi.testclient import TestClient
from pydantic import ValidationError

import routers.session as session_router
import routers.adaptive as adaptive_router
import services.grader as grader_service
import services.learning_path as learning_path
import services.rag as rag
import services.session as session_service
from main import app
from models.learning_path import PathBrief
from models.grader import AIFeedback
from models.quiz import Question, QuizResponse
from models.session import QuizSession, SessionStartRequest


def _brief() -> PathBrief:
    return PathBrief(
        title="RAG 学习路径",
        scope="RAG 基础",
        level="beginner",
        target_count=3,
        keywords=["retrieval", "generation"],
    )


def _question(*, question: str = "RAG 是什么？") -> Question:
    return Question(
        question=question,
        options=["检索增强生成", "关系型数据库"],
        answer="检索增强生成",
        explanation="原文将 RAG 定义为检索增强生成。",
        source="RAG combines retrieval and generation.",
    )


def _session(session_id: str, *, status: str) -> QuizSession:
    answers = ["检索增强生成"] if status == "completed" else []
    return QuizSession(
        session_id=session_id,
        document_id="notes.md",
        user_id="user-1",
        questions=[_question()],
        user_answers=answers,
        status=status,
    )


class TestLearningPathEvidenceBoundary(unittest.IsolatedAsyncioTestCase):
    async def test_missing_document_stops_before_brief_provider_call(self):
        extract_brief = AsyncMock()
        with (
            patch.object(
                learning_path,
                "ensure_document_available",
                AsyncMock(side_effect=NotFoundError("missing")),
                create=True,
            ),
            patch.object(learning_path, "extract_brief", extract_brief),
        ):
            with self.assertRaises(NotFoundError):
                await learning_path.generate_learning_path("missing.md")

        extract_brief.assert_not_awaited()

    async def test_all_retrieval_failures_stop_before_synthesis(self):
        synthesize = AsyncMock()
        with (
            patch.object(
                learning_path,
                "ensure_document_available",
                AsyncMock(),
            ),
            patch.object(
                learning_path, "extract_brief", AsyncMock(return_value=_brief())
            ),
            patch.object(
                learning_path,
                "hybrid_query_document",
                AsyncMock(side_effect=RuntimeError("vector store unavailable")),
            ),
            patch.object(learning_path, "synthesize", synthesize),
        ):
            with self.assertRaises(RuntimeError):
                await learning_path.generate_learning_path("notes.md")

        synthesize.assert_not_awaited()

    async def test_empty_retrieval_evidence_stops_before_synthesis(self):
        synthesize = AsyncMock()
        with (
            patch.object(
                learning_path,
                "ensure_document_available",
                AsyncMock(),
            ),
            patch.object(
                learning_path, "extract_brief", AsyncMock(return_value=_brief())
            ),
            patch.object(
                learning_path,
                "hybrid_query_document",
                AsyncMock(return_value={"documents": [[]], "ids": [[]]}),
            ),
            patch.object(learning_path, "synthesize", synthesize),
        ):
            with self.assertRaises(RuntimeError):
                await learning_path.generate_learning_path("notes.md")

        synthesize.assert_not_awaited()


class TestQuizProviderBoundary(unittest.IsolatedAsyncioTestCase):
    async def test_question_generation_uses_retrying_parse_entrypoint(self):
        parsed = QuizResponse(questions=[_question()])
        llm_parse = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
            )
        )
        raw_parse = AsyncMock(side_effect=AssertionError("raw provider call used"))

        with (
            patch.object(rag, "llm_parse", llm_parse, create=True),
            patch.object(rag.client.beta.chat.completions, "parse", raw_parse),
        ):
            result = await rag.generate_question_from_chunks(
                ["RAG combines retrieval and generation."],
                count=1,
                difficulty="easy",
                type="choice",
            )

        self.assertEqual(len(result.questions), 1)
        llm_parse.assert_awaited_once()
        raw_parse.assert_not_awaited()

    async def test_empty_provider_quiz_does_not_create_session(self):
        req = SessionStartRequest(
            document_id="notes.md",
            description="RAG",
            count=1,
            difficulty="easy",
            type="choice",
            user_id="user-1",
        )
        before = dict(session_service.sessions)

        with (
            patch.object(
                session_service,
                "_ensure_document_available",
                AsyncMock(),
                create=True,
            ),
            patch.object(
                session_service, "get_user_profile", AsyncMock(return_value=None)
            ),
            patch.object(
                session_service,
                "generate_question",
                AsyncMock(return_value=QuizResponse(questions=[])),
            ),
        ):
            with self.assertRaises(RuntimeError):
                await session_service.start_session(req)

        self.assertEqual(session_service.sessions, before)

    async def test_blank_provider_question_does_not_create_session(self):
        req = SessionStartRequest(
            document_id="notes.md",
            description="RAG",
            count=1,
            difficulty="easy",
            type="choice",
            user_id="user-1",
        )
        before = dict(session_service.sessions)

        with (
            patch.object(
                session_service,
                "_ensure_document_available",
                AsyncMock(),
                create=True,
            ),
            patch.object(
                session_service, "get_user_profile", AsyncMock(return_value=None)
            ),
            patch.object(
                session_service,
                "generate_question",
                AsyncMock(
                    return_value=QuizResponse(questions=[_question(question=" ")])
                ),
            ),
        ):
            with self.assertRaises(RuntimeError):
                await session_service.start_session(req)

        self.assertEqual(session_service.sessions, before)

    async def test_objective_answer_must_resolve_to_an_option(self):
        req = SessionStartRequest(
            document_id="notes.md",
            description="RAG",
            count=1,
            difficulty="easy",
            type="choice",
            user_id="user-1",
        )
        invalid = _question()
        invalid.answer = "Z"
        before = dict(session_service.sessions)

        with (
            patch.object(
                session_service,
                "_ensure_document_available",
                AsyncMock(),
            ),
            patch.object(
                session_service, "get_user_profile", AsyncMock(return_value=None)
            ),
            patch.object(
                session_service,
                "generate_question",
                AsyncMock(return_value=QuizResponse(questions=[invalid])),
            ),
        ):
            with self.assertRaises(session_service.InvalidQuizResponseError):
                await session_service.start_session(req)

        self.assertEqual(session_service.sessions, before)


class TestAdaptiveStartBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)

    def setUp(self):
        adaptive_router._sessions.clear()
        adaptive_router.sessions.clear()

    def tearDown(self):
        adaptive_router._sessions.clear()
        adaptive_router.sessions.clear()

    @staticmethod
    def _payload() -> dict:
        return {
            "user_id": "user-1",
            "document_id": "missing.md",
            "goal": "学习 RAG",
        }

    def test_missing_document_is_404_before_safety_profile_or_provider_calls(self):
        injection = AsyncMock(return_value=(False, ""))
        mastery = AsyncMock(return_value=None)
        decision = AsyncMock()
        with (
            patch.object(
                adaptive_router,
                "ensure_document_available",
                AsyncMock(side_effect=NotFoundError("missing")),
                create=True,
            ),
            patch.object(adaptive_router, "check_injection", injection),
            patch.object(adaptive_router, "get_mastery", mastery),
            patch.object(adaptive_router, "decide_next_step", decision),
        ):
            response = self.client.post(
                "/agent/adaptive/start",
                json=self._payload(),
            )

        self.assertEqual(response.status_code, 404, response.text)
        self.assertEqual(response.json(), {"detail": "文档不存在"})
        injection.assert_not_awaited()
        mastery.assert_not_awaited()
        decision.assert_not_awaited()

    def test_document_store_failure_is_redacted_503(self):
        with patch.object(
            adaptive_router,
            "ensure_document_available",
            AsyncMock(side_effect=InternalError("storage-secret")),
            create=True,
        ):
            response = self.client.post(
                "/agent/adaptive/start",
                json=self._payload(),
            )

        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json(), {"detail": "文档存储暂时不可用"})
        self.assertNotIn("storage-secret", response.text)


class TestAdaptiveProviderBoundary(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        adaptive_router.sessions.clear()

    async def asyncTearDown(self):
        adaptive_router.sessions.clear()

    async def test_empty_provider_quiz_is_not_persisted(self):
        asess = adaptive_router.AdaptiveSession(
            adaptive_session_id="adapt-empty",
            user_id="user-1",
            document_id="notes.md",
            goal="学习 RAG",
            turn=1,
            history=[],
        )
        decision = adaptive_router.NextStepDecision(
            action="continue",
            topic="RAG",
            count=1,
            question_type="choice",
            reason="测试",
        )
        with patch.object(
            adaptive_router,
            "generate_question",
            AsyncMock(return_value=QuizResponse(questions=[])),
        ):
            with self.assertRaises(session_service.InvalidQuizResponseError):
                await adaptive_router._serve_turn(asess, decision, turn=1)

        self.assertIsNone(asess.current_quiz_session_id)
        self.assertFalse(adaptive_router.sessions)


class TestChoiceAnswerNormalization(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_sessions = dict(session_service.sessions)
        session_service.sessions.clear()

    def tearDown(self):
        session_service.sessions.clear()
        session_service.sessions.update(self.original_sessions)

    async def test_answer_label_matches_full_option_in_all_score_paths(self):
        session_service.sessions["choice"] = QuizSession(
            session_id="choice",
            document_id="notes.md",
            user_id="user-1",
            questions=[
                Question(
                    question="哪个选项描述了 RAG？",
                    options=["A. 缓存", "B. 排序", "C. 检索增强生成", "D. 压缩"],
                    answer="C",
                    explanation="原文说明 RAG 是检索增强生成。",
                    source="RAG combines retrieval and generation.",
                    type="choice",
                )
            ],
            user_answers=[],
            status="active",
        )
        memory_commit = AsyncMock()

        with patch.object(
            session_service,
            "commit_learning_memory",
            memory_commit,
        ):
            answer = await session_service.submit_answer("choice", "C. 检索增强生成")

        result = await session_service.get_result("choice")
        written_report = memory_commit.await_args.args[1]

        self.assertTrue(answer.correct)
        self.assertEqual(result.correct, 1)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(written_report.correct, 1)
        self.assertTrue(written_report.grades[0].is_correct)
        memory_commit.assert_awaited_once()

    async def test_full_choice_text_does_not_trigger_unnecessary_llm_grading(self):
        session_service.sessions["choice-grade"] = QuizSession(
            session_id="choice-grade",
            document_id="notes.md",
            user_id="user-1",
            questions=[
                Question(
                    question="哪个选项描述了 RAG？",
                    options=["A. 缓存", "B. 排序", "C. 检索增强生成", "D. 压缩"],
                    answer="C",
                    explanation="原文说明 RAG 是检索增强生成。",
                    source="RAG combines retrieval and generation.",
                    type="choice",
                )
            ],
            user_answers=["C. 检索增强生成"],
            status="completed",
        )
        llm_grade = AsyncMock(
            return_value=AIFeedback(
                is_correct=True,
                feedback="正确",
                knowledge_gap="",
            )
        )

        with patch.object(grader_service, "_llm_grade", llm_grade):
            report = await grader_service.grade_session("choice-grade")

        self.assertEqual(report.correct, 1)
        self.assertTrue(report.grades[0].is_correct)
        llm_grade.assert_not_awaited()

    async def test_short_answer_profile_waits_for_semantic_grading(self):
        session_service.sessions["short"] = QuizSession(
            session_id="short",
            document_id="notes.md",
            user_id="user-1",
            questions=[
                Question(
                    question="解释 RAG",
                    options=None,
                    answer="检索增强生成",
                    explanation="语义相同即可。",
                    source="notes.md",
                    type="short_answer",
                )
            ],
            user_answers=[],
            status="active",
        )
        write_back = AsyncMock()

        with patch.object(session_service, "_write_back_profile", write_back):
            response = await session_service.submit_answer(
                "short",
                "先检索资料，再基于资料生成答案",
                question_index=0,
            )

        self.assertTrue(response.is_last)
        self.assertFalse(session_service.sessions["short"].profile_written)
        write_back.assert_not_awaited()

    def test_true_false_option_labels_use_the_same_normalization(self):
        question = Question(
            question="RAG 包含检索步骤。",
            options=["A. 正确", "B. 错误"],
            answer="A",
            explanation="RAG 的 R 代表 retrieval。",
            source="notes.md",
            type="true_false",
        )

        self.assertTrue(session_service.answers_match(question, "A. 正确"))


class TestSessionHttpBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)

    def setUp(self):
        self.original_sessions = dict(session_service.sessions)
        session_service.sessions.clear()

    def tearDown(self):
        session_service.sessions.clear()
        session_service.sessions.update(self.original_sessions)

    @staticmethod
    def _start_payload() -> dict:
        return {
            "document_id": "notes.md",
            "description": "RAG",
            "count": 1,
            "difficulty": "easy",
            "type": "choice",
            "user_id": "user-1",
        }

    def test_start_missing_document_is_404_before_profile_or_provider(self):
        profile = AsyncMock()
        provider = AsyncMock()
        with (
            patch.object(
                session_service,
                "_ensure_document_available",
                AsyncMock(side_effect=NotFoundError("missing")),
            ),
            patch.object(session_service, "get_user_profile", profile),
            patch.object(session_service, "generate_question", provider),
        ):
            response = self.client.post("/session/start", json=self._start_payload())

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "文档不存在"})
        profile.assert_not_awaited()
        provider.assert_not_awaited()

    def test_start_document_store_failure_is_503(self):
        with patch.object(
            session_service,
            "_ensure_document_available",
            AsyncMock(side_effect=InternalError("storage-secret")),
        ):
            response = self.client.post("/session/start", json=self._start_payload())

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "文档存储暂时不可用"})
        self.assertNotIn("storage-secret", response.text)

    def test_missing_session_is_404_for_every_session_endpoint(self):
        requests = (
            (
                "post",
                "/session/missing/answer",
                {"answer": "A", "question_index": 0},
            ),
            ("get", "/session/missing/result", None),
            ("post", "/session/missing/grade", None),
            ("post", "/session/missing/report", None),
        )

        for method, path, payload in requests:
            with self.subTest(path=path):
                request = getattr(self.client, method)
                response = request(path, json=payload) if payload else request(path)
                self.assertEqual(response.status_code, 404)

    def test_session_state_conflicts_are_409(self):
        session_service.sessions["active"] = _session("active", status="active")
        session_service.sessions["completed"] = _session(
            "completed", status="completed"
        )

        requests = (
            (
                "post",
                "/session/completed/answer",
                {"answer": "A", "question_index": 1},
            ),
            ("get", "/session/active/result", None),
            ("post", "/session/active/grade", None),
            ("post", "/session/active/report", None),
        )

        for method, path, payload in requests:
            with self.subTest(path=path):
                request = getattr(self.client, method)
                response = request(path, json=payload) if payload else request(path)
                self.assertEqual(response.status_code, 409)

    def test_provider_validation_error_is_redacted(self):
        try:
            QuizResponse.model_validate(
                {
                    "questions": [
                        {
                            "question": None,
                            "options": [],
                            "answer": "provider-secret-answer",
                            "explanation": "x",
                            "source": "x",
                        }
                    ]
                }
            )
        except ValidationError as error:
            provider_validation_error = error
        else:  # pragma: no cover - guards against an unexpected Pydantic behavior change
            self.fail("expected provider response validation to fail")

        with patch.object(
            session_router,
            "start_session",
            AsyncMock(side_effect=provider_validation_error),
        ):
            response = self.client.post("/session/start", json=self._start_payload())

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "模型返回的题目格式无效"})
        self.assertNotIn("provider-secret-answer", response.text)


if __name__ == "__main__":
    unittest.main()
