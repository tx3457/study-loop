"""HTTP recovery contracts for durable wrong-question practice sessions."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from main import app
from models.quiz import Question
from models.session import QuizSession
import routers.session as session_router
import routers.wrong_questions as wrong_questions_router
from services.quiz_sessions import QuizSessionStore


def _practice_session() -> QuizSession:
    return QuizSession(
        session_id="durable-wrong-practice",
        document_id="notes.md",
        user_id="user-1",
        questions=[
            Question(
                question="What does RRF do?",
                options=["A. Fuse ranked lists", "B. Store embeddings"],
                answer="A",
                explanation="PRIVATE-WRONG-QUESTION-EXPLANATION",
                source="PRIVATE-WRONG-QUESTION-SOURCE",
                type="choice",
            )
        ],
        user_answers=[],
        status="active",
    )


class TestDurableWrongQuestionPracticeHttp(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.tempdir.name) / "quiz-sessions.sqlite3"
        self.quiz_store = QuizSessionStore(sqlite_path=str(self.database_path))
        self.patches = (
            patch.object(wrong_questions_router, "quiz_sessions", self.quiz_store),
            patch.object(session_router, "quiz_sessions", self.quiz_store),
        )
        for active_patch in self.patches:
            active_patch.start()
        self.client = TestClient(app, raise_server_exceptions=False)

    def tearDown(self) -> None:
        self.client.close()
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.tempdir.cleanup()

    def test_practice_start_replay_and_reload_use_one_durable_session(self) -> None:
        prepare = AsyncMock(return_value=_practice_session())
        headers = {"Idempotency-Key": "wrong-practice-start-key"}
        # user_id 不再是查询参数——身份由服务端解析，调用方改不了它。
        # 「同键不同载荷必须冲突」这条契约仍然要验，改用调用方真能改的字段。
        path = "/wrong-questions/notes.md/practice"

        with patch.object(
            wrong_questions_router,
            "prepare_repractice_session",
            prepare,
        ):
            first = self.client.post(path, headers=headers)
            replay = self.client.post(path, headers=headers)
            mismatch = self.client.post(
                "/wrong-questions/other-notes.md/practice",
                headers=headers,
            )

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(replay.json(), first.json())
        prepare.assert_awaited_once()
        self.assertEqual(mismatch.status_code, 409, mismatch.text)
        self.assertEqual(mismatch.json()["reason"], "payload_mismatch")

        session_id = first.json()["session_id"]
        stored = asyncio.run(self.quiz_store.inspect(session_id))
        self.assertEqual(stored.aggregate.origin, "wrong_question")

        reopened = QuizSessionStore(sqlite_path=str(self.database_path))
        with (
            patch.object(wrong_questions_router, "quiz_sessions", reopened),
            patch.object(session_router, "quiz_sessions", reopened),
        ):
            snapshot = self.client.get(f"/session/{session_id}")
            with patch.object(
                session_router,
                "write_objective_profile",
                AsyncMock(),
            ):
                answer = self.client.post(
                    f"/session/{session_id}/answer",
                    json={"answer": "A. Fuse ranked lists", "question_index": 0},
                    headers={"Idempotency-Key": "wrong-practice-answer-key"},
                )
            result = self.client.get(f"/session/{session_id}/result")

        self.assertEqual(snapshot.status_code, 200, snapshot.text)
        self.assertEqual(snapshot.json()["origin"], "wrong_question")
        self.assertNotIn("PRIVATE-WRONG-QUESTION-EXPLANATION", snapshot.text)
        self.assertNotIn("PRIVATE-WRONG-QUESTION-SOURCE", snapshot.text)
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertTrue(answer.json()["is_last"])
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["session_id"], session_id)


if __name__ == "__main__":
    unittest.main()
