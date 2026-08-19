import unittest
import uuid
from unittest.mock import patch

from models.grader import GradingReport, QuestionGrade
from models.quiz import Question


def _wrong_report(session_id: str) -> GradingReport:
    return GradingReport(
        session_id=session_id,
        total=1,
        correct=0,
        score=0.0,
        grades=[
            QuestionGrade(
                index=0,
                question="RRF 的作用是什么？",
                user_answer="只做向量检索",
                correct_answer="融合多路检索排名",
                is_correct=False,
                ai_feedback="RRF 会融合多个有序结果集。",
                knowledge_gap="混合检索",
            )
        ],
    )


def _correct_report(session_id: str) -> GradingReport:
    return GradingReport(
        session_id=session_id,
        total=1,
        correct=1,
        score=1.0,
        grades=[
            QuestionGrade(
                index=0,
                question="RRF 的作用是什么？",
                user_answer="融合多路检索排名",
                correct_answer="融合多路检索排名",
                is_correct=True,
            )
        ],
    )


class TestWrongQuestionMemory(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.user_id = f"wrong-user-{uuid.uuid4()}"
        self.document_id = "retrieval.md"
        self.session_id = f"source-{uuid.uuid4()}"
        self.question = Question(
            question="RRF 的作用是什么？",
            options=["只做向量检索", "融合多路检索排名"],
            answer="融合多路检索排名",
            explanation="RRF 根据各结果的名次做无量纲融合。",
            source="chunk-1",
            type="choice",
        )

    async def test_repeated_write_upserts_one_enriched_wrong_entry(self):
        from services.memory import write_episodic_memory
        from services.wrong_questions import get_wrong_questions

        report = _wrong_report(self.session_id)
        await write_episodic_memory(
            self.user_id,
            report,
            self.document_id,
            questions=[self.question],
        )
        await write_episodic_memory(
            self.user_id,
            report,
            self.document_id,
            questions=[self.question],
        )

        bank = await get_wrong_questions(self.document_id, user_id=self.user_id)

        self.assertEqual(bank.total, 1)
        entry = bank.entries[0]
        self.assertEqual(entry.entry_id, f"{self.session_id}:0")
        self.assertEqual(entry.options, self.question.options)
        self.assertEqual(entry.question_type, "choice")
        self.assertEqual(entry.explanation, self.question.explanation)

    async def test_repractice_creates_answerable_session_for_same_user(self):
        from services.memory import write_episodic_memory
        import services.wrong_questions as wrong_questions

        await write_episodic_memory(
            self.user_id,
            _wrong_report(self.session_id),
            self.document_id,
            questions=[self.question],
        )

        sessions = {}
        with patch.object(wrong_questions, "_get_sessions", return_value=sessions):
            practice_id, views = await wrong_questions.start_repractice(
                self.document_id,
                user_id=self.user_id,
            )

        practice = sessions[practice_id]
        self.assertEqual(practice.user_id, self.user_id)
        self.assertEqual(practice.document_id, self.document_id)
        self.assertEqual(practice.questions[0].type, "choice")
        self.assertEqual(practice.questions[0].options, self.question.options)
        self.assertEqual(views[0].question, self.question.question)

    async def test_wrong_questions_are_isolated_by_user(self):
        from services.memory import write_episodic_memory
        from services.wrong_questions import get_wrong_questions

        await write_episodic_memory(
            self.user_id,
            _wrong_report(self.session_id),
            self.document_id,
            questions=[self.question],
        )

        other = await get_wrong_questions(
            self.document_id,
            user_id=f"other-{uuid.uuid4()}",
        )
        self.assertEqual(other.total, 0)

    async def test_resolved_practice_disappears_but_keeps_audit_event(self):
        import services.memory as memory
        from services.wrong_questions import get_wrong_questions

        source_error_id = f"{self.session_id}:0"
        await memory.write_episodic_memory(
            self.user_id,
            _wrong_report(self.session_id),
            self.document_id,
            questions=[self.question],
        )
        practice_question = self.question.model_copy(
            update={"source": f"wrong-question:{source_error_id}"}
        )
        await memory.write_episodic_memory(
            self.user_id,
            _correct_report("resolved-session"),
            self.document_id,
            questions=[practice_question],
        )

        bank = await get_wrong_questions(self.document_id, user_id=self.user_id)
        stored = memory.store.get(
            ("users", self.user_id, "error_log"),
            source_error_id,
        )

        self.assertEqual(bank.total, 0)
        self.assertIsNotNone(stored)
        self.assertEqual(stored.value["resolved_session_id"], "resolved-session")
        self.assertTrue(stored.value["resolved_at"])

    async def test_legacy_store_key_can_be_resolved_without_creating_a_copy(self):
        import services.memory as memory
        from services.wrong_questions import get_wrong_questions

        legacy_key = "err_legacy_without_payload_id"
        memory.store.put(
            ("users", self.user_id, "error_log"),
            legacy_key,
            {
                "session_id": "legacy-session",
                "document_id": self.document_id,
                "question": self.question.question,
                "options": self.question.options,
                "question_type": self.question.type,
                "correct_answer": self.question.answer,
                "explanation": self.question.explanation,
                "user_answer": "只做向量检索",
            },
        )
        bank = await get_wrong_questions(self.document_id, user_id=self.user_id)
        self.assertEqual(bank.entries[0].entry_id, legacy_key)

        practice_question = self.question.model_copy(
            update={"source": f"wrong-question:{legacy_key}"}
        )
        await memory.write_episodic_memory(
            self.user_id,
            _correct_report("legacy-resolved-session"),
            self.document_id,
            questions=[practice_question],
        )

        self.assertEqual(
            (await get_wrong_questions(self.document_id, user_id=self.user_id)).total,
            0,
        )
        self.assertIsNotNone(
            memory.store.get(("users", self.user_id, "error_log"), legacy_key)
        )

    async def test_more_than_store_default_limit_is_returned(self):
        from services.memory import append_error
        from services.wrong_questions import get_wrong_questions

        for index in range(15):
            await append_error(
                self.user_id,
                {
                    "error_id": f"many:{index}",
                    "session_id": f"many-session-{index}",
                    "question_index": index,
                    "document_id": self.document_id,
                    "question": f"题目 {index}",
                    "correct_answer": f"答案 {index}",
                    "user_answer": "错误答案",
                },
            )

        bank = await get_wrong_questions(self.document_id, user_id=self.user_id)
        self.assertEqual(bank.total, 15)


if __name__ == "__main__":
    unittest.main()
