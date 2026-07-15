import uuid
from models.wrong_questions import WrongEntry, WrongQuestionBank
from models.grader import GradingReport
from models.quiz import Question
from models.session import QuizSession, QuestionView, SessionStartRequest

# 内存错题库：{document_id: {entry_id: WrongEntry}}
# Phase 3 升级为持久化存储
wrong_bank: dict[str, dict[str, WrongEntry]] = {}

# 避免循环导入，在函数内部 import sessions
def _get_sessions():
    from services.session import sessions
    return sessions


def collect_wrong_answers(session_id: str, report: GradingReport) -> int:
    """从批改报告中提取错题，写入错题库。返回新增条数。"""
    sessions = _get_sessions()
    session = sessions.get(session_id)
    if not session:
        return 0

    doc_id = session.document_id
    if doc_id not in wrong_bank:
        wrong_bank[doc_id] = {}

    added = 0
    for g in report.grades:
        if not g.is_correct:
            entry_id = str(uuid.uuid4())
            wrong_bank[doc_id][entry_id] = WrongEntry(
                entry_id=entry_id,
                document_id=doc_id,
                question=g.question,
                options=next(
                    (q.options for q in session.questions if q.question == g.question),
                    None,
                ),
                correct_answer=g.correct_answer,
                explanation=next(
                    (q.explanation for q in session.questions if q.question == g.question),
                    "",
                ),
                user_answer=g.user_answer,
                knowledge_gap=g.knowledge_gap,
                session_id=session_id,
            )
            added += 1
    return added


def get_wrong_questions(document_id: str) -> WrongQuestionBank:
    entries = list(wrong_bank.get(document_id, {}).values())
    return WrongQuestionBank(
        document_id=document_id,
        total=len(entries),
        entries=entries,
    )


def start_repractice(document_id: str) -> tuple[str, list[QuestionView]]:
    """用错题库里的题目创建一个新的答题会话。"""
    sessions = _get_sessions()
    entries = list(wrong_bank.get(document_id, {}).values())
    if not entries:
        raise ValueError(f"No wrong questions for document '{document_id}'")

    # WrongEntry → Question
    questions = [
        Question(
            question=e.question,
            options=e.options,
            answer=e.correct_answer,
            explanation=e.explanation,
            source="wrong-question-bank",
        )
        for e in entries
    ]

    session_id = str(uuid.uuid4())
    sessions[session_id] = QuizSession(
        session_id=session_id,
        document_id=document_id,
        questions=questions,
        user_answers=[],
        status="active",
    )

    questions_view = [
        QuestionView(index=i, question=q.question, options=q.options)
        for i, q in enumerate(questions)
    ]
    return session_id, questions_view
