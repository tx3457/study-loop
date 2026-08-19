import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from dotenv import load_dotenv

from models.grader import AIFeedback, QuestionGrade, GradingReport
from models.quiz import Question
from models.session import QuizSession
from services.llm import llm_parse, structured_client as client, structured_model as model
from services.session import answers_match, sessions

load_dotenv(Path(__file__).parent.parent / ".env")

_grade_locks: dict[str, asyncio.Lock] = {}

GRADER_SYSTEM_PROMPT = """
你是一位耐心的教学助手，专门帮助学生从错误中学习。

你的任务：
1. 判断学生答案是否正确（对于简答题，语义正确即可）
2. 针对学生的具体错误给出个性化讲解，说明哪里错了以及为什么
3. 用一句话概括该错误暴露的知识盲点

注意：feedback 要直接对学生说话，指出其答案的具体问题，而不是泛泛解释正确答案。
"""


async def _llm_grade(question: Question, user_answer: str) -> AIFeedback:
    """调用 LLM 对单道题进行批改"""
    options_text = ""
    if question.options:
        options_text = f"\n选项：{', '.join(question.options)}"
    evaluation_instruction = (
        "请根据语义判断简答题是否正确。"
        if question.type == "short_answer"
        else "该客观题已由程序判定为错误；is_correct 必须为 false，仅补充讲解与知识盲点。"
    )

    response = await llm_parse(
        messages=[
            {"role": "system", "content": GRADER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
题目：{question.question}{options_text}
正确答案：{question.answer}
参考解析：{question.explanation}

学生答案：{user_answer}

批改要求：{evaluation_instruction}

请批改并给出个性化讲解。
""",
            },
        ],
        response_format=AIFeedback,
        client=client,
        model=model,
    )
    return response.choices[0].message.parsed


async def grade_quiz_session(
    session: QuizSession,
    *,
    checkpoint: Callable[[], Awaitable[None]] | None = None,
) -> GradingReport:
    """Grade an explicitly owned session and optionally persist partial caches."""
    if session.status != "completed":
        raise ValueError("Session not completed yet — submit all answers first")
    if session.grading_report is not None:
        return session.grading_report.model_copy(deep=True)

    questions = session.questions
    user_answers = session.user_answers
    if len(user_answers) != len(questions):
        raise ValueError("Session answers are incomplete")

    needs_llm: list[int] = []
    deterministic_changed = False
    for index, (question, user_answer) in enumerate(zip(questions, user_answers)):
        if index in session.question_grades:
            continue

        deterministic_correct = answers_match(question, user_answer)
        if question.type == "short_answer" or not deterministic_correct:
            needs_llm.append(index)
            continue

        session.question_grades[index] = QuestionGrade(
            index=index,
            question=question.question,
            user_answer=user_answer,
            correct_answer=question.answer,
            is_correct=True,
        )
        deterministic_changed = True

    if deterministic_changed and checkpoint is not None:
        await checkpoint()

    # Cache every successful item independently. If one provider call fails,
    # a retry only evaluates the still-missing questions.
    llm_tasks = [_llm_grade(questions[index], user_answers[index]) for index in needs_llm]
    llm_results = await asyncio.gather(*llm_tasks, return_exceptions=True)
    failures: list[BaseException] = []
    llm_changed = False
    for index, result in zip(needs_llm, llm_results):
        if isinstance(result, BaseException):
            failures.append(result)
            continue

        question = questions[index]
        user_answer = user_answers[index]
        # The model judges free-text semantics, but may never override the
        # deterministic correctness of choice and true/false questions.
        is_correct = (
            result.is_correct
            if question.type == "short_answer"
            else answers_match(question, user_answer)
        )
        session.question_grades[index] = QuestionGrade(
            index=index,
            question=question.question,
            user_answer=user_answer,
            correct_answer=question.answer,
            is_correct=is_correct,
            ai_feedback=result.feedback,
            knowledge_gap=result.knowledge_gap,
        )
        llm_changed = True

    if llm_changed and checkpoint is not None:
        await checkpoint()
    if failures:
        raise failures[0]

    grades = [
        session.question_grades[index].model_copy(deep=True)
        for index in range(len(questions))
    ]
    correct_count = sum(grade.is_correct for grade in grades)
    total = len(questions)
    report = GradingReport(
        session_id=session.session_id,
        total=total,
        correct=correct_count,
        score=round(correct_count / total, 2) if total > 0 else 0.0,
        grades=grades,
    )
    session.grading_report = report.model_copy(deep=True)
    if checkpoint is not None:
        await checkpoint()
    return report


async def grade_session(session_id: str) -> GradingReport:
    """Legacy wrapper for Adaptive/Tutor sessions stored in memory."""
    session = sessions.get(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")
    if session.status != "completed":
        raise ValueError("Session not completed yet — submit all answers first")

    lock = _grade_locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        session = sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        return await grade_quiz_session(session)
