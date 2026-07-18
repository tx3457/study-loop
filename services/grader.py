import asyncio
from pathlib import Path

from dotenv import load_dotenv

from models.grader import AIFeedback, QuestionGrade, GradingReport
from models.quiz import Question
from services.llm import structured_client as client, structured_model as model
from services.session import answers_match, sessions

load_dotenv(Path(__file__).parent.parent / ".env")

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

    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": GRADER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
题目：{question.question}{options_text}
正确答案：{question.answer}
参考解析：{question.explanation}

学生答案：{user_answer}

请批改并给出个性化讲解。
""",
            },
        ],
        response_format=AIFeedback,
    )
    return response.choices[0].message.parsed


async def grade_session(session_id: str) -> GradingReport:
    session = sessions.get(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")
    if session.status != "completed":
        raise ValueError("Session not completed yet — submit all answers first")

    questions = session.questions
    user_answers = session.user_answers

    # 先做字符串比较，确定哪些题需要 LLM 批改
    # short_answer 始终调用 LLM（字符串比较不适用于自由文本）
    needs_llm: list[int] = []
    string_correct: list[bool] = []
    for i, (q, ans) in enumerate(zip(questions, user_answers)):
        if getattr(q, "type", "choice") == "short_answer":
            needs_llm.append(i)
            string_correct.append(False)  # 由 LLM 决定
        else:
            is_correct = answers_match(q, ans)
            string_correct.append(is_correct)
            if not is_correct:
                needs_llm.append(i)

    # 并发调用 LLM 批改所有需要批改的题
    llm_tasks = [_llm_grade(questions[i], user_answers[i]) for i in needs_llm]
    llm_results: list[AIFeedback] = await asyncio.gather(*llm_tasks)
    llm_map: dict[int, AIFeedback] = dict(zip(needs_llm, llm_results))

    # 组装逐题结果
    grades: list[QuestionGrade] = []
    correct_count = 0
    for i, (q, ans) in enumerate(zip(questions, user_answers)):
        if i in llm_map:
            fb = llm_map[i]
            is_correct = fb.is_correct
            ai_feedback = fb.feedback
            knowledge_gap = fb.knowledge_gap
        else:
            is_correct = string_correct[i]
            ai_feedback = None
            knowledge_gap = None

        if is_correct:
            correct_count += 1

        grades.append(
            QuestionGrade(
                index=i,
                question=q.question,
                user_answer=ans,
                correct_answer=q.answer,
                is_correct=is_correct,
                ai_feedback=ai_feedback,
                knowledge_gap=knowledge_gap,
            )
        )

    total = len(questions)
    return GradingReport(
        session_id=session_id,
        total=total,
        correct=correct_count,
        score=round(correct_count / total, 2) if total > 0 else 0.0,
        grades=grades,
    )
