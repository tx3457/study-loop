import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from models.report import LearningReport, _ReportCore
from services.session import sessions
from services.grader import grade_session

load_dotenv(Path(__file__).parent.parent / ".env")

# 报告核心生成用 json_schema 结构化输出 → 走 structured 供应商
from services.llm import structured_client as client, structured_model as model

REPORT_SYSTEM_PROMPT = """
你是学习评估专家。根据学生的答题记录，生成一份结构化学习评估报告。

要求：
- 从题目内容中识别所有涉及的知识点/主题（通常 3-6 个）
- 按知识点统计 question_count 和 correct_count，计算 mastery_pct（0-100）
- strengths：mastery_pct ≥ 70% 的知识点名称列表
- weaknesses：mastery_pct < 70% 的知识点名称列表
- recommendations：针对薄弱点给出 3-5 条具体可行的学习建议
- summary：100字以内的总体评语，指出本次学习的主要收获和改进方向
"""


async def generate_report(session_id: str) -> LearningReport:
    session = sessions.get(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")
    if session.status != "completed":
        raise ValueError("Session not completed yet — submit all answers first")

    # 获取批改结果（含 AI 讲解）
    grading = await grade_session(session_id)

    # 构建答题记录文本
    records = []
    for g in grading.grades:
        status = "✓" if g.is_correct else "✗"
        record = f"{g.index + 1}. [{status}] 题目：{g.question} | 学生答案：{g.user_answer} | 正确答案：{g.correct_answer}"
        if g.knowledge_gap:
            record += f" | 知识盲点：{g.knowledge_gap}"
        records.append(record)
    records_text = "\n".join(records)

    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": f"""
学生答题记录（共 {grading.total} 题，答对 {grading.correct} 题）：

{records_text}

请生成学习评估报告。
"""},
        ],
        response_format=_ReportCore,
    )
    core = response.choices[0].message.parsed

    return LearningReport(
        session_id=session_id,
        document_id=session.document_id,
        overall_score=grading.score,
        topic_mastery=core.topic_mastery,
        strengths=core.strengths,
        weaknesses=core.weaknesses,
        recommendations=core.recommendations,
        summary=core.summary,
    )
