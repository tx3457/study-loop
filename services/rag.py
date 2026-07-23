from pathlib import Path

from dotenv import load_dotenv

from models.quiz import QuizResponse
from services.llm import (
    llm_parse,
    structured_client as client,
    structured_model as model,
)
from services.tracing import traceable
from services.vectorstore import retrieve_with_rewrite

load_dotenv(Path(__file__).parent.parent / ".env")


SYSTEM_PROMPT = """
你是出题专家，严格根据提供的原文段落生成题目。

题型规则：
- choice（选择题）：提供4个选项，answer 填正确选项的内容
- true_false（判断题）：options 填 ["正确", "错误"]，answer 填 "正确" 或 "错误"
- short_answer（简答题）：options 留空列表 []，answer 填简短答案

难度规则（0.0-1.0 连续值）：
- 0.0-0.3：基础——考查单一概念，答案直接出自原文
- 0.3-0.6：进阶——需要理解和归纳，涉及原文推断
- 0.6-0.8：深入——需要综合多个概念，涉及分析和对比
- 0.8-1.0：专家——需要深度分析，涉及原理推导或批判性思考

要求：
- 题目必须基于原文内容，不得编造
- explanation 需引用原文依据
- source 填写题目对应的原文片段
"""


@traceable(
    name="quiz_generation",
    run_type="chain",
    metadata={"model": "structured_output"},
)
async def generate_question_from_chunks(
    chunks: list[str],
    count: int,
    difficulty: str,
    type: str,
    difficulty_score: float | None = None,
    weak_points: list[str] | None = None,
    reflected_message: str = "",
):
    """接受预取的 chunks，直接生成题目。供评估脚本对比不同检索策略使用。

    reflected_message:上一轮被 critic 拒绝时,
    把格式化的拒绝原因注入下一轮 prompt,让 LLM 明确改进方向。
    """
    chunks_text = "\n\n".join(chunks)

    # 难度描述：有 difficulty_score 时用连续值，否则用文字
    difficulty_desc = (
        f"{difficulty_score:.2f}/1.0" if difficulty_score is not None else difficulty
    )

    # 自适应上下文：有薄弱知识点时注入
    adaptive_section = ""
    if weak_points:
        gaps_str = "、".join(weak_points[:5])
        adaptive_section = f"\n- 用户薄弱知识点（优先围绕这些知识点出题）：{gaps_str}"

    # Reflection 回灌:critic 拒绝后的反思文本,优先级最高,放最显眼位置
    reflection_section = ""
    if reflected_message:
        reflection_section = f"\n\n---\n{reflected_message}\n---\n"

    message = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""原文段落：
{chunks_text}

出题要求：
- 数量：{count} 道 {type} 题
- 难度：{difficulty_desc}{adaptive_section}{reflection_section}""",
        },
    ]
    question = await llm_parse(
        message,
        QuizResponse,
        client=client,
        model=model,
    )
    quiz = question.choices[0].message.parsed
    for q in quiz.questions:
        q.type = type  # 注入题型，供 grader 判断是否需要语义批改
    return quiz


async def generate_question(
    document_id: str,
    description: str,
    count: int,
    difficulty: str,
    type: str,
    difficulty_score: float | None = None,
    weak_points: list[str] | None = None,
    reflected_message: str = "",
):
    result = await retrieve_with_rewrite(document_id, description)
    chunks = result["documents"][0]
    return await generate_question_from_chunks(
        chunks,
        count,
        difficulty,
        type,
        difficulty_score=difficulty_score,
        weak_points=weak_points,
        reflected_message=reflected_message,
    )
