from pydantic import BaseModel


class AIFeedback(BaseModel):
    """LLM 结构化批改结果"""
    is_correct: bool      # 用于 short_answer 语义判断；choice/true_false 由字符串比较决定
    feedback: str         # 针对学生具体答案的个性化讲解
    knowledge_gap: str    # 该错误暴露的知识盲点


class QuestionGrade(BaseModel):
    index: int
    question: str
    user_answer: str
    correct_answer: str
    is_correct: bool
    ai_feedback: str | None = None    # 答对时为 None
    knowledge_gap: str | None = None  # 答对时为 None


class GradingReport(BaseModel):
    session_id: str
    total: int
    correct: int
    score: float
    grades: list[QuestionGrade]
