from pydantic import BaseModel
from models.quiz import Question


class SessionStartRequest(BaseModel):
    document_id: str
    description: str
    count: int = 5
    difficulty: str = "medium"   # easy / medium / hard
    type: str = "choice"         # choice / true_false / short_answer
    user_id: str = "default_user"


class QuestionView(BaseModel):
    """题目视图：对用户隐藏答案和解析"""
    index: int
    question: str
    options: list[str] | None = None


class AnswerRequest(BaseModel):
    answer: str


class AnswerResult(BaseModel):
    correct: bool
    correct_answer: str
    explanation: str
    is_last: bool
    next_index: int | None = None


class SessionResult(BaseModel):
    session_id: str
    document_id: str
    total: int
    correct: int
    score: float              # 0.0 – 1.0
    details: list[dict]       # 逐题明细


class QuizSession(BaseModel):
    session_id: str
    document_id: str
    user_id: str
    questions: list[Question]
    user_answers: list[str]
    status: str               # "active" / "completed"
    profile_written: bool = False   # 画像写回幂等标记（submit_answer 与 /grade 两条路径去重）
