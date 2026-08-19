from typing import Literal

from pydantic import BaseModel, Field, field_validator

from models.grader import GradingReport, QuestionGrade
from models.quiz import Question
from models.report import LearningReport


class SessionStartRequest(BaseModel):
    document_id: str
    description: str
    count: int = Field(default=5, ge=1, le=10)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    type: Literal["choice", "true_false", "short_answer"] = "choice"
    user_id: str = "default_user"


class QuestionView(BaseModel):
    """题目视图：对用户隐藏答案和解析"""

    index: int
    question: str
    options: list[str] | None = None
    type: str = "choice"


class AnswerRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=4000)
    question_index: int = Field(ge=0)

    @field_validator("answer")
    @classmethod
    def reject_blank_answer(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("answer must not be blank")
        return value


class AnswerResult(BaseModel):
    evaluation_status: Literal["final", "pending_ai"] = "final"
    correct: bool | None
    correct_answer: str | None
    explanation: str | None
    is_last: bool
    next_index: int | None = None


class SessionResultDetail(BaseModel):
    index: int
    question: str
    user_answer: str
    correct_answer: str | None
    correct: bool | None
    explanation: str | None
    evaluation_status: Literal["final", "pending_ai"]


class SessionResult(BaseModel):
    session_id: str
    document_id: str
    total: int
    correct: int
    incorrect: int
    pending: int = 0
    score: float | None  # 0.0 – 1.0; None while semantic grading is pending
    details: list[SessionResultDetail]


class QuizSession(BaseModel):
    session_id: str
    document_id: str
    user_id: str
    questions: list[Question]
    user_answers: list[str]
    status: str  # "active" / "completed"
    question_grades: dict[int, QuestionGrade] = Field(default_factory=dict)
    grading_report: GradingReport | None = None
    learning_report: LearningReport | None = None
    profile_written: bool = (
        False  # 画像写回幂等标记（submit_answer 与 /grade 两条路径去重）
    )
    extras_written: bool = False
    review_schedule_written: bool = False
