from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from models.grader import GradingReport, QuestionGrade
from models.quiz import Question
from models.report import LearningReport


class SessionStartRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=512)
    description: str = Field(max_length=4000)
    count: int = Field(default=5, ge=1, le=10)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    type: Literal["choice", "true_false", "short_answer"] = "choice"
    user_id: str = Field(default="default_user", min_length=1, max_length=128)

    @field_validator("document_id", "user_id")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        return value.strip()


class QuestionView(BaseModel):
    """题目视图：对用户隐藏答案和解析"""

    index: int
    question: str
    options: list[str] | None = None
    type: Literal["choice", "true_false", "short_answer"] = "choice"


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
    revision: int | None = Field(default=None, ge=1)
    expires_at: float | None = None


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
    revision: int | None = Field(default=None, ge=1)
    expires_at: float | None = None


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


class QuizSessionAggregate(BaseModel):
    """Versioned durable payload for a Web Quiz session.

    The private ``Question`` objects remain server-side.  Only
    :class:`SessionSnapshot` is safe to return to a browser.
    """

    schema_version: Literal[1] = 1
    origin: Literal["standard", "wrong_question"] = "standard"
    session: QuizSession
    last_answer_index: int | None = Field(default=None, ge=0)
    last_answer_result: AnswerResult | None = None
    answer_request_hashes: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_state_machine(self):
        session = self.session
        total = len(session.questions)
        answered = len(session.user_answers)

        if total == 0:
            raise ValueError("quiz session must contain at least one question")
        if answered > total:
            raise ValueError("quiz session contains too many answers")
        if session.status == "active":
            if answered >= total:
                raise ValueError("active quiz session must have an unanswered question")
        elif session.status == "completed":
            if answered != total:
                raise ValueError("completed quiz session must contain every answer")
        else:
            raise ValueError("invalid durable quiz session status")

        if answered == 0:
            if self.last_answer_index is not None or self.last_answer_result is not None:
                raise ValueError("unanswered quiz session cannot contain feedback")
        else:
            if self.last_answer_index != answered - 1 or self.last_answer_result is None:
                raise ValueError("last answer feedback does not match quiz progress")
            expected_last = session.status == "completed"
            if self.last_answer_result.is_last != expected_last:
                raise ValueError("last answer completion flag does not match quiz status")
            expected_next = None if expected_last else answered
            if self.last_answer_result.next_index != expected_next:
                raise ValueError("last answer next index does not match quiz progress")

        if len(self.answer_request_hashes) > answered:
            raise ValueError("quiz session contains too many answer request bindings")
        for key_hash, request_hash in self.answer_request_hashes.items():
            if (
                len(key_hash) != 64
                or len(request_hash) != 64
                or any(char not in "0123456789abcdef" for char in key_hash)
                or any(char not in "0123456789abcdef" for char in request_hash)
            ):
                raise ValueError("invalid answer request binding")

        for index, grade in session.question_grades.items():
            if index < 0 or index >= total:
                raise ValueError("cached question grade index is out of range")
            if index >= answered:
                raise ValueError("cached question grade has no submitted answer")
            question = session.questions[index]
            if (
                grade.index != index
                or grade.question != question.question
                or grade.user_answer != session.user_answers[index]
                or grade.correct_answer != question.answer
            ):
                raise ValueError("cached question grade does not match quiz session")

        grading = session.grading_report
        if grading is not None:
            if session.status != "completed":
                raise ValueError("active quiz session cannot contain a grading report")
            if grading.session_id != session.session_id or grading.total != total:
                raise ValueError("grading report does not belong to quiz session")
            if len(grading.grades) != total:
                raise ValueError("grading report does not cover every question")
            expected_indices = list(range(total))
            if [grade.index for grade in grading.grades] != expected_indices:
                raise ValueError("grading report indices do not match quiz session")
            if sorted(session.question_grades) != expected_indices:
                raise ValueError("grading report is missing cached question grades")
            if any(
                grade.model_dump() != session.question_grades[index].model_dump()
                for index, grade in enumerate(grading.grades)
            ):
                raise ValueError("grading report differs from cached question grades")
            correct = sum(grade.is_correct for grade in grading.grades)
            expected_score = round(correct / total, 2)
            if grading.correct != correct or grading.score != expected_score:
                raise ValueError("grading report summary is inconsistent")

        report = session.learning_report
        if report is not None:
            if grading is None:
                raise ValueError("learning report requires a canonical grading report")
            if (
                report.session_id != session.session_id
                or report.document_id != session.document_id
                or report.overall_score != grading.score
            ):
                raise ValueError("learning report does not belong to quiz session")

        return self


class SessionSnapshot(BaseModel):
    """Browser-safe projection used to recover a Quiz page after navigation."""

    schema_version: Literal[1] = 1
    origin: Literal["standard", "wrong_question"] = "standard"
    session_id: str
    document_id: str
    revision: int = Field(ge=1)
    status: Literal["active", "completed"]
    total: int = Field(ge=1)
    answered_count: int = Field(ge=0)
    questions: list[QuestionView]
    last_answer_index: int | None = Field(default=None, ge=0)
    last_user_answer: str | None = None
    last_answer_result: AnswerResult | None = None
    result: SessionResult | None = None
    grading_report: GradingReport | None = None
    learning_report: LearningReport | None = None
    expires_at: float
    busy: bool = False
