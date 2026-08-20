"""Versioned models for durable Adaptive learning sessions.

The public response models intentionally contain only :class:`QuestionView`
objects.  The full ``QuizSession`` (including answers and explanations) is a
private field of :class:`AdaptiveSessionAggregate` and must never be returned
directly by an HTTP endpoint.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from models.adaptive import AdaptiveTurn, NextStepDecision
from models.grader import GradingReport
from models.session import QuestionView, QuizSession


MAX_ADAPTIVE_ANSWERS = 10
MAX_ANSWER_LENGTH = 4_000
MAX_SUBMIT_RECEIPTS = 16
MAX_TRAJECTORY_TURNS = 100
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_ACTIONS = {
    "advance",
    "remediate",
    "teach",
    "continue",
    "switch_to_plan",
    "finish",
}
_DIFFICULTIES = {"easy", "medium", "hard"}
_QUESTION_TYPES = {"choice", "true_false", "short_answer"}
_TERMINATE_REASONS = {
    "agent_finish",
    "mastery_reached",
    "max_turns",
    "switch_to_plan",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        revalidate_instances="always",
    )


def _strip_required(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
    return value


def _validate_decision(decision: NextStepDecision) -> None:
    if decision.action not in _ACTIONS:
        raise ValueError("adaptive decision has an invalid action")
    if decision.difficulty not in _DIFFICULTIES:
        raise ValueError("adaptive decision has an invalid difficulty")
    if decision.question_type not in _QUESTION_TYPES:
        raise ValueError("adaptive decision has an invalid question type")
    if not 0.0 <= decision.difficulty_score <= 1.0 or not math.isfinite(decision.difficulty_score):
        raise ValueError("adaptive decision difficulty score is out of range")
    if not 1 <= decision.count <= MAX_ADAPTIVE_ANSWERS:
        raise ValueError("adaptive decision question count is out of range")
    if not decision.topic.strip() or len(decision.topic) > 4_000:
        raise ValueError("adaptive decision topic must not be blank")
    if len(decision.reason) > 8_000:
        raise ValueError("adaptive decision reason is too long")
    if len(decision.target_weak_points) > 100 or any(
        not point.strip() or len(point) > 2_000 for point in decision.target_weak_points
    ):
        raise ValueError("adaptive decision weak points are invalid")


class AdaptiveStartRequest(_StrictModel):
    user_id: str = Field(default="default_user", min_length=1, max_length=128)
    document_id: str = Field(min_length=1, max_length=512)
    goal: str = Field(min_length=1, max_length=4_000)

    @field_validator("user_id", "document_id", "goal", mode="before")
    @classmethod
    def strip_text(cls, value: Any) -> Any:
        return _strip_required(value)


class AdaptiveSubmitRequest(_StrictModel):
    adaptive_session_id: str = Field(min_length=1, max_length=128)
    turn: int = Field(ge=1, le=MAX_TRAJECTORY_TURNS)
    revision: int = Field(ge=1)
    answers: list[str] = Field(
        default_factory=list,
        max_length=MAX_ADAPTIVE_ANSWERS,
    )

    @field_validator("adaptive_session_id", mode="before")
    @classmethod
    def strip_session_id(cls, value: Any) -> Any:
        return _strip_required(value)

    @field_validator("answers")
    @classmethod
    def normalize_answers(cls, answers: list[str]) -> list[str]:
        normalized: list[str] = []
        for answer in answers:
            answer = answer.strip()
            if not answer:
                raise ValueError("answers must not contain blank values")
            if len(answer) > MAX_ANSWER_LENGTH:
                raise ValueError("answer is too long")
            normalized.append(answer)
        return normalized


class AdaptiveQuestionFeedback(_StrictModel):
    index: int = Field(ge=0, le=MAX_ADAPTIVE_ANSWERS - 1)
    question: str = Field(min_length=1, max_length=4_000)
    your_answer: str = Field(min_length=1, max_length=MAX_ANSWER_LENGTH)
    correct_answer: str = Field(min_length=1, max_length=MAX_ANSWER_LENGTH)
    is_correct: bool
    ai_feedback: str | None = Field(default=None, max_length=8_000)
    knowledge_gap: str | None = Field(default=None, max_length=2_000)


class AdaptiveTurnArtifact(_StrictModel):
    """Canonical business result for one externally visible Adaptive state."""

    adaptive_session_id: str = Field(min_length=1, max_length=128)
    turn: int = Field(ge=1, le=MAX_TRAJECTORY_TURNS)
    done: bool = False
    turn_type: Literal["quiz", "teach"] = "quiz"
    questions: list[QuestionView] = Field(
        default_factory=list,
        max_length=MAX_ADAPTIVE_ANSWERS,
    )
    lesson: str | None = Field(default=None, max_length=40_000)
    decision: NextStepDecision | None = None
    last_report_score: float | None = Field(default=None, ge=0.0, le=1.0)
    last_report_gaps: list[str] = Field(default_factory=list, max_length=100)
    last_report_feedback: list[AdaptiveQuestionFeedback] = Field(
        default_factory=list,
        max_length=MAX_ADAPTIVE_ANSWERS,
    )
    mastery: float | None = Field(default=None, ge=0.0, le=1.0)
    trajectory: list[AdaptiveTurn] = Field(
        default_factory=list,
        min_length=1,
        max_length=MAX_TRAJECTORY_TURNS,
    )
    summary: str = Field(default="", max_length=40_000)
    terminate_reason: str = Field(default="", max_length=128)
    learning_path: dict[str, Any] | None = None

    @field_validator("adaptive_session_id", mode="before")
    @classmethod
    def strip_session_id(cls, value: Any) -> Any:
        return _strip_required(value)

    @field_validator("lesson", mode="before")
    @classmethod
    def normalize_optional_lesson(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        return value

    @field_validator("last_report_gaps")
    @classmethod
    def normalize_gaps(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            value = value.strip()
            if not value:
                raise ValueError("knowledge gaps must not contain blank values")
            if len(value) > 2_000:
                raise ValueError("knowledge gap is too long")
            normalized.append(value)
        return normalized

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        if self.decision is None:
            raise ValueError("adaptive artifact requires a decision")
        _validate_decision(self.decision)

        trajectory_turns = [item.turn for item in self.trajectory]
        if trajectory_turns != list(range(1, self.turn + 1)):
            raise ValueError("adaptive trajectory must cover every turn exactly once")
        for item in self.trajectory:
            if item.action not in _ACTIONS:
                raise ValueError("adaptive trajectory contains an invalid action")
            if not math.isfinite(item.difficulty_score) or not 0.0 <= item.difficulty_score <= 1.0:
                raise ValueError("adaptive trajectory difficulty is out of range")
            if item.score is not None and (
                not math.isfinite(item.score) or not 0.0 <= item.score <= 1.0
            ):
                raise ValueError("adaptive trajectory score is out of range")
            if item.mastery_after is not None and (
                not math.isfinite(item.mastery_after) or not 0.0 <= item.mastery_after <= 1.0
            ):
                raise ValueError("adaptive trajectory mastery is out of range")
            if not item.topic.strip() or len(item.topic) > 4_000:
                raise ValueError("adaptive trajectory topic is invalid")
            if len(item.reason) > 8_000:
                raise ValueError("adaptive trajectory reason is too long")
            if len(item.knowledge_gaps) > 100 or any(
                not gap.strip() or len(gap) > 2_000 for gap in item.knowledge_gaps
            ):
                raise ValueError("adaptive trajectory gaps are invalid")

        question_indices = [question.index for question in self.questions]
        if question_indices != list(range(len(self.questions))):
            raise ValueError("public question indices must be contiguous")
        feedback_indices = [feedback.index for feedback in self.last_report_feedback]
        if len(feedback_indices) != len(set(feedback_indices)):
            raise ValueError("feedback indices must be unique")

        if self.done:
            if self.questions or self.lesson is not None:
                raise ValueError("completed artifact cannot expose pending content")
            if not self.summary.strip() or not self.terminate_reason.strip():
                raise ValueError("completed artifact requires terminal metadata")
            if self.terminate_reason not in _TERMINATE_REASONS:
                raise ValueError("completed artifact has an invalid terminate reason")
            if (self.decision.action == "finish") != (self.terminate_reason == "agent_finish"):
                raise ValueError("finish decision and terminate reason disagree")
            if (self.decision.action == "switch_to_plan") != (
                self.terminate_reason == "switch_to_plan"
            ):
                raise ValueError("switch_to_plan decision and terminate reason disagree")
            if self.terminate_reason == "switch_to_plan" and self.learning_path is None:
                raise ValueError("switch_to_plan completion requires a learning path")
            if self.learning_path is not None and self.terminate_reason != "switch_to_plan":
                raise ValueError("learning path is only valid for switch_to_plan")
            return self

        if self.summary or self.terminate_reason or self.learning_path is not None:
            raise ValueError("active artifact cannot contain terminal metadata")
        if self.decision.action in {"finish", "switch_to_plan"}:
            raise ValueError("terminal adaptive decision cannot remain active")
        if self.turn_type == "quiz":
            if not self.questions or self.lesson is not None:
                raise ValueError("quiz artifact requires questions and no lesson")
            if self.decision.action == "teach":
                raise ValueError("teach decision cannot produce a quiz artifact")
        else:
            if self.questions or self.lesson is None:
                raise ValueError("teach artifact requires a lesson and no questions")
            if self.decision.action != "teach":
                raise ValueError("teach artifact requires a teach decision")
        return self


class AdaptiveTurnResponse(AdaptiveTurnArtifact):
    """Browser-safe Adaptive response and reload snapshot."""

    schema_version: Literal[1] = 1
    revision: int = Field(ge=1)
    expires_at: float = Field(gt=0)
    busy: bool = False
    learning_path_id: str | None = Field(
        default=None,
        pattern=r"^lp_[0-9a-f]{32}$",
    )

    @model_validator(mode="after")
    def validate_published_learning_path(self) -> Self:
        requires_path = self.terminate_reason == "switch_to_plan"
        if requires_path != (self.learning_path_id is not None):
            raise ValueError("switch_to_plan response requires exactly one published path id")
        return self


class AdaptivePendingSubmit(_StrictModel):
    """Durable operation input and checkpoints for an unfinished submit."""

    key_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    request_hash: str = Field(pattern=_SHA256_PATTERN)
    turn: int = Field(ge=1, le=MAX_TRAJECTORY_TURNS)
    revision: int = Field(ge=1)
    answers: list[str] = Field(
        default_factory=list,
        max_length=MAX_ADAPTIVE_ANSWERS,
    )
    next_decision: NextStepDecision | None = None
    mastery: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("answers")
    @classmethod
    def normalize_answers(cls, answers: list[str]) -> list[str]:
        return AdaptiveSubmitRequest.normalize_answers(answers)

    @model_validator(mode="after")
    def validate_pending_decision(self) -> Self:
        if self.next_decision is not None:
            _validate_decision(self.next_decision)
            if self.mastery is None:
                raise ValueError("checkpointed adaptive decision requires mastery")
        return self


class AdaptiveSubmitReceipt(_StrictModel):
    """Completed submit binding stored alongside the canonical aggregate."""

    key_hash: str = Field(pattern=_SHA256_PATTERN)
    request_hash: str = Field(pattern=_SHA256_PATTERN)
    turn: int = Field(ge=1, le=MAX_TRAJECTORY_TURNS)
    revision: int = Field(ge=1)
    response: AdaptiveTurnArtifact

    @model_validator(mode="after")
    def validate_response_turn(self) -> Self:
        if self.response.turn not in {self.turn, self.turn + 1}:
            raise ValueError("submit receipt response turn is inconsistent")
        return self


class AdaptiveSessionAggregate(_StrictModel):
    """Private, versioned source of truth for one Adaptive session."""

    schema_version: Literal[1] = 1
    adaptive_session_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    document_id: str = Field(min_length=1, max_length=512)
    goal: str = Field(min_length=1, max_length=4_000)
    status: Literal["active", "completed"] = "active"
    current_quiz: QuizSession | None = None
    last_report: GradingReport | None = None
    current_artifact: AdaptiveTurnArtifact
    pending: AdaptivePendingSubmit | None = None
    submit_receipts: dict[str, AdaptiveSubmitReceipt] = Field(default_factory=dict)

    @field_validator(
        "current_quiz",
        "last_report",
        "current_artifact",
        "pending",
        mode="before",
    )
    @classmethod
    def copy_nested_models(cls, value: Any) -> Any:
        # Pydantic otherwise keeps already-validated model instances by
        # reference.  A durable aggregate must not change when a route later
        # mutates its working QuizSession or artifact.
        if isinstance(value, BaseModel):
            return value.model_copy(deep=True)
        return value

    @field_validator("submit_receipts", mode="before")
    @classmethod
    def copy_submit_receipts(cls, value: Any) -> Any:
        # Dict containers are rebuilt by Pydantic, but already-validated model
        # values would otherwise remain shared with the caller.
        return copy.deepcopy(value)

    @field_validator(
        "adaptive_session_id",
        "user_id",
        "document_id",
        "goal",
        mode="before",
    )
    @classmethod
    def strip_identity(cls, value: Any) -> Any:
        return _strip_required(value)

    @staticmethod
    def _validate_quiz_integrity(quiz: QuizSession) -> None:
        total = len(quiz.questions)
        answered = len(quiz.user_answers)
        if not 1 <= total <= MAX_ADAPTIVE_ANSWERS:
            raise ValueError("adaptive quiz question count is out of range")
        if quiz.status == "active":
            if answered != 0 or quiz.question_grades or quiz.grading_report is not None:
                raise ValueError("active adaptive quiz cannot contain grading state")
        elif quiz.status == "completed":
            if answered != total:
                raise ValueError("completed adaptive quiz requires every answer")
        else:
            raise ValueError("adaptive quiz has an invalid status")

        for question in quiz.questions:
            if not question.question.strip() or len(question.question) > 4_000:
                raise ValueError("adaptive quiz question text is invalid")
            if not question.answer.strip() or len(question.answer) > 4_000:
                raise ValueError("adaptive quiz answer text is invalid")
            if len(question.explanation) > 8_000 or len(question.source) > 8_000:
                raise ValueError("adaptive quiz private metadata is too long")
            if question.options is not None and (
                len(question.options) > 20
                or any(not option.strip() or len(option) > 4_000 for option in question.options)
            ):
                raise ValueError("adaptive quiz options are invalid")

        for index, grade in quiz.question_grades.items():
            if index < 0 or index >= total or index >= answered:
                raise ValueError("adaptive quiz grade index is out of range")
            question = quiz.questions[index]
            if (
                grade.index != index
                or grade.question != question.question
                or grade.user_answer != quiz.user_answers[index]
                or grade.correct_answer != question.answer
            ):
                raise ValueError("adaptive quiz cached grade is inconsistent")

        report = quiz.grading_report
        if report is not None:
            expected_indices = list(range(total))
            if (
                quiz.status != "completed"
                or report.session_id != quiz.session_id
                or report.total != total
                or len(report.grades) != total
                or [grade.index for grade in report.grades] != expected_indices
                or sorted(quiz.question_grades) != expected_indices
            ):
                raise ValueError("adaptive quiz grading report is inconsistent")
            if any(
                report.grades[index].model_dump() != quiz.question_grades[index].model_dump()
                for index in expected_indices
            ):
                raise ValueError("adaptive quiz report differs from cached grades")
            correct = sum(grade.is_correct for grade in report.grades)
            expected_score = round(correct / total, 2)
            if report.correct != correct or report.score != expected_score:
                raise ValueError("adaptive quiz report summary is inconsistent")
        if quiz.profile_written and report is None:
            raise ValueError("adaptive quiz cannot write profile before grading")

    @staticmethod
    def _validate_report_integrity(report: GradingReport) -> None:
        if (
            not report.session_id
            or report.total <= 0
            or len(report.grades) != report.total
            or [grade.index for grade in report.grades] != list(range(report.total))
        ):
            raise ValueError("adaptive last report is inconsistent")
        correct = sum(grade.is_correct for grade in report.grades)
        expected_score = round(correct / report.total, 2)
        if report.correct != correct or report.score != expected_score:
            raise ValueError("adaptive last report summary is inconsistent")

    @model_validator(mode="after")
    def validate_state_machine(self) -> Self:
        artifact = self.current_artifact
        if artifact.adaptive_session_id != self.adaptive_session_id:
            raise ValueError("adaptive artifact belongs to another session")
        if (self.status == "completed") != artifact.done:
            raise ValueError("adaptive status and artifact completion disagree")

        quiz = self.current_quiz
        if self.last_report is not None:
            self._validate_report_integrity(self.last_report)
        if quiz is not None:
            expected_quiz_id = f"adaptive:{self.adaptive_session_id}:turn:{artifact.turn}"
            if quiz.session_id != expected_quiz_id:
                raise ValueError("adaptive quiz id does not match session turn")
            if quiz.user_id != self.user_id or quiz.document_id != self.document_id:
                raise ValueError("adaptive quiz ownership does not match session")
            self._validate_quiz_integrity(quiz)
            if quiz.status == "completed" and quiz.grading_report is not None:
                report_matches = self.last_report is not None and (
                    self.last_report.model_dump() == quiz.grading_report.model_dump()
                )
                if quiz.profile_written or artifact.done:
                    if not report_matches:
                        raise ValueError("completed adaptive quiz report was not recorded")
                elif (
                    self.last_report is not None
                    and self.last_report.session_id == quiz.session_id
                    and not report_matches
                ):
                    raise ValueError("adaptive last report differs from current quiz report")

        pending = self.pending
        if pending is not None:
            if artifact.done:
                raise ValueError("completed adaptive session cannot have a pending submit")
            if pending.turn != artifact.turn:
                raise ValueError("pending submit turn does not match current artifact")
            if pending.key_hash is not None and pending.key_hash in self.submit_receipts:
                raise ValueError("pending submit key is already completed")

        # While a submit is pending, ``last_report`` may already contain the
        # newly graded turn while the public artifact intentionally still
        # represents the pre-submit screen. Every stable state must expose an
        # exact projection of its canonical report, however.
        if pending is None:
            report = self.last_report
            expected_score = report.score if report is not None else None
            expected_gaps = (
                [
                    grade.knowledge_gap
                    for grade in report.grades
                    if not grade.is_correct and grade.knowledge_gap
                ]
                if report is not None
                else []
            )
            expected_feedback = (
                [
                    AdaptiveQuestionFeedback(
                        index=grade.index,
                        question=grade.question,
                        your_answer=grade.user_answer,
                        correct_answer=grade.correct_answer,
                        is_correct=grade.is_correct,
                        ai_feedback=grade.ai_feedback,
                        knowledge_gap=grade.knowledge_gap,
                    )
                    for grade in report.grades
                ]
                if report is not None
                else []
            )
            if artifact.last_report_score != expected_score:
                raise ValueError("adaptive artifact score differs from canonical report")
            if artifact.last_report_gaps != expected_gaps:
                raise ValueError("adaptive artifact gaps differ from canonical report")
            if [item.model_dump(mode="json") for item in artifact.last_report_feedback] != [
                item.model_dump(mode="json") for item in expected_feedback
            ]:
                raise ValueError("adaptive artifact feedback differs from canonical report")

        if artifact.done:
            if pending is not None:
                raise ValueError("completed adaptive session cannot have pending work")
            expected_current_report_id = f"adaptive:{self.adaptive_session_id}:turn:{artifact.turn}"
            answered_current_turn = (
                self.last_report is not None
                and self.last_report.session_id == expected_current_report_id
            ) or artifact.trajectory[-1].score is not None
            if quiz is None and answered_current_turn:
                raise ValueError("completed answered turn requires its private adaptive quiz")
            if quiz is not None:
                if quiz.status == "active":
                    raise ValueError("completed adaptive session cannot have an active quiz")
                if quiz.grading_report is None:
                    raise ValueError("completed adaptive quiz requires a grading report")
                if not quiz.profile_written:
                    raise ValueError("completed adaptive quiz requires committed memory")
        elif artifact.turn_type == "quiz":
            if quiz is None:
                raise ValueError("quiz artifact requires a private quiz")
            expected_questions = [
                QuestionView(
                    index=index,
                    question=question.question,
                    options=question.options,
                    type=question.type,
                )
                for index, question in enumerate(quiz.questions)
            ]
            if [item.model_dump() for item in artifact.questions] != [
                item.model_dump() for item in expected_questions
            ]:
                raise ValueError("public questions do not match private quiz")
            if pending is None:
                if quiz.status != "active":
                    raise ValueError("ready quiz must remain active")
            else:
                if len(pending.answers) != len(quiz.questions):
                    raise ValueError("pending answers do not cover the current quiz")
                if quiz.status == "active" and quiz.user_answers:
                    raise ValueError("active pending quiz cannot contain bound answers")
                if quiz.status == "completed" and quiz.user_answers != pending.answers:
                    raise ValueError("completed quiz answers differ from pending submit")
                if (pending.mastery is not None or pending.next_decision is not None) and (
                    quiz.grading_report is None
                ):
                    raise ValueError("adaptive decision checkpoint requires grading")
                if (pending.mastery is not None or pending.next_decision is not None) and (
                    not quiz.profile_written
                ):
                    raise ValueError("adaptive decision checkpoint requires committed memory")
        else:
            if quiz is not None:
                raise ValueError("teach artifact cannot retain a current quiz")
            if pending is not None and pending.answers:
                raise ValueError("teach submit cannot contain answers")

        if len(self.submit_receipts) > MAX_SUBMIT_RECEIPTS:
            raise ValueError("adaptive session contains too many submit receipts")
        for key_hash, receipt in self.submit_receipts.items():
            if not _is_sha256(key_hash) or key_hash != receipt.key_hash:
                raise ValueError("adaptive submit receipt key is invalid")
            if receipt.response.adaptive_session_id != self.adaptive_session_id:
                raise ValueError("adaptive submit receipt belongs to another session")
            if receipt.turn > artifact.turn:
                raise ValueError("adaptive submit receipt refers to a future turn")
            if receipt.response.turn > artifact.turn:
                raise ValueError("adaptive submit receipt response refers to a future turn")
            if receipt.response.done and (
                self.status != "completed"
                or receipt.response.model_dump(mode="json") != artifact.model_dump(mode="json")
            ):
                raise ValueError("completed adaptive submit receipt is not the canonical state")
        return self

    @classmethod
    def validate_for_persistence(
        cls,
        value: AdaptiveSessionAggregate | dict[str, Any],
    ) -> AdaptiveSessionAggregate:
        """Cross a canonical JSON boundary and rerun every nested validator."""

        payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
        try:
            canonical = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("adaptive session payload must be valid JSON") from exc
        return cls.model_validate_json(canonical)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)
