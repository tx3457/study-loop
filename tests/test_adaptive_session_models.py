from __future__ import annotations

import pytest
from pydantic import ValidationError

from models.adaptive import AdaptiveTurn, NextStepDecision
from models.adaptive_session import (
    MAX_SUBMIT_RECEIPTS,
    AdaptivePendingSubmit,
    AdaptiveQuestionFeedback,
    AdaptiveSessionAggregate,
    AdaptiveStartRequest,
    AdaptiveSubmitReceipt,
    AdaptiveSubmitRequest,
    AdaptiveTurnArtifact,
    AdaptiveTurnResponse,
)
from models.grader import GradingReport, QuestionGrade
from models.quiz import Question
from models.session import QuestionView, QuizSession


KEY_HASH = "a" * 64
REQUEST_HASH = "b" * 64


def decision(action: str = "continue") -> NextStepDecision:
    return NextStepDecision(
        action=action,
        topic="二分查找",
        difficulty="medium",
        difficulty_score=0.5,
        question_type="choice",
        count=1,
        reason="继续巩固",
    )


def trajectory(turn: int = 1, action: str = "continue") -> list[AdaptiveTurn]:
    return [
        AdaptiveTurn(
            turn=index,
            action=action if index == turn else "continue",
            topic="二分查找",
            difficulty_score=0.5,
            reason="继续巩固",
        )
        for index in range(1, turn + 1)
    ]


def private_question(answer: str = "B") -> Question:
    return Question(
        question="二分查找每轮如何缩小区间？",
        options=["A. 一个元素", "B. 一半"],
        answer=answer,
        explanation="比较中点后排除一半区间",
        source="chunk-secret",
        type="choice",
    )


def public_questions() -> list[QuestionView]:
    question = private_question()
    return [
        QuestionView(
            index=0,
            question=question.question,
            options=question.options,
            type="choice",
        )
    ]


def quiz(
    *,
    status: str = "active",
    answer: str = "B",
    turn: int = 1,
) -> QuizSession:
    question = private_question()
    user_answers = [answer] if status == "completed" else []
    return QuizSession(
        session_id=f"adaptive:adapt_1:turn:{turn}",
        document_id="doc-1",
        user_id="default_user",
        questions=[question],
        user_answers=user_answers,
        status=status,
    )


def add_report(value: QuizSession, *, correct: bool = True) -> GradingReport:
    grade = QuestionGrade(
        index=0,
        question=value.questions[0].question,
        user_answer=value.user_answers[0],
        correct_answer=value.questions[0].answer,
        is_correct=correct,
    )
    value.question_grades = {0: grade}
    report = GradingReport(
        session_id=value.session_id,
        total=1,
        correct=int(correct),
        score=float(correct),
        grades=[grade],
    )
    value.grading_report = report
    return report


def quiz_artifact(*, turn: int = 1) -> AdaptiveTurnArtifact:
    return AdaptiveTurnArtifact(
        adaptive_session_id="adapt_1",
        turn=turn,
        turn_type="quiz",
        questions=public_questions(),
        decision=decision(),
        trajectory=trajectory(turn),
    )


def teach_artifact() -> AdaptiveTurnArtifact:
    return AdaptiveTurnArtifact(
        adaptive_session_id="adapt_1",
        turn=1,
        turn_type="teach",
        lesson="先比较区间中点，再保留可能包含目标的一半。",
        decision=decision("teach"),
        trajectory=trajectory(action="teach"),
    )


def done_artifact(report: GradingReport | None = None) -> AdaptiveTurnArtifact:
    completed_trajectory = trajectory()
    if report is not None:
        completed_trajectory[-1].score = report.score
    return AdaptiveTurnArtifact(
        adaptive_session_id="adapt_1",
        turn=1,
        done=True,
        decision=decision("finish"),
        last_report_score=report.score if report is not None else None,
        last_report_gaps=(
            [
                grade.knowledge_gap
                for grade in report.grades
                if not grade.is_correct and grade.knowledge_gap
            ]
            if report is not None
            else []
        ),
        last_report_feedback=(
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
        ),
        trajectory=completed_trajectory,
        summary="本轮学习已完成。",
        terminate_reason="agent_finish",
    )


def aggregate(**updates) -> AdaptiveSessionAggregate:
    payload = {
        "adaptive_session_id": "adapt_1",
        "user_id": "default_user",
        "document_id": "doc-1",
        "goal": "掌握二分查找",
        "status": "active",
        "current_quiz": quiz(),
        "current_artifact": quiz_artifact(),
    }
    payload.update(updates)
    return AdaptiveSessionAggregate(**payload)


def aggregate_after_receipt(
    **updates,
) -> AdaptiveSessionAggregate:
    payload = {
        "current_quiz": quiz(turn=2),
        "current_artifact": quiz_artifact(turn=2),
    }
    payload.update(updates)
    return aggregate(**payload)


def response_artifact(*, turn: int = 2) -> AdaptiveTurnArtifact:
    return quiz_artifact(turn=turn)


def receipt(*, key_hash: str = KEY_HASH) -> AdaptiveSubmitReceipt:
    return AdaptiveSubmitReceipt(
        key_hash=key_hash,
        request_hash=REQUEST_HASH,
        turn=1,
        revision=1,
        response=response_artifact(),
    )


def test_start_request_strips_text_and_uses_default_user() -> None:
    value = AdaptiveStartRequest(document_id="  doc-1  ", goal="  学习目标  ")

    assert value.user_id == "default_user"
    assert value.document_id == "doc-1"
    assert value.goal == "学习目标"


@pytest.mark.parametrize(
    ("field", "value"),
    [("user_id", "   "), ("document_id", "\t"), ("goal", "\n")],
)
def test_start_request_rejects_blank_identity(field: str, value: str) -> None:
    payload = {"user_id": "user", "document_id": "doc", "goal": "goal"}
    payload[field] = value

    with pytest.raises(ValidationError):
        AdaptiveStartRequest(**payload)


def test_start_request_enforces_lengths_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AdaptiveStartRequest(document_id="d" * 513, goal="goal")
    with pytest.raises(ValidationError):
        AdaptiveStartRequest(document_id="doc", goal="goal", unexpected=True)


def test_submit_request_normalizes_answers_and_allows_teach_continue() -> None:
    value = AdaptiveSubmitRequest(
        adaptive_session_id="  adapt_1 ",
        turn=1,
        revision=3,
        answers=["  B  "],
    )
    teach = AdaptiveSubmitRequest(
        adaptive_session_id="adapt_1",
        turn=1,
        revision=3,
        answers=[],
    )

    assert value.adaptive_session_id == "adapt_1"
    assert value.answers == ["B"]
    assert teach.answers == []


@pytest.mark.parametrize(
    "answers",
    [["   "], ["x" * 4_001], ["x"] * 11],
)
def test_submit_request_rejects_invalid_answers(answers: list[str]) -> None:
    with pytest.raises(ValidationError):
        AdaptiveSubmitRequest(
            adaptive_session_id="adapt_1",
            turn=1,
            revision=1,
            answers=answers,
        )


def test_artifact_enforces_quiz_teach_and_terminal_shapes() -> None:
    assert quiz_artifact().questions
    assert teach_artifact().lesson
    assert done_artifact().done

    with pytest.raises(ValidationError):
        AdaptiveTurnArtifact(
            adaptive_session_id="adapt_1",
            turn=1,
            turn_type="quiz",
            lesson="不应存在",
            decision=decision(),
            trajectory=trajectory(),
        )
    with pytest.raises(ValidationError):
        AdaptiveTurnArtifact(
            adaptive_session_id="adapt_1",
            turn=1,
            turn_type="teach",
            decision=decision("teach"),
            trajectory=trajectory(action="teach"),
        )
    with pytest.raises(ValidationError):
        AdaptiveTurnArtifact(
            adaptive_session_id="adapt_1",
            turn=1,
            done=True,
            decision=decision("finish"),
            trajectory=trajectory(),
        )


def test_artifact_rejects_turn_gaps_and_active_terminal_decision() -> None:
    with pytest.raises(ValidationError):
        AdaptiveTurnArtifact(
            adaptive_session_id="adapt_1",
            turn=2,
            questions=public_questions(),
            decision=decision(),
            trajectory=trajectory(1),
        )
    with pytest.raises(ValidationError):
        AdaptiveTurnArtifact(
            adaptive_session_id="adapt_1",
            turn=1,
            questions=public_questions(),
            decision=decision("switch_to_plan"),
            trajectory=trajectory(),
        )


def test_public_response_contains_metadata_but_not_private_question_fields() -> None:
    public = AdaptiveTurnResponse(
        **quiz_artifact().model_dump(),
        revision=1,
        expires_at=2_000_000_000.0,
        busy=False,
    )

    dumped = public.model_dump_json()
    assert public.schema_version == 1
    assert '"revision":1' in dumped
    assert "chunk-secret" not in dumped
    assert '"answer"' not in dumped
    assert '"explanation"' not in dumped


def test_valid_quiz_aggregate_round_trips_with_private_quiz() -> None:
    value = aggregate()

    restored = AdaptiveSessionAggregate.model_validate_json(value.model_dump_json())

    assert restored == value
    assert restored.current_quiz is not None
    assert restored.current_quiz.questions[0].answer == "B"


@pytest.mark.parametrize(
    "updates",
    [
        {"adaptive_session_id": "other"},
        {"user_id": "other"},
        {"document_id": "other"},
    ],
)
def test_aggregate_rejects_artifact_or_quiz_ownership_mismatch(updates: dict) -> None:
    payload = {
        "adaptive_session_id": "adapt_1",
        "user_id": "default_user",
        "document_id": "doc-1",
        "goal": "goal",
        "status": "active",
        "current_quiz": quiz(),
        "current_artifact": quiz_artifact(),
    }
    payload.update(updates)

    with pytest.raises(ValidationError):
        AdaptiveSessionAggregate(**payload)


def test_aggregate_rejects_public_question_projection_mismatch() -> None:
    artifact = quiz_artifact()
    artifact.questions[0].question = "被篡改的题面"

    with pytest.raises(ValidationError, match="public questions"):
        aggregate(current_artifact=artifact)


def test_aggregate_requires_deterministic_quiz_id() -> None:
    current = quiz()
    current.session_id = "another-adaptive-session"

    with pytest.raises(ValidationError, match="quiz id"):
        aggregate(current_quiz=current)


def test_ready_quiz_must_be_active_and_teach_has_no_quiz() -> None:
    completed = quiz(status="completed")
    add_report(completed)

    with pytest.raises(ValidationError, match="ready quiz"):
        aggregate(current_quiz=completed)
    with pytest.raises(ValidationError, match="teach artifact"):
        aggregate(current_artifact=teach_artifact())

    taught = aggregate(current_quiz=None, current_artifact=teach_artifact())
    assert taught.current_artifact.turn_type == "teach"


def test_pending_submit_must_match_turn_and_quiz_answers() -> None:
    pending = AdaptivePendingSubmit(
        key_hash=KEY_HASH,
        request_hash=REQUEST_HASH,
        turn=1,
        revision=1,
        answers=["B"],
    )
    completed = quiz(status="completed")

    value = aggregate(current_quiz=completed, pending=pending)
    assert value.current_quiz is not None
    assert value.current_quiz.status == "completed"

    with pytest.raises(ValidationError, match="turn"):
        aggregate(pending=pending.model_copy(update={"turn": 2}))
    with pytest.raises(ValidationError, match="differ"):
        aggregate(
            current_quiz=quiz(status="completed", answer="A"),
            pending=pending,
        )


def test_pending_decision_requires_completed_grading_checkpoint() -> None:
    pending = AdaptivePendingSubmit(
        key_hash=KEY_HASH,
        request_hash=REQUEST_HASH,
        turn=1,
        revision=1,
        answers=["B"],
        mastery=1.0,
        next_decision=decision(),
    )

    with pytest.raises(ValidationError, match="requires grading"):
        aggregate(current_quiz=quiz(status="completed"), pending=pending)

    completed = quiz(status="completed")
    report = add_report(completed)
    completed.profile_written = True
    value = aggregate(current_quiz=completed, last_report=report, pending=pending)
    assert value.pending is not None
    assert value.pending.next_decision is not None


def test_pending_decision_requires_checkpointed_mastery() -> None:
    with pytest.raises(ValidationError, match="requires mastery"):
        AdaptivePendingSubmit(
            key_hash=KEY_HASH,
            request_hash=REQUEST_HASH,
            turn=1,
            revision=1,
            answers=["B"],
            next_decision=decision(),
        )


def test_pending_decision_requires_memory_commit_and_matching_last_report() -> None:
    completed = quiz(status="completed")
    report = add_report(completed)
    pending = AdaptivePendingSubmit(
        key_hash=KEY_HASH,
        request_hash=REQUEST_HASH,
        turn=1,
        revision=1,
        answers=["B"],
        mastery=1.0,
        next_decision=decision(),
    )

    with pytest.raises(ValidationError, match="committed memory"):
        aggregate(current_quiz=completed, last_report=report, pending=pending)

    completed.profile_written = True
    wrong_report = report.model_copy(update={"session_id": "previous-quiz"})
    with pytest.raises(ValidationError, match="was not recorded"):
        aggregate(current_quiz=completed, last_report=wrong_report, pending=pending)


def test_teach_pending_accepts_only_empty_answers() -> None:
    empty = AdaptivePendingSubmit(
        key_hash=None,
        request_hash=REQUEST_HASH,
        turn=1,
        revision=1,
        answers=[],
    )
    value = aggregate(current_quiz=None, current_artifact=teach_artifact(), pending=empty)
    assert value.pending == empty

    with pytest.raises(ValidationError, match="teach submit"):
        aggregate(
            current_quiz=None,
            current_artifact=teach_artifact(),
            pending=empty.model_copy(update={"answers": ["B"]}),
        )


def test_completed_state_has_no_pending_or_active_quiz() -> None:
    completed = aggregate(
        status="completed",
        current_quiz=None,
        current_artifact=done_artifact(),
    )
    assert completed.status == "completed"

    pending = AdaptivePendingSubmit(
        key_hash=KEY_HASH,
        request_hash=REQUEST_HASH,
        turn=1,
        revision=1,
        answers=["B"],
    )
    with pytest.raises(ValidationError, match="pending"):
        aggregate(
            status="completed",
            current_quiz=None,
            current_artifact=done_artifact(),
            pending=pending,
        )
    with pytest.raises(ValidationError, match="active quiz"):
        aggregate(status="completed", current_artifact=done_artifact())


def test_completed_quiz_keeps_canonical_report_for_recovery() -> None:
    completed_quiz = quiz(status="completed")
    report = add_report(completed_quiz)
    completed_quiz.profile_written = True

    value = aggregate(
        status="completed",
        current_quiz=completed_quiz,
        last_report=report,
        current_artifact=done_artifact(report),
    )

    assert value.last_report == report


def test_completed_quiz_requires_memory_and_terminal_reason_consistency() -> None:
    completed_quiz = quiz(status="completed")
    report = add_report(completed_quiz)

    with pytest.raises(ValidationError, match="committed memory"):
        aggregate(
            status="completed",
            current_quiz=completed_quiz,
            last_report=report,
            current_artifact=done_artifact(report),
        )

    with pytest.raises(ValidationError, match="terminate reason"):
        AdaptiveTurnArtifact(
            **{
                **done_artifact().model_dump(),
                "terminate_reason": "mastery_reached",
            }
        )


@pytest.mark.parametrize("value", ["A" * 64, "a" * 63, "g" * 64, " a" * 32])
def test_pending_and_receipt_reject_invalid_hashes(value: str) -> None:
    with pytest.raises(ValidationError):
        AdaptivePendingSubmit(
            key_hash=value,
            request_hash=REQUEST_HASH,
            turn=1,
            revision=1,
            answers=[],
        )
    with pytest.raises(ValidationError):
        receipt(key_hash=value)


def test_receipt_key_identity_and_response_ownership_are_strict() -> None:
    valid_receipt = receipt()
    value = aggregate_after_receipt(submit_receipts={KEY_HASH: valid_receipt})
    assert value.submit_receipts[KEY_HASH] == valid_receipt

    with pytest.raises(ValidationError, match="key"):
        aggregate(submit_receipts={"c" * 64: valid_receipt})

    foreign = receipt()
    foreign.response.adaptive_session_id = "other"
    with pytest.raises(ValidationError, match="another session"):
        aggregate(submit_receipts={KEY_HASH: foreign})


def test_receipt_contains_only_artifact_and_requires_consistent_turn() -> None:
    value = receipt()
    assert isinstance(value.response, AdaptiveTurnArtifact)
    assert not isinstance(value.response, AdaptiveTurnResponse)
    assert "expires_at" not in value.response.model_dump()
    assert "busy" not in value.response.model_dump()

    with pytest.raises(ValidationError, match="turn"):
        AdaptiveSubmitReceipt(
            key_hash=KEY_HASH,
            request_hash=REQUEST_HASH,
            turn=1,
            revision=1,
            response=response_artifact(turn=3),
        )


def test_receipt_response_cannot_advance_beyond_canonical_state() -> None:
    with pytest.raises(ValidationError, match="response refers to a future turn"):
        aggregate(submit_receipts={KEY_HASH: receipt()})


def test_completed_receipt_must_equal_completed_canonical_state() -> None:
    terminal = done_artifact()
    terminal_receipt = AdaptiveSubmitReceipt(
        key_hash=KEY_HASH,
        request_hash=REQUEST_HASH,
        turn=1,
        revision=1,
        response=terminal,
    )

    with pytest.raises(ValidationError, match="not the canonical state"):
        aggregate(submit_receipts={KEY_HASH: terminal_receipt})

    value = aggregate(
        status="completed",
        current_quiz=None,
        current_artifact=terminal,
        submit_receipts={KEY_HASH: terminal_receipt},
    )
    assert value.submit_receipts[KEY_HASH].response == terminal


def test_completed_answered_turn_cannot_drop_its_private_quiz() -> None:
    completed = quiz(status="completed")
    report = add_report(completed)
    completed.profile_written = True
    terminal = done_artifact(report)

    with pytest.raises(ValidationError, match="requires its private adaptive quiz"):
        aggregate(
            status="completed",
            current_quiz=None,
            current_artifact=terminal,
            last_report=report,
        )


def test_stable_artifact_report_projection_is_canonical() -> None:
    completed = quiz(status="completed")
    report = add_report(completed)
    completed.profile_written = True

    with pytest.raises(ValidationError, match="score differs"):
        aggregate(
            status="completed",
            current_quiz=completed,
            current_artifact=done_artifact(),
            last_report=report,
        )


def test_web_visible_nested_text_uses_the_same_backend_bounds() -> None:
    overlong_decision = decision()
    overlong_decision.reason = "x" * 8_001
    with pytest.raises(ValidationError, match="reason is too long"):
        AdaptiveTurnArtifact(
            adaptive_session_id="adapt_1",
            turn=1,
            questions=public_questions(),
            decision=overlong_decision,
            trajectory=trajectory(),
        )

    overlong_question = quiz()
    overlong_question.questions[0].question = "q" * 4_001
    with pytest.raises(ValidationError, match="question text"):
        aggregate(current_quiz=overlong_question)

    padded_question = quiz()
    padded_question.questions[0].question = " " * 100 + "q" * 4_000
    with pytest.raises(ValidationError, match="question text"):
        aggregate(current_quiz=padded_question)

    too_many_options = quiz()
    too_many_options.questions[0].options = [f"option-{index}" for index in range(21)]
    with pytest.raises(ValidationError, match="options"):
        aggregate(current_quiz=too_many_options)

    overlong_trajectory = quiz_artifact()
    overlong_trajectory.trajectory[0].knowledge_gaps = ["g" * 2_001]
    with pytest.raises(ValidationError, match="trajectory gaps"):
        aggregate(current_artifact=overlong_trajectory)


def test_receipt_count_is_bounded_and_pending_key_cannot_be_completed() -> None:
    receipts: dict[str, AdaptiveSubmitReceipt] = {}
    for index in range(MAX_SUBMIT_RECEIPTS + 1):
        key = f"{index:064x}"
        receipts[key] = receipt(key_hash=key)

    with pytest.raises(ValidationError, match="too many"):
        aggregate(submit_receipts=receipts)

    pending = AdaptivePendingSubmit(
        key_hash=KEY_HASH,
        request_hash=REQUEST_HASH,
        turn=1,
        revision=1,
        answers=["B"],
    )
    with pytest.raises(ValidationError, match="already completed"):
        aggregate(pending=pending, submit_receipts={KEY_HASH: receipt()})


def test_invalid_cached_grade_and_report_are_rejected() -> None:
    completed = quiz(status="completed")
    report = add_report(completed)
    completed.question_grades[0].correct_answer = "被篡改"
    pending = AdaptivePendingSubmit(
        key_hash=KEY_HASH,
        request_hash=REQUEST_HASH,
        turn=1,
        revision=1,
        answers=["B"],
    )

    with pytest.raises(ValidationError, match="cached grade"):
        aggregate(current_quiz=completed, last_report=report, pending=pending)


def test_model_copy_must_be_revalidated_before_persistence() -> None:
    value = aggregate()
    invalid = value.model_copy(update={"adaptive_session_id": "other"})

    with pytest.raises(ValidationError, match="another session"):
        AdaptiveSessionAggregate.validate_for_persistence(invalid)


def test_mutated_model_instances_are_fully_revalidated() -> None:
    value = aggregate(current_quiz=None, current_artifact=teach_artifact())
    value.current_artifact.lesson = None

    with pytest.raises(ValidationError, match="teach artifact"):
        AdaptiveSessionAggregate.model_validate(value)

    invalid_receipt = receipt()
    invalid_receipt.response = response_artifact(turn=3)
    with pytest.raises(ValidationError, match="response turn"):
        aggregate(submit_receipts={KEY_HASH: invalid_receipt})


def test_aggregate_does_not_alias_caller_payload() -> None:
    source_quiz = quiz(turn=2)
    source_receipt = receipt()
    value = aggregate_after_receipt(
        current_quiz=source_quiz,
        submit_receipts={KEY_HASH: source_receipt},
    )

    source_quiz.questions[0].answer = "A"
    source_receipt.response.adaptive_session_id = "other"

    assert value.current_quiz is not None
    assert value.current_quiz.questions[0].answer == "B"
    assert value.submit_receipts[KEY_HASH].response.adaptive_session_id == "adapt_1"
