"""Deterministic Tutor quiz-session creation and recovery helpers."""

import hashlib
import json
from collections.abc import Mapping, Sequence

from models.quiz import Question
from models.session import QuizSession
from services.session import sessions, validate_provider_questions

MAX_TUTOR_ANSWER_LENGTH = 4000


def tutor_questions(state: Mapping) -> list[Question]:
    quiz = state.get("quiz") or {}
    raw_questions = quiz.get("questions", []) if isinstance(quiz, dict) else []
    questions = [
        Question.model_validate(question)
        for question in raw_questions
        if isinstance(question, dict)
    ]
    if not questions or len(questions) != len(raw_questions):
        raise ValueError("Tutor quiz does not contain a valid question set")
    requested_type = state.get("type", "choice")
    if requested_type not in {"choice", "true_false", "short_answer"}:
        raise ValueError(f"Unsupported Tutor question type: {requested_type}")
    validate_provider_questions(questions, requested_type)
    return questions


def tutor_session_id(state: Mapping) -> str:
    identity = json.dumps(
        {
            "thread_id": state.get("thread_id", ""),
            "user_id": state.get("user_id", ""),
            "document_id": state.get("document_id", ""),
            "turn": state.get("turn", 0),
            "quiz": state.get("quiz") or {},
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"tutor_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"


def validate_tutor_answers(
    raw_answers: object,
    questions: Sequence[Question],
) -> list[str]:
    if not isinstance(raw_answers, (list, tuple)):
        raise ValueError("Tutor answers must be a list")
    if len(raw_answers) != len(questions):
        raise ValueError(
            f"Tutor answer count mismatch: expected {len(questions)}, "
            f"got {len(raw_answers)}"
        )

    answers: list[str] = []
    for raw_answer in raw_answers:
        if not isinstance(raw_answer, str):
            raise ValueError("Each Tutor answer must be text")
        answer = raw_answer.strip()
        if not answer:
            raise ValueError("Tutor answers must not be blank")
        if len(answer) > MAX_TUTOR_ANSWER_LENGTH:
            raise ValueError(
                f"Tutor answers must be at most {MAX_TUTOR_ANSWER_LENGTH} characters"
            )
        answers.append(answer)
    return answers


def tutor_quiz_view(questions: Sequence[Question]) -> dict:
    """Return the pre-answer public projection without answers or explanations."""

    return {
        "questions": [
            {
                "index": index,
                "question": question.question,
                "options": question.options,
                "type": question.type,
            }
            for index, question in enumerate(questions)
        ]
    }


def ensure_tutor_session(
    state: Mapping,
    *,
    completed_answers: object | None = None,
) -> QuizSession:
    """Create or validate this round's immutable session.

    When ``completed_answers`` is supplied, answers are validated before any
    mutation. Replaying the same answers is idempotent; different answers are
    rejected.
    """

    questions = tutor_questions(state)
    session_id = tutor_session_id(state)
    answers = (
        validate_tutor_answers(completed_answers, questions)
        if completed_answers is not None
        else None
    )

    existing = sessions.get(session_id)
    if existing is None:
        existing = QuizSession(
            session_id=session_id,
            document_id=str(state.get("document_id", "")),
            user_id=str(state.get("user_id", "")),
            questions=questions,
            user_answers=answers or [],
            status="completed" if answers is not None else "active",
        )
        sessions[session_id] = existing
        return existing

    if (
        existing.document_id != state.get("document_id", "")
        or existing.user_id != state.get("user_id", "")
        or existing.questions != questions
    ):
        raise RuntimeError("Tutor quiz session identity collision")

    if answers is not None:
        if existing.status == "active":
            existing.user_answers = answers
            existing.status = "completed"
        elif existing.status != "completed" or existing.user_answers != answers:
            raise RuntimeError("Tutor quiz session was already completed differently")
    return existing


def restore_completed_tutor_session(state: Mapping) -> QuizSession:
    """Rebuild an in-memory session after a checkpointed post-answer crash."""

    expected_session_id = tutor_session_id(state)
    if state.get("session_id") != expected_session_id:
        raise RuntimeError("Tutor checkpoint session identity mismatch")
    if "answers" not in state:
        raise ValueError("Tutor checkpoint does not contain submitted answers")
    return ensure_tutor_session(state, completed_answers=state.get("answers"))
