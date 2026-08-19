from typing import Literal

from pydantic import BaseModel


class WrongEntry(BaseModel):
    entry_id: str
    document_id: str
    question: str
    options: list[str] | None = None
    question_type: Literal["choice", "true_false", "short_answer"] = "short_answer"
    correct_answer: str
    explanation: str
    user_answer: str       # 当时的错误答案
    knowledge_gap: str | None = None
    session_id: str        # 来源会话


class WrongQuestionBank(BaseModel):
    document_id: str
    total: int
    entries: list[WrongEntry]
