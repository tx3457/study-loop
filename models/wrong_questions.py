from pydantic import BaseModel


class WrongEntry(BaseModel):
    entry_id: str          # uuid
    document_id: str
    question: str
    options: list[str] | None = None
    correct_answer: str
    explanation: str
    user_answer: str       # 当时的错误答案
    knowledge_gap: str | None = None
    session_id: str        # 来源会话


class WrongQuestionBank(BaseModel):
    document_id: str
    total: int
    entries: list[WrongEntry]
