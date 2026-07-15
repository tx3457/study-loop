from pydantic import BaseModel

class Quiz(BaseModel):
    document_id: str
    filename: str
    chunks: int
    status: str

class QuizRequest(BaseModel):
    document_id: str
    count: int
    description: str
    difficulty: str #easy/medium/hard
    type: str #choice/true_false/short_answer

class Question(BaseModel):
    question: str
    options: list[str] | None = None
    answer: str
    explanation: str
    source: str
    type: str = "choice"    # choice / true_false / short_answer


class QuizResponse(BaseModel):
    questions: list[Question]

class ReviewResult(BaseModel):
    passed: bool
    reason: str
