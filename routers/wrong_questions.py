from fastapi import APIRouter, HTTPException
from models.wrong_questions import WrongQuestionBank
from models.session import QuestionView
from pydantic import BaseModel
from services.wrong_questions import get_wrong_questions, start_repractice

router = APIRouter(prefix="/wrong-questions")


class RepracticeResponse(BaseModel):
    session_id: str
    total: int
    questions: list[QuestionView]


@router.get("/{document_id}", response_model=WrongQuestionBank)
async def list_wrong_questions(document_id: str):
    return get_wrong_questions(document_id)


@router.post("/{document_id}/practice", response_model=RepracticeResponse)
async def repractice(document_id: str):
    try:
        session_id, questions = start_repractice(document_id)
        return RepracticeResponse(session_id=session_id, total=len(questions), questions=questions)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
