from fastapi import APIRouter, Depends
from models.quiz import QuizRequest
from services.auth import require_user_id
from services.rag import generate_question

router = APIRouter()


@router.post("/generate/quiz/native")
async def generate_quiz(
    request: QuizRequest, user_id: str = Depends(require_user_id)
):
    """直连 RAG 出题端点，无 orchestrator/supervisor 封装。"""
    return await generate_question(
        request.document_id,
        request.description,
        request.count,
        request.difficulty,
        request.type,
        owner_id=user_id,
    )
