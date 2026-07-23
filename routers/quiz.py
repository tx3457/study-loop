from fastapi import APIRouter
from models.quiz import QuizRequest
from services.rag import generate_question

router = APIRouter()


@router.post("/generate/quiz/native")
async def generate_quiz(request: QuizRequest):
    """直连 RAG 出题端点，无 orchestrator/supervisor 封装。"""
    return await generate_question(
        request.document_id, request.description, request.count, request.difficulty, request.type
    )
