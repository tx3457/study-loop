from fastapi import APIRouter
from models.quiz import QuizRequest
from services.rag import generate_question

router = APIRouter()


@router.post("/generate/quiz/native")
async def generate_quiz(request: QuizRequest):
    """直连 RAG 出题（无 orchestrator/supervisor 封装，前端在用）。

    note: 早期两段式 ReAct 端点 /generate/quiz/agent 及其 confirm（依赖 services/agent.py 的
    legacy graph）已于 supervisor MAS 重构中删除——出题主链路改走
    agents/quiz_agent.py（Hybrid RAG + 证据门 + critic）；单次出题见 /agent/tutor/oneshot。
    """
    return await generate_question(
        request.document_id, request.description, request.count, request.difficulty, request.type
    )
