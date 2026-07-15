from fastapi import APIRouter
from models.learning_path import LearningPath
from services.learning_path import generate_learning_path

router = APIRouter()


@router.post("/learning-path/{document_id}", response_model=LearningPath)
async def create_learning_path(document_id: str):
    return await generate_learning_path(document_id)
