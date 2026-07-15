from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.session import SessionStartRequest, AnswerRequest, AnswerResult, SessionResult, QuestionView
from models.grader import GradingReport
from models.report import LearningReport
from services.session import start_session, submit_answer, get_result
from services.grader import grade_session
from services.report import generate_report
from services.wrong_questions import collect_wrong_answers
from services.memory import write_episodic_memory, update_semantic_memory
from services.session import sessions

router = APIRouter(prefix="/session")


class StartSessionResponse(BaseModel):
    session_id: str
    total: int
    questions: list[QuestionView]


@router.post("/start", response_model=StartSessionResponse)
async def start(req: SessionStartRequest):
    session_id, questions = await start_session(req)
    return StartSessionResponse(session_id=session_id, total=len(questions), questions=questions)


@router.post("/{session_id}/answer", response_model=AnswerResult)
async def answer(session_id: str, req: AnswerRequest):
    try:
        return await submit_answer(session_id, req.answer)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{session_id}/result", response_model=SessionResult)
async def result(session_id: str):
    try:
        return await get_result(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{session_id}/grade", response_model=GradingReport)
async def grade(session_id: str):
    try:
        report = await grade_session(session_id)
        collect_wrong_answers(session_id, report)           # 自动收集错题

        # 画像写回：答完最后一题时 submit_answer 已写（必经路径），
        # 这里仅在当时写失败的情况下兜底重写，避免 EMA / weak_points 重复累积
        session = sessions.get(session_id)
        if session and not session.profile_written:
            await write_episodic_memory(session.user_id, report, session.document_id)
            await update_semantic_memory(session.user_id, report, session.document_id)
            session.profile_written = True

        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{session_id}/report", response_model=LearningReport)
async def report(session_id: str):
    try:
        return await generate_report(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
