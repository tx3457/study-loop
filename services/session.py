import logging
import uuid
from models.grader import GradingReport, QuestionGrade
from models.session import QuizSession, SessionStartRequest, QuestionView, AnswerResult, SessionResult
from services.rag import generate_question
from services.memory import get_user_profile, update_semantic_memory, write_episodic_memory

logger = logging.getLogger(__name__)

# 内存会话存储（服务重启后清空，Phase 3 升级为持久化）
sessions: dict[str, QuizSession] = {}

# 文字难度 → 连续值（无用户画像时的兜底）
_DIFFICULTY_SCORE = {"easy": 0.2, "medium": 0.5, "hard": 0.8}


async def start_session(req: SessionStartRequest) -> tuple[str, list[QuestionView]]:
    # 自适应：读取用户画像，计算本次出题参数
    profile = await get_user_profile(req.user_id)
    if profile and req.document_id in profile.get("topic_mastery", {}):
        # 有该文档的历史成绩：难度 = 掌握度 + 0.15（略高于当前水平，触发学习区间）
        mastery = profile["topic_mastery"][req.document_id]
        difficulty_score = round(min(mastery + 0.15, 1.0), 2)
        weak_points = profile.get("weak_points", [])
    else:
        # 新用户或新文档：将文字难度转换为连续值，无薄弱知识点
        difficulty_score = _DIFFICULTY_SCORE.get(req.difficulty, 0.5)
        weak_points = []

    quiz_response = await generate_question(
        req.document_id, req.description, req.count, req.difficulty, req.type,
        difficulty_score=difficulty_score,
        weak_points=weak_points,
    )
    session_id = str(uuid.uuid4())
    session = QuizSession(
        session_id=session_id,
        document_id=req.document_id,
        user_id=req.user_id,
        questions=quiz_response.questions,
        user_answers=[],
        status="active",
    )
    sessions[session_id] = session

    questions_view = [
        QuestionView(index=i, question=q.question, options=q.options)
        for i, q in enumerate(session.questions)
    ]
    return session_id, questions_view


async def submit_answer(session_id: str, answer: str) -> AnswerResult:
    session = sessions.get(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")
    if session.status == "completed":
        raise ValueError("Session already completed")

    current_index = len(session.user_answers)
    if current_index >= len(session.questions):
        raise ValueError("All questions already answered")

    question = session.questions[current_index]
    session.user_answers.append(answer)

    correct = answer.strip().upper() == question.answer.strip().upper()

    is_last = len(session.user_answers) == len(session.questions)
    if is_last:
        session.status = "completed"
        # 画像写回：此前只有 orchestrator 轨道（adapt_writer）写画像，
        # /session/* 轨道答完即丢 → 学习报告恒空、自适应难度永远冷启动。
        # 这里复用同一套 bank API 写回，fail-soft 不影响答题结果返回。
        try:
            await _write_back_profile(session, session_id)
            session.profile_written = True
        except Exception as e:
            logger.warning(f"[session] 画像写回失败（不影响答题结果）: {e}")

    return AnswerResult(
        correct=correct,
        correct_answer=question.answer,
        explanation=question.explanation,
        is_last=is_last,
        next_index=current_index + 1 if not is_last else None,
    )


async def _write_back_profile(session: QuizSession, session_id: str) -> None:
    """会话完成后把成绩写回画像（session_briefs / error_log / mastery EMA / weak_points）。

    与 adapt_writer 走同一套 write_episodic_memory + update_semantic_memory，
    保证两条轨道（/session/* 与 /agent/stream grade）的记忆口径一致。
    本轨道为字符串比较批改，无 LLM 提炼的 knowledge_gap，错题以题干截断兜底。
    """
    grades = []
    correct_count = 0
    for i, (q, ua) in enumerate(zip(session.questions, session.user_answers)):
        is_correct = ua.strip().upper() == q.answer.strip().upper()
        correct_count += is_correct
        grades.append(QuestionGrade(
            index=i,
            question=q.question,
            user_answer=ua,
            correct_answer=q.answer,
            is_correct=is_correct,
            ai_feedback=None if is_correct else q.explanation,
            knowledge_gap=None if is_correct else q.question[:40],
        ))
    total = len(session.questions)
    report = GradingReport(
        session_id=session_id,
        total=total,
        correct=correct_count,
        score=round(correct_count / total, 2) if total else 0.0,
        grades=grades,
    )
    await write_episodic_memory(session.user_id, report, session.document_id)
    await update_semantic_memory(session.user_id, report, session.document_id)


async def get_result(session_id: str) -> SessionResult:
    session = sessions.get(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")

    details = []
    correct_count = 0
    for i, (q, user_ans) in enumerate(zip(session.questions, session.user_answers)):
        is_correct = user_ans.strip().upper() == q.answer.strip().upper()
        if is_correct:
            correct_count += 1
        details.append({
            "index": i,
            "question": q.question,
            "user_answer": user_ans,
            "correct_answer": q.answer,
            "correct": is_correct,
            "explanation": q.explanation,
        })

    total = len(session.questions)
    return SessionResult(
        session_id=session_id,
        document_id=session.document_id,
        total=total,
        correct=correct_count,
        score=round(correct_count / total, 2) if total > 0 else 0.0,
        details=details,
    )
