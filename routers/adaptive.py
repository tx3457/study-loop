"""
自适应学习闭环端点

该端点运行多轮「出题 → 作答 → 批改 → 决策」闭环，由 LLM 根据学生表现
决定下一步主题和难度，直到掌握度达标或达到轮次上限。

两段式 HTTP(复用 autonomous 的 HITL 思路,前端无状态):
  POST /agent/adaptive/start  {user_id, document_id, goal}
      → agent 决策开场 → 出题 → {adaptive_session_id, questions, decision, turn=1}
  POST /agent/adaptive/submit {adaptive_session_id, answers[], turn}
      → 批改 → 更新画像(EMA mastery + weak_points)→ agent 决策下一步 → 终止?
            是 → {done:true, summary, trajectory}
            否 → {questions, decision, turn+1, last_report}

"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from chromadb.errors import ChromaError, NotFoundError
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field, ValidationError

from models.adaptive import AdaptiveTurn, NextStepDecision
from models.grader import GradingReport
from models.session import QuestionView, QuizSession
from services.adaptive_loop import decide_next_step, generate_lesson, should_terminate
from services.injection import check_injection
from services.grader import grade_session
from services.learning_path import generate_learning_path
from services.memory import (
    append_decision,
    get_mastery,
    get_weak_points,
    update_semantic_memory,
    write_episodic_memory,
)
from services.rag import generate_question
from services.idempotency import (
    abort_idempotency_claim,
    normalize_idempotency_key,
    request_idempotency,
)
from services.session import (
    InvalidQuizResponseError,
    sessions,
    validate_provider_questions,
)
from services.tool_registry import SideEffectAmbiguousError
from services.vectorstore import ensure_document_available

router = APIRouter()
logger = logging.getLogger(__name__)

SESSION_TTL_SEC = 3600  # 1 小时未续跑视为过期
SESSION_MAX_COUNT = 200  # 最多保 200 个 session(FIFO 淘汰)


# ═══════════════════════════════════════════════════════════════════════════
# 会话状态(多轮闭环跨 HTTP 请求保存)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class AdaptiveSession:
    adaptive_session_id: str
    user_id: str
    document_id: str
    goal: str
    turn: int
    history: list[AdaptiveTurn]
    current_quiz_session_id: Optional[str] = None
    current_decision: Optional[NextStepDecision] = None
    current_turn_type: str = "quiz"  # "quiz" | "teach"
    last_report: Optional[GradingReport] = None  # 最近一次批改(teach 轮承接表现上下文)
    done: bool = False
    created_at: float = field(default_factory=time.time)


_sessions: dict[str, AdaptiveSession] = {}
_submit_locks: dict[str, asyncio.Lock] = {}


def _purge_expired() -> None:
    now = time.time()
    for cid in [
        c for c, s in _sessions.items() if now - s.created_at > SESSION_TTL_SEC
    ]:
        _sessions.pop(cid, None)
        _submit_locks.pop(cid, None)


def _save_session(s: AdaptiveSession) -> None:
    _purge_expired()
    if len(_sessions) >= SESSION_MAX_COUNT and s.adaptive_session_id not in _sessions:
        oldest = min(_sessions.values(), key=lambda x: x.created_at)
        _sessions.pop(oldest.adaptive_session_id, None)
        _submit_locks.pop(oldest.adaptive_session_id, None)
    _sessions[s.adaptive_session_id] = s


# ═══════════════════════════════════════════════════════════════════════════
# 请求 / 响应模型
# ═══════════════════════════════════════════════════════════════════════════
class AdaptiveStartRequest(BaseModel):
    user_id: str = Field(default="default", description="用户 ID")
    document_id: str = Field(..., description="已建库的文档 ID")
    goal: str = Field(..., description="学习目标 / 主题")


class AdaptiveSubmitRequest(BaseModel):
    adaptive_session_id: str
    turn: int = Field(ge=1, description="客户端正在回答的轮次")
    answers: list[str] = Field(..., description="本轮逐题作答,顺序与下发题目一致")


class AdaptiveTurnResponse(BaseModel):
    adaptive_session_id: str
    turn: int
    done: bool = False
    turn_type: str = "quiz"  # "quiz"(出题轮)| "teach"(讲解轮)
    questions: list[QuestionView] = Field(default_factory=list)
    lesson: Optional[str] = None  # turn_type=teach 时的纯讲解内容
    decision: Optional[NextStepDecision] = None  # agent 本步的决策(含 reason,可解释)
    last_report_score: Optional[float] = None
    last_report_gaps: list[str] = Field(default_factory=list)
    last_report_feedback: list[dict] = Field(
        default_factory=list
    )  # 逐题反馈(grader 已产出)
    mastery: Optional[float] = None
    trajectory: list[AdaptiveTurn] = Field(default_factory=list)
    summary: str = ""
    terminate_reason: str = ""
    learning_path: Optional[dict] = None  # switch_to_plan 时填


# ═══════════════════════════════════════════════════════════════════════════
# 编排 helper
# ═══════════════════════════════════════════════════════════════════════════
async def _serve_turn(
    asess: AdaptiveSession, decision: NextStepDecision, turn: int
) -> list[QuestionView]:
    """按 agent 决策出一轮题,建底层 QuizSession,记一条轨迹(得分待批改后回填)。"""
    quiz = await generate_question(
        asess.document_id,
        decision.topic,
        decision.count,
        decision.difficulty,
        decision.question_type,
        difficulty_score=decision.difficulty_score,
        weak_points=decision.target_weak_points,
    )
    questions = getattr(quiz, "questions", None)
    if not isinstance(questions, list):
        raise InvalidQuizResponseError("模型题目响应结构无效")
    validate_provider_questions(questions, decision.question_type)
    qsid = str(uuid.uuid4())
    sessions[qsid] = QuizSession(
        session_id=qsid,
        document_id=asess.document_id,
        user_id=asess.user_id,
        questions=questions,
        user_answers=[],
        status="active",
    )
    asess.current_quiz_session_id = qsid
    asess.current_decision = decision
    asess.current_turn_type = "quiz"
    asess.history.append(
        AdaptiveTurn(
            turn=turn,
            action=decision.action,
            topic=decision.topic,
            difficulty_score=decision.difficulty_score,
            reason=decision.reason,
        )
    )
    return [
        QuestionView(index=i, question=q.question, options=q.options)
        for i, q in enumerate(quiz.questions)
    ]


async def _serve_teach_turn(
    asess: AdaptiveSession, decision: NextStepDecision, turn: int
) -> str:
    """teach 轮:生成纯讲解(不出题),记一条轨迹(无得分)。返回讲解文本。"""
    # 薄弱点:优先用 agent 指定的;没有则取上一轮(刚批改的题)暴露的盲点
    wp = decision.target_weak_points
    if not wp and asess.history:
        wp = asess.history[-1].knowledge_gaps
    lesson = await generate_lesson(
        document_id=asess.document_id,
        topic=decision.topic,
        weak_points=wp or [],
        last_report=asess.last_report,
    )
    asess.current_quiz_session_id = None
    asess.current_decision = decision
    asess.current_turn_type = "teach"
    asess.history.append(
        AdaptiveTurn(
            turn=turn,
            action="teach",
            topic=decision.topic,
            difficulty_score=decision.difficulty_score,
            reason=decision.reason,
        )
    )
    return lesson


def _report_feedback(report: GradingReport) -> list[dict]:
    """把 grader 已产出的逐题反馈整理给前端(现成内容,别浪费)。"""
    return [
        {
            "index": g.index,
            "question": g.question,
            "your_answer": g.user_answer,
            "correct_answer": g.correct_answer,
            "is_correct": g.is_correct,
            "ai_feedback": g.ai_feedback,
            "knowledge_gap": g.knowledge_gap,
        }
        for g in report.grades
    ]


async def _grade_and_update(
    asess: AdaptiveSession, answers: list[str]
) -> GradingReport:
    """填答案 → 批改 → 更新画像(EMA mastery + weak_points + 审计)→ 回填本轮轨迹得分。"""
    qsid = asess.current_quiz_session_id
    qs = sessions.get(qsid) if qsid else None
    if qs is None:
        raise HTTPException(status_code=400, detail="当前没有待批改的题目")
    if len(answers) != len(qs.questions):
        raise HTTPException(
            status_code=400,
            detail=f"答案数 {len(answers)} 与题目数 {len(qs.questions)} 不符",
        )
    qs.user_answers = list(answers)
    qs.status = "completed"

    report = await grade_session(qsid)
    asess.last_report = report
    await update_semantic_memory(
        asess.user_id, report, asess.document_id
    )  # EMA mastery + weak_points
    await write_episodic_memory(
        asess.user_id, report, asess.document_id
    )  # session_briefs + error_log
    if asess.current_decision is not None:
        await append_decision(
            asess.user_id,
            {  # 决策审计 trace
                "agent": "adaptive_loop",
                "decision": asess.current_decision.action,
                "rationale": asess.current_decision.reason,
                "turn": asess.turn,
                "score": report.score,
            },
        )

    # 回填本轮轨迹(history[-1] 对应刚答完这套题)
    if asess.history:
        gaps = [
            g.knowledge_gap
            for g in report.grades
            if not g.is_correct and g.knowledge_gap
        ]
        asess.history[-1].score = report.score
        asess.history[-1].mastery_after = await get_mastery(
            asess.user_id, asess.document_id
        )
        asess.history[-1].knowledge_gaps = gaps
    return report


def _build_summary(
    asess: AdaptiveSession,
    mastery: Optional[float],
    term_reason: str,
    decision: NextStepDecision,
) -> str:
    reason_map = {
        "agent_finish": "agent 判定可结束",
        "mastery_reached": "掌握度达标",
        "max_turns": "达到最大轮次",
        "switch_to_plan": "转入系统学习路径",
    }
    traj = " → ".join(
        f"T{t.turn}({t.score:.2f})" if t.score is not None else f"T{t.turn}(-)"
        for t in asess.history
    )
    m = f"{mastery:.2f}" if mastery is not None else "未知"
    return (
        f"结束原因:{reason_map.get(term_reason, term_reason)}。共 {len(asess.history)} 轮,"
        f"最终掌握度 {m}。得分轨迹:{traj}。最后评估:{decision.reason}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 端点
# ═══════════════════════════════════════════════════════════════════════════
async def _execute_adaptive_start(req: AdaptiveStartRequest) -> AdaptiveTurnResponse:
    """开启一个自适应辅导会话:agent 决策开场策略 → 出第一轮题。"""
    is_injection, reason = await check_injection(req.goal)
    if is_injection:
        raise HTTPException(status_code=400, detail=f"输入安全检查未通过:{reason}")

    mastery = await get_mastery(req.user_id, req.document_id)
    weak_points = await get_weak_points(req.user_id, req.document_id)

    # 开场也走 agent 决策(last_report=None),让闭环端到端由 agent 驱动
    decision = await decide_next_step(
        goal=req.goal,
        mastery=mastery,
        weak_points=weak_points,
        history=[],
        last_report=None,
    )

    asess = AdaptiveSession(
        adaptive_session_id=f"adapt_{uuid.uuid4().hex[:16]}",
        user_id=req.user_id,
        document_id=req.document_id,
        goal=req.goal,
        turn=1,
        history=[],
    )
    logger.info(
        f"[adaptive] start sid={asess.adaptive_session_id} open={decision.action} "
        f"topic={decision.topic} diff={decision.difficulty_score:.2f}"
    )

    # 开场可能直接讲解(teach)或出题(其余动作)
    if decision.action == "teach":
        lesson = await _serve_teach_turn(asess, decision, turn=1)
        _save_session(asess)
        return AdaptiveTurnResponse(
            adaptive_session_id=asess.adaptive_session_id,
            turn=1,
            turn_type="teach",
            lesson=lesson,
            decision=decision,
            mastery=mastery,
            trajectory=asess.history,
        )
    views = await _serve_turn(asess, decision, turn=1)
    _save_session(asess)
    return AdaptiveTurnResponse(
        adaptive_session_id=asess.adaptive_session_id,
        turn=1,
        turn_type="quiz",
        questions=views,
        decision=decision,
        mastery=mastery,
        trajectory=asess.history,
    )


@router.post("/agent/adaptive/start", response_model=AdaptiveTurnResponse)
async def adaptive_start(req: AdaptiveStartRequest) -> AdaptiveTurnResponse:
    try:
        await ensure_document_available(req.document_id)
        return await _execute_adaptive_start(req)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail="文档不存在") from exc
    except ChromaError as exc:
        logger.exception("adaptive start document lookup failed")
        raise HTTPException(status_code=503, detail="文档存储暂时不可用") from exc
    except (InvalidQuizResponseError, ValidationError) as exc:
        logger.warning(
            "adaptive provider returned invalid structured output: %s",
            type(exc).__name__,
        )
        raise HTTPException(status_code=503, detail="模型返回的题目格式无效") from exc


async def _execute_adaptive_submit(
    req: AdaptiveSubmitRequest,
    *,
    before_effect,
) -> AdaptiveTurnResponse:
    """推进闭环:出题轮→批改+更新画像;讲解轮→直接推进。再 agent 决策下一步 → 终止/下一轮。

    前端约定:讲解(teach)轮没有题,前端读完点"继续",submit 传 answers=[] 即可推进。
    """
    asess = _sessions.get(req.adaptive_session_id)
    if asess is None:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    if asess.done:
        raise HTTPException(status_code=409, detail="会话已结束")
    if req.turn != asess.turn:
        raise HTTPException(status_code=409, detail="提交轮次已过期，请刷新后重试")

    prev_was_teach = asess.current_turn_type == "teach"
    if prev_was_teach:
        if req.answers:
            raise HTTPException(status_code=400, detail="讲解轮不接受题目答案")
        # 讲解轮无题可批,直接推进;承接上一份成绩上下文(asess.last_report)
        report = asess.last_report
        feedback: list[dict] = []
        last_score = None
        last_gaps: list[str] = []
    else:
        qsid = asess.current_quiz_session_id
        quiz_session = sessions.get(qsid) if qsid else None
        if quiz_session is None:
            raise HTTPException(status_code=409, detail="当前没有待批改的题目")
        if len(req.answers) != len(quiz_session.questions):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"答案数 {len(req.answers)} 与题目数 "
                    f"{len(quiz_session.questions)} 不符"
                ),
            )
        # 出题轮:批改 + 更新画像 + 回填轨迹
        await before_effect()
        report = await _grade_and_update(asess, req.answers)
        feedback = _report_feedback(report)
        last_score = report.score
        last_gaps = [
            g.knowledge_gap
            for g in report.grades
            if not g.is_correct and g.knowledge_gap
        ]

    mastery = await get_mastery(asess.user_id, asess.document_id)
    weak_points = await get_weak_points(asess.user_id, asess.document_id)

    # agent 推理下一步;上一步若是 teach,禁止再 teach(强制出题验证)
    decision = await decide_next_step(
        goal=asess.goal,
        mastery=mastery,
        weak_points=weak_points,
        history=asess.history,
        last_report=report,
        allow_teach=not prev_was_teach,
    )

    # Quiz 轮在批改前已经越过副作用边界；teach 轮直到这里仍可安全重试。
    if prev_was_teach:
        await before_effect()

    # 终止判定(达标 / 轮数 / agent 主动结束)
    terminate, term_reason = should_terminate(
        mastery=mastery, turn=asess.turn, decision=decision
    )
    if terminate:
        asess.done = True
        _save_session(asess)
        return AdaptiveTurnResponse(
            adaptive_session_id=asess.adaptive_session_id,
            turn=asess.turn,
            done=True,
            decision=decision,
            last_report_score=last_score,
            last_report_gaps=last_gaps,
            last_report_feedback=feedback,
            mastery=mastery,
            trajectory=asess.history,
            summary=_build_summary(asess, mastery, term_reason, decision),
            terminate_reason=term_reason,
        )

    # agent 决定转系统学习路径规划 → 生成路径并结束
    if decision.action == "switch_to_plan":
        asess.done = True
        _save_session(asess)
        path = await generate_learning_path(asess.document_id)
        return AdaptiveTurnResponse(
            adaptive_session_id=asess.adaptive_session_id,
            turn=asess.turn,
            done=True,
            decision=decision,
            last_report_score=last_score,
            last_report_gaps=last_gaps,
            last_report_feedback=feedback,
            mastery=mastery,
            trajectory=asess.history,
            learning_path=path.model_dump(),
            summary=_build_summary(asess, mastery, "switch_to_plan", decision),
            terminate_reason="switch_to_plan",
        )

    asess.turn += 1

    # 讲解轮:生成纯讲解,不出题
    if decision.action == "teach":
        lesson = await _serve_teach_turn(asess, decision, turn=asess.turn)
        _save_session(asess)
        return AdaptiveTurnResponse(
            adaptive_session_id=asess.adaptive_session_id,
            turn=asess.turn,
            turn_type="teach",
            lesson=lesson,
            decision=decision,
            last_report_score=last_score,
            last_report_gaps=last_gaps,
            last_report_feedback=feedback,
            mastery=mastery,
            trajectory=asess.history,
        )

    # 出题轮
    views = await _serve_turn(asess, decision, turn=asess.turn)
    _save_session(asess)
    return AdaptiveTurnResponse(
        adaptive_session_id=asess.adaptive_session_id,
        turn=asess.turn,
        turn_type="quiz",
        questions=views,
        decision=decision,
        last_report_score=last_score,
        last_report_gaps=last_gaps,
        last_report_feedback=feedback,
        mastery=mastery,
        trajectory=asess.history,
    )


@router.post("/agent/adaptive/submit", response_model=AdaptiveTurnResponse)
async def adaptive_submit(
    req: AdaptiveSubmitRequest,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
) -> AdaptiveTurnResponse:
    """Advance one adaptive turn with optional durable retry protection."""
    key = normalize_idempotency_key(idempotency_key)
    claimed = False
    if key:
        decision = await request_idempotency.begin(
            key, "agent.adaptive.submit", req.model_dump(mode="json")
        )
        if decision.replayed:
            return AdaptiveTurnResponse.model_validate(decision.response)
        claimed = True

    effect_started = False

    async def mark_effect() -> None:
        nonlocal effect_started
        if effect_started:
            return
        if key:
            await request_idempotency.mark_effect_started(key, "adaptive_submit")
        effect_started = True

    try:
        if req.adaptive_session_id not in _sessions:
            raise HTTPException(status_code=404, detail="会话不存在或已过期")

        lock = _submit_locks.setdefault(req.adaptive_session_id, asyncio.Lock())
        async with lock:
            response = await _execute_adaptive_submit(req, before_effect=mark_effect)
            if key:
                await request_idempotency.complete(
                    key, response.model_dump(mode="json")
                )
            return response
    except BaseException as exc:
        durable_effect = False
        if claimed and key:
            try:
                durable_effect = await abort_idempotency_claim(
                    request_idempotency,
                    key,
                )
            except BaseException:
                logger.exception("adaptive submit receipt cleanup failed")

        ambiguous = effect_started or durable_effect
        if ambiguous:
            _sessions.pop(req.adaptive_session_id, None)
            _submit_locks.pop(req.adaptive_session_id, None)

        if isinstance(exc, asyncio.CancelledError):
            raise
        if ambiguous and not isinstance(exc, SideEffectAmbiguousError):
            raise SideEffectAmbiguousError("adaptive_submit") from exc
        raise
