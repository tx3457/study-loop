"""
Supervisor-based MAS guided 辅导实验端点（Lab）

这是一个由 TeachingSupervisor 动态编排的实验工作流——
  diagnostic → quiz → critic →(reviser↔critic 精修)→ wait_for_answers(HITL 暂停)
            → [Command(resume=answers)] → grader → supervisor 决策下一轮 / finish

两段式 HTTP（HITL：用 LangGraph interrupt + checkpointer 持久化中断点，跨请求 thread_id 续跑）：
  POST /agent/tutor/start  {document_id, goal}   身份取自 Authorization 头
      → 跑到 wait_for_answers interrupt 暂停 → {thread_id, quiz, awaiting_answers, turn, supervisor_reason}
  POST /agent/tutor/submit {thread_id, quiz_session_id, answers}
      → Command(resume=answers) 续跑 grader→supervisor→下一轮(下一个 interrupt) 或 finish
      → {thread_id, grading_report, quiz?, done, mastery, supervisor_reason, awaiting_answers}

边界：纯讲解 tutor worker 仍未完成，也没有 Web 页面。默认不注册这些路由；
仅在进程启动前设置 MAS_SUPERVISOR_ENABLED=true 时用于开发和架构实验。

"""
import asyncio
import logging
import uuid
from weakref import WeakValueDictionary

from fastapi import APIRouter, Depends, HTTPException

from services.auth import require_user_id
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator

from agents.supervisor import supervisor_enabled
from agents.tutor_graph import compile_tutor_graph, tutor_graph
from services.checkpoint import default_checkpoint_path, open_sqlite_checkpointer
from services.memory import get_mastery
from services.memory_context import build_returning_context
from services.tutor_sessions import tutor_session_id

router = APIRouter(tags=["Experimental Tutor Lab"])
logger = logging.getLogger(__name__)

# tutor guided 会话用独立 checkpoint db（与 orchestrator 的 db 隔离，避免 thread_id 串台）
_TUTOR_DB_PATH = default_checkpoint_path().replace("orchestrator.db", "tutor.db")
_SUBMIT_LOCKS: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()


# ═══════════════════════════════════════════════════════════════════════════
# 请求 / 响应模型
# ═══════════════════════════════════════════════════════════════════════════
class TutorStartRequest(BaseModel):
    document_id: str = Field(..., description="已建库的文档 ID")
    goal: str = Field(..., description="学习目标 / 主题")


class TutorSubmitRequest(BaseModel):
    thread_id: str = Field(..., description="start 返回的会话线程 ID")
    quiz_session_id: str = Field(
        ...,
        min_length=8,
        max_length=80,
        description="当前待作答题目的会话令牌",
    )
    answers: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="本轮逐题作答，顺序与下发题目一致",
    )

    @field_validator("answers")
    @classmethod
    def validate_answers(cls, answers: list[str]) -> list[str]:
        normalized = []
        for answer in answers:
            answer = answer.strip()
            if not answer:
                raise ValueError("answers must not contain blanks")
            if len(answer) > 4000:
                raise ValueError("each answer must be at most 4000 characters")
            normalized.append(answer)
        return normalized


class TutorOneshotRequest(BaseModel):
    """oneshot 单跳请求：action 明确，supervisor 单跳到对应 worker → finish。"""
    action: str = Field(default="quiz", description="quiz / grade / plan")
    document_id: str = Field(..., description="已建库的文档 ID")
    description: str = Field(default="", description="出题主题 / 学习目标（quiz/plan 用）")
    count: int = Field(default=3, ge=1, le=10, description="题目数量（quiz 用）")
    difficulty: str = Field(default="medium", description="easy/medium/hard（quiz 用）")
    type: str = Field(default="choice", description="choice/true_false/short_answer（quiz 用）")
    session_id: str | None = Field(default=None, description="待批改会话 ID（grade 用，作答已在 session 内）")


class TutorOneshotResponse(BaseModel):
    """oneshot 响应：按 action 回 quiz / grading_report / learning_path 之一。"""
    action: str
    quiz: dict | None = None
    grading_report: dict | None = None
    learning_path: dict | None = None
    terminate_reason: str = ""
    supervisor_reason: str = ""


class TutorAssistRequest(BaseModel):
    """assist 自由问答请求：supervisor 路由到 assistant ReAct worker。"""
    query: str = Field(..., description="用户自然语言问题 / 学习目标")
    document_id: str | None = Field(default=None, description="可选文档 ID")


class TutorAssistContinueRequest(BaseModel):
    """assist 续跑请求：assistant ask_user interrupt 后用户的回答。"""
    thread_id: str = Field(..., description="assist 返回的会话线程 ID")
    user_reply: str = Field(..., description="用户对 ask_user 问题的回答")


class TutorAssistResponse(BaseModel):
    """assist 响应：finalize → final_answer；ask_user → awaiting_user_input + user_question。"""
    thread_id: str
    awaiting_user_input: bool = False
    done: bool = False
    final_answer: str = ""
    user_question: str | None = None
    tools_called: list[str] = Field(default_factory=list)


class TutorTurnResponse(BaseModel):
    thread_id: str
    quiz_session_id: str | None = None
    awaiting_answers: bool = False
    done: bool = False
    turn: int = 0
    quiz: dict | None = None                       # 待作答题目（awaiting_answers=True 时）
    lesson: str | None = None                      # 本轮讲解正文（supervisor 派了 tutor 才有）
    grading_report: dict | None = None             # 上一轮批改结果（submit 后）
    mastery: float | None = None
    supervisor_reason: str = ""
    terminate_reason: str = ""
    history: list[dict] = Field(default_factory=list)
    returning_context: dict | None = None      # 跨会话"欢迎回来"上下文（start 时回传；新用户 is_returning=False）
    welcome_back: str = ""                      # 欢迎回来文案（回访用户才有，前端可直接展示）


# ═══════════════════════════════════════════════════════════════════════════
# 编排 helper
# ═══════════════════════════════════════════════════════════════════════════
def _require_enabled() -> None:
    """防御性复核：处理启动后开关变化或直接函数调用。"""
    if not supervisor_enabled():
        raise HTTPException(
            status_code=503,
            detail="supervisor-based MAS 未启用（MAS_SUPERVISOR_ENABLED=false）",
        )


def _extract_interrupt(result: dict) -> dict | None:
    """从 ainvoke 返回值取 interrupt payload（langgraph 1.1.3：result['__interrupt__'][0].value）。"""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    # Interrupt 对象有 .value；防御性兼容 dict 形态
    return getattr(first, "value", None) if not isinstance(first, dict) else first.get("value")


async def _mastery_of(user_id: str, document_id: str) -> float | None:
    try:
        return await get_mastery(user_id, document_id)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 端点
# ═══════════════════════════════════════════════════════════════════════════
@router.post("/agent/tutor/start", response_model=TutorTurnResponse)
async def tutor_start(
    req: TutorStartRequest, user_id: str = Depends(require_user_id)
) -> TutorTurnResponse:
    """开启一个 supervisor 辅导会话：跑到首个 wait_for_answers interrupt 暂停，回题目等作答。"""
    _require_enabled()
    thread_id = f"{user_id}:{req.document_id}:{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}
    # 跨会话记忆：开场即读"欢迎回来"上下文（注入 init_state 供 supervisor，并回传前端）
    returning_context = await build_returning_context(user_id, req.document_id)
    welcome_back = returning_context.get("welcome_msg", "") if returning_context.get("is_returning") else ""
    init_state = {
        "thread_id": thread_id,
        "user_id": user_id,
        "document_id": req.document_id,
        "goal": req.goal,
        "description": req.goal,
        "mode": "guided",
        "turn": 0,
        "handoff_count": 0,
        "history": [],
        "returning_context": returning_context,
    }

    async with open_sqlite_checkpointer(_TUTOR_DB_PATH) as cp:
        graph = compile_tutor_graph(cp)
        result = await graph.ainvoke(init_state, config=config)

    payload = _extract_interrupt(result)
    if payload is not None:
        # 跑到 wait_for_answers 暂停 → 回题目等作答
        return TutorTurnResponse(
            thread_id=thread_id,
            quiz_session_id=payload.get("session_id"),
            awaiting_answers=True,
            turn=payload.get("turn", 0),
            quiz=payload.get("quiz"),
            lesson=result.get("lesson"),
            supervisor_reason=payload.get("supervisor_reason", "") or result.get("supervisor_reason", ""),
            returning_context=returning_context,
            welcome_back=welcome_back,
        )

    # 没暂停就结束了（如冷启动即 finish / 异常收尾）
    return TutorTurnResponse(
        thread_id=thread_id,
        done=bool(result.get("done")),
        turn=result.get("turn", 0),
        lesson=result.get("lesson"),
        supervisor_reason=result.get("supervisor_reason", ""),
        terminate_reason=result.get("terminate_reason", ""),
        mastery=await _mastery_of(user_id, req.document_id),
        history=result.get("history", []) or [],
        returning_context=returning_context,
        welcome_back=welcome_back,
    )


@router.post("/agent/tutor/submit", response_model=TutorTurnResponse)
async def tutor_submit(req: TutorSubmitRequest) -> TutorTurnResponse:
    """续跑：Command(resume=answers) → grader → supervisor 决策 → 下一轮 interrupt 或 finish。"""
    _require_enabled()
    config = {"configurable": {"thread_id": req.thread_id}}

    submit_lock = _SUBMIT_LOCKS.setdefault(req.thread_id, asyncio.Lock())
    async with submit_lock:
        async with open_sqlite_checkpointer(_TUTOR_DB_PATH) as cp:
            graph = compile_tutor_graph(cp)
            # 恢复前先确认该 thread 存在且确实停在 wait_for_answers。
            snapshot = await graph.aget_state(config)
            if not snapshot.values:
                raise HTTPException(status_code=404, detail="会话不存在或已过期")
            if not snapshot.next:
                raise HTTPException(status_code=400, detail="会话已结束，无待续跑的中断点")
            if "wait_for_answers" in snapshot.next:
                pending_session_id = tutor_session_id(snapshot.values)
                if req.quiz_session_id != pending_session_id:
                    raise HTTPException(
                        status_code=409,
                        detail="题目会话令牌已过期，请使用最新一轮题目",
                    )
                result = await graph.ainvoke(
                    Command(resume=req.answers),
                    config=config,
                )
            elif "grader" in snapshot.next:
                # The answer node committed, but the process/request stopped
                # before grader ran. Continue that checkpoint only when the
                # retry is an exact replay of the committed turn.
                committed_session_id = snapshot.values.get("session_id")
                committed_answers = list(snapshot.values.get("answers") or [])
                if (
                    req.quiz_session_id != committed_session_id
                    or req.answers != committed_answers
                ):
                    raise HTTPException(
                        status_code=409,
                        detail="该轮答案已提交，重试内容与已提交内容不一致",
                    )
                result = await graph.ainvoke(None, config=config)
            else:
                raise HTTPException(status_code=409, detail="会话当前不在等待答案状态")

    user_id = result.get("user_id", "")
    document_id = result.get("document_id", "")
    mastery = await _mastery_of(user_id, document_id)
    grading_report = result.get("last_report") or result.get("grading_report")

    payload = _extract_interrupt(result)
    if payload is not None:
        # 续跑后又停在下一轮 wait_for_answers → 下一份题目
        return TutorTurnResponse(
            thread_id=req.thread_id,
            quiz_session_id=payload.get("session_id"),
            awaiting_answers=True,
            turn=payload.get("turn", 0),
            quiz=payload.get("quiz"),
            lesson=result.get("lesson"),
            grading_report=grading_report,
            mastery=mastery,
            supervisor_reason=payload.get("supervisor_reason", "") or result.get("supervisor_reason", ""),
            history=result.get("history", []) or [],
        )

    # 跑到 finish 收尾
    return TutorTurnResponse(
        thread_id=req.thread_id,
        done=bool(result.get("done")),
        turn=result.get("turn", 0),
        lesson=result.get("lesson"),
        grading_report=grading_report,
        mastery=mastery,
        supervisor_reason=result.get("supervisor_reason", ""),
        terminate_reason=result.get("terminate_reason", ""),
        history=result.get("history", []) or [],
    )


@router.post("/agent/tutor/oneshot", response_model=TutorOneshotResponse)
async def tutor_oneshot(
    req: TutorOneshotRequest, user_id: str = Depends(require_user_id)
) -> TutorOneshotResponse:
    """oneshot 单跳端点：supervisor 按 action 单跳到对应 worker → finish。

    与 start/submit 不同：mode="oneshot" 不进 guided 循环、不 interrupt，一次 ainvoke 跑到收尾。
    无需 checkpointer（无中断点），用默认 tutor_graph 即可。
      action="quiz"  → 单跳 quiz（可选过一次 critic）→ {quiz}
      action="grade" → 单跳 grader（读 session_id 批改）→ {grading_report}
      action="plan"  → 单跳 planner → {learning_path}
    """
    _require_enabled()
    init_state = {
        "action": req.action,
        "user_id": user_id,
        "document_id": req.document_id,
        "description": req.description,
        "goal": req.description,
        "count": req.count,
        "difficulty": req.difficulty,
        "type": req.type,
        "session_id": req.session_id,
        "mode": "oneshot",
        "turn": 0,
        "handoff_count": 0,
        "history": [],
    }

    result = await tutor_graph.ainvoke(init_state)

    grading_report = result.get("grading_report") or result.get("last_report")
    return TutorOneshotResponse(
        action=req.action,
        quiz=result.get("quiz"),
        grading_report=grading_report,
        learning_path=result.get("learning_path"),
        terminate_reason=result.get("terminate_reason", ""),
        supervisor_reason=result.get("supervisor_reason", ""),
    )


def _assist_response(thread_id: str, result: dict) -> TutorAssistResponse:
    """把 assist ainvoke 结果转响应：ask_user interrupt → awaiting；否则 finalize 收尾。"""
    payload = _extract_interrupt(result)
    if payload is not None:
        # assistant 内部 ask_user → interrupt 暂停 → 回问题等用户答
        return TutorAssistResponse(
            thread_id=thread_id,
            awaiting_user_input=True,
            user_question=payload.get("question", ""),
            tools_called=result.get("tools_called", []) or [],
        )
    return TutorAssistResponse(
        thread_id=thread_id,
        done=bool(result.get("done")),
        final_answer=result.get("final_answer", ""),
        tools_called=result.get("tools_called", []) or [],
    )


@router.post("/agent/tutor/assist", response_model=TutorAssistResponse)
async def tutor_assist(
    req: TutorAssistRequest, user_id: str = Depends(require_user_id)
) -> TutorAssistResponse:
    """assist 自由问答端点：supervisor 路由到 assistant ReAct worker。

    mode="assist"，带 checkpointer + thread_id（assistant 的 ask_user 用 interrupt 暂停需要）。
    跑到 finalize（done + final_answer）或 ask_user interrupt（awaiting_user_input + user_question）。
    """
    _require_enabled()
    thread_id = f"{user_id}:assist:{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}
    init_state = {
        "thread_id": thread_id,
        "user_id": user_id,
        "document_id": req.document_id,
        "goal": req.query,
        "description": req.query,
        "mode": "assist",
        "turn": 0,
        "handoff_count": 0,
        "history": [],
        "messages": [],
        "tools_called": [],
    }

    async with open_sqlite_checkpointer(_TUTOR_DB_PATH) as cp:
        graph = compile_tutor_graph(cp)
        result = await graph.ainvoke(init_state, config=config)

    return _assist_response(thread_id, result)


@router.post("/agent/tutor/assist/continue", response_model=TutorAssistResponse)
async def tutor_assist_continue(req: TutorAssistContinueRequest) -> TutorAssistResponse:
    """assist 续跑：assistant ask_user interrupt 后，Command(resume=user_reply) 续跑 ReAct。"""
    _require_enabled()
    config = {"configurable": {"thread_id": req.thread_id}}

    async with open_sqlite_checkpointer(_TUTOR_DB_PATH) as cp:
        graph = compile_tutor_graph(cp)
        snapshot = await graph.aget_state(config)
        if not snapshot.values:
            raise HTTPException(status_code=404, detail="会话不存在或已过期")
        if not snapshot.next:
            raise HTTPException(status_code=400, detail="会话已结束，无待续跑的中断点")

        result = await graph.ainvoke(Command(resume=req.user_reply), config=config)

    return _assist_response(req.thread_id, result)
