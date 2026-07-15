"""
Autonomous Agent 端点（Phase 8 P2 升级：Plan-and-Execute → 真 ReAct + HITL）

── 升级前 ──────────────────────────────────────────────────────────────────
- Plan-and-Execute：强制生成 plan → 强制 LLM 第一轮调工具 → 检测无 tool_calls 判结束
- 缺陷：剥夺 LLM 决策权、退出靠"猜"、无 HITL、LLM 健忘

── 升级后 ──────────────────────────────────────────────────────────────────
- 真 ReAct：LLM 全权决定每轮动作（含"不调工具"）
- finalize(reason, final_answer)：LLM 主动声明结束 + 给理由（explicit > implicit）
- ask_user(question)：LLM 卡住时求助用户，触发两段式 HITL
- 每轮 inject [Current state] summary：LLM 不健忘
- plan 降级为可选 hint：短 query 跳过 plan，省一次 LLM call

── 两段式 HITL 协议（ask_user）──────────────────────────────────────────────
  Step 1: POST /agent/autonomous
          → LLM 调 ask_user → 响应 awaiting_user_input=True + conversation_id + question
  Step 2: 前端弹框，用户输入答案
  Step 3: POST /agent/autonomous/continue
          → 携带 conversation_id + user_reply → 后端恢复 messages 继续 ReAct loop

── 面试讲点 ────────────────────────────────────────────────────────────────
1. 范式升级：Plan-and-Execute → 真 ReAct（参考 Yao et al. 2022）
2. Explicit > implicit：finalize 工具取代"无 tool_calls 即结束"的脆弱信号
3. Web HITL：两段式 HTTP 协议（无状态可扩展），替代 NovelClaw 的文件 IPC
4. State summary injection：context engineering 让 LLM 不健忘
5. 复用 P1-2 的 ToolRegistry，dispatch_tool 自带超时 / 重试 / audit
"""
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from models.citation import CitationView, GroundingStatus
from services.citations import (
    EvidenceRegistry,
    collect_search_evidence,
    resolve_citations,
)
from services.injection import check_injection, check_output_leak
from services.llm import llm_chat
from services.react_controls import (
    CONTROL_TOOL_NAMES,
    build_control_tools,
    build_react_decision_prompt,
    build_react_system_prompt,
)
from services.tools import allowed_tool_names, get_tool_definitions
from services.tool_loop import run_tool_round

load_dotenv(Path(__file__).parent.parent / ".env")

router = APIRouter()
logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))

MAX_AUTONOMOUS_ROUNDS = 8
PLAN_SKIP_QUERY_LEN = 80               # 短于此长度的 query 跳过 plan 阶段
SESSION_TTL_SEC = 3600                 # 1 小时未续跑的 conversation 视为过期
SESSION_MAX_COUNT = 200                # 最多保 200 个 session（FIFO 淘汰）

# 业务工具白名单单一数据源：从 ToolRegistry 派生（D4），不再硬编码。
# run_tool_round 内部也用 allowed_tool_names()，端点侧只在 state summary 里引用。


# 控制工具不进 ToolRegistry，由 ReAct loop 本地处理。
_CONTROL_TOOLS = build_control_tools(
    ask_user_resume_hint="调用后循环会暂停，等用户回答后通过 /agent/autonomous/continue 续跑。"
)
_REACT_SYSTEM = build_react_system_prompt()


_PLAN_SYSTEM = (
    "你是 ReAct Agent 的规划助手。任务：把用户的学习目标拆解为 2-5 个可执行步骤。\n\n"
    "输出格式严格按以下编号列表：\n"
    "1. <第一步描述>\n2. <第二步描述>\n...\n"
    "不要输出任何解释，只输出编号列表。"
)


# ═══════════════════════════════════════════════════════════════════════════
# Session State（HITL 中断后续跑用）
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class AutonomousSession:
    """暂停在 ask_user 的会话状态，等用户回答后从这里恢复。"""
    conversation_id: str
    messages: list                                  # OpenAI messages
    plan: list[str]
    steps: list["StepRecord"]
    tools_called: list[str]
    rounds_used: int
    user_id: str
    document_id: Optional[str]
    evidence_registry: EvidenceRegistry
    grounding_required: bool
    pending_ask_call_id: str                        # 待回答的 ask_user 工具 call_id
    created_at: float = field(default_factory=time.time)


_sessions: dict[str, AutonomousSession] = {}        # 内存 session 表（demo 阶段够，生产换 Redis）


def _save_session(s: AutonomousSession) -> None:
    """FIFO + TTL 清理后写入。"""
    _purge_expired_sessions()
    if len(_sessions) >= SESSION_MAX_COUNT:
        # FIFO 淘汰最旧
        oldest = min(_sessions.values(), key=lambda x: x.created_at)
        _sessions.pop(oldest.conversation_id, None)
    _sessions[s.conversation_id] = s


def _purge_expired_sessions() -> None:
    now = time.time()
    expired = [cid for cid, s in _sessions.items() if now - s.created_at > SESSION_TTL_SEC]
    for cid in expired:
        _sessions.pop(cid, None)


# ═══════════════════════════════════════════════════════════════════════════
# 请求 / 响应模型
# ═══════════════════════════════════════════════════════════════════════════
class AutonomousRequest(BaseModel):
    query: str = Field(..., description="用户自然语言学习目标")
    user_id: str = Field(default="default", description="用户 ID")
    document_id: Optional[str] = Field(default=None, description="可选文档 ID")
    grounding_required: bool = Field(
        default=False,
        description="为 true 时，文档回答必须返回本轮检索得到的有效 chunk ID，否则安全弃答。",
    )


class ContinueRequest(BaseModel):
    conversation_id: str = Field(..., description="ask_user 时返回的 conversation_id")
    user_reply: str = Field(..., description="用户对 ask_user 问题的回答")


class StepRecord(BaseModel):
    """单步执行记录，Trajectory Eval 可消费"""
    round_index: int
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    observation_preview: Optional[str] = None
    blocked_reason: Optional[str] = None


class AutonomousResponse(BaseModel):
    plan: list[str] = Field(default_factory=list, description="可选的 plan 步骤列表（短 query 时为空）")
    steps: list[StepRecord] = Field(default_factory=list, description="实际执行的步骤序列")
    final_answer: str = Field(default="", description="最终面向用户的回复（finalize 触发时填）")
    rounds_used: int = Field(default=0, description="实际执行轮数")
    truncated: bool = Field(default=False, description="是否触发 max_rounds 强制收尾")
    tools_called: list[str] = Field(default_factory=list, description="所有被调用的工具名（去重）")

    # HITL 字段（P2 新增）
    awaiting_user_input: bool = Field(default=False, description="是否在等用户回答 ask_user")
    user_question: Optional[str] = Field(default=None, description="ask_user 的具体问题")
    conversation_id: Optional[str] = Field(default=None, description="续跑用 ID")

    # 范式标记（P2 新增）
    finalize_reason: Optional[str] = Field(default=None, description="LLM 调用 finalize 时给的结束理由")

    # 可验证引用字段（向后兼容：旧客户端可忽略）
    citations: list[CitationView] = Field(default_factory=list)
    invalid_citation_ids: list[str] = Field(default_factory=list)
    abstained: bool = Field(default=False, description="是否因证据不足而安全弃答")
    grounding_status: GroundingStatus = Field(default=GroundingStatus.NOT_REQUESTED)


# ═══════════════════════════════════════════════════════════════════════════
# State Summary（每轮注入给 LLM 防健忘）
# ═══════════════════════════════════════════════════════════════════════════
def _build_state_summary(
    steps: list[StepRecord],
    tools_called: list[str],
    plan: list[str],
    round_idx: int,
    evidence_registry: EvidenceRegistry,
) -> str:
    """浓缩当前进度，让 LLM 一眼看清。"""
    tools_used_unique = list(dict.fromkeys(tools_called))
    tools_remaining = [t for t in (allowed_tool_names() - set(tools_used_unique))]
    parts = [
        f"轮次：{round_idx + 1}/{MAX_AUTONOMOUS_ROUNDS}",
        f"已调用工具（按顺序）：{tools_used_unique or '无'}",
    ]
    if plan:
        parts.append(f"原始 plan ({len(plan)} 步)：{'; '.join(plan)}")
    # 最近 2 步的 observation 摘要
    recent = steps[-2:] if steps else []
    if recent:
        parts.append("最近 observation：")
        for s in recent:
            preview = (s.observation_preview or "")[:120]
            parts.append(f"  · {s.tool_name}({s.tool_args}) → {preview}")
    if evidence_registry:
        parts.append(f"本轮可引用 chunk_ids：{list(evidence_registry)[-8:]}")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Plan 阶段（可选）
# ═══════════════════════════════════════════════════════════════════════════
def _parse_plan(plan_text: str) -> list[str]:
    steps: list[str] = []
    for line in plan_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0].isdigit():
            parts = stripped.split(".", 1) if "." in stripped[:4] else stripped.split(")", 1)
            if len(parts) == 2:
                steps.append(parts[1].strip())
        elif stripped.startswith(("-", "*", "•")):
            steps.append(stripped.lstrip("-*• ").strip())
    return steps[:6]


async def _generate_plan(query: str, context_hint: str) -> list[str]:
    try:
        resp = await llm_chat(
            [
                {"role": "system", "content": _PLAN_SYSTEM + context_hint},
                {"role": "user", "content": query},
            ],
            client=_client,
        )
        return _parse_plan(resp.choices[0].message.content or "")
    except Exception as e:
        logger.warning(f"[autonomous] plan generation failed: {e}, skip plan")
        return []


# ═══════════════════════════════════════════════════════════════════════════
# 核心：ReAct Loop（拆出来供主入口和 continue 端点复用）
# ═══════════════════════════════════════════════════════════════════════════
async def _run_react_loop(
    messages: list,
    plan: list[str],
    steps: list[StepRecord],
    tools_called: list[str],
    user_id: str,
    document_id: Optional[str],
    starting_round: int,
    run_id: str,
    evidence_registry: EvidenceRegistry,
    grounding_required: bool,
) -> AutonomousResponse:
    """从 starting_round 开始跑 ReAct 循环。命中 finalize / ask_user / max_rounds 时返回。"""
    truncated = False
    final_answer = ""
    finalize_reason: Optional[str] = None

    for round_idx in range(starting_round, MAX_AUTONOMOUS_ROUNDS):
        # ── 注入 [Current state] 让 LLM 不健忘（仅本次调用，不持久化）──
        state_summary = _build_state_summary(
            steps, tools_called, plan, round_idx, evidence_registry
        )
        state_msg = [
            {"role": "system", "content": build_react_decision_prompt(state_summary)}
        ]

        # ── 单轮：复用 run_tool_round（D4）。控制工具交回本函数处理。──
        try:
            rr = await run_tool_round(
                messages,
                tools=get_tool_definitions() + _CONTROL_TOOLS,
                client=_client,
                control_tools=CONTROL_TOOL_NAMES,
                run_id=run_id,
                user_id=user_id,
                tool_choice="auto",
                extra_call_messages=state_msg,
            )
        except Exception as e:
            logger.exception(f"[autonomous] LLM call failed round {round_idx}: {e}")
            steps.append(StepRecord(round_index=round_idx, blocked_reason=f"LLM 调用失败: {e}"))
            break

        # ── 无 tool_calls：LLM 直接给文字（视为隐式 finalize）──
        if not rr.has_tool_calls:
            final_answer = rr.content or ""
            if final_answer:
                is_leak, leak_reason = check_output_leak(final_answer)
                if is_leak:
                    logger.warning(f"[autonomous] output leak blocked: {leak_reason}")
                    final_answer = "输出包含敏感信息已拦截。"
                finalize_reason = "implicit_finalize_no_tool_calls"
            else:
                final_answer = "（LLM 未给出回复且未调用工具，循环结束）"
                finalize_reason = "empty_response"
            return _build_response(
                plan, steps, tools_called, final_answer,
                round_idx + 1, truncated, finalize_reason,
                evidence_registry=evidence_registry,
                grounding_required=grounding_required,
            )

        # ── 逐个处理本轮 outcomes（顺序与 LLM 给的 tool_calls 一致）──
        for oc in rr.outcomes:
            fn_name, fn_args = oc.name, oc.arguments

            # ── 控制工具：finalize ──
            if fn_name == "finalize":
                final_answer = fn_args.get("final_answer", "")
                finalize_reason = fn_args.get("reason", "explicit_finalize")
                citation_ids = fn_args.get("citation_ids", [])
                if not isinstance(citation_ids, list):
                    citation_ids = []
                abstained = bool(fn_args.get("abstained", False))
                is_leak, leak_reason = check_output_leak(final_answer)
                if is_leak:
                    logger.warning(f"[autonomous] output leak in finalize: {leak_reason}")
                    final_answer = "输出包含敏感信息已拦截。"
                # 补一个 tool message 让 messages 完整（OpenAI 协议要求 tool_call 都有 response）
                messages.append({"role": "tool", "tool_call_id": oc.call_id, "content": "Acknowledged."})
                steps.append(StepRecord(
                    round_index=round_idx, tool_name="finalize", tool_args=fn_args,
                    observation_preview="(loop ended)",
                ))
                return _build_response(
                    plan, steps, tools_called, final_answer,
                    round_idx + 1, truncated, finalize_reason,
                    evidence_registry=evidence_registry,
                    grounding_required=grounding_required,
                    citation_ids=citation_ids,
                    abstained=abstained,
                )

            # ── 控制工具：ask_user → 保存 session 并返回 ──
            if fn_name == "ask_user":
                question = fn_args.get("question", "请提供更多信息。")
                conversation_id = f"conv_{uuid.uuid4().hex[:16]}"
                # 注意：messages 已含 assistant message（含 ask_user 的 tool_call）
                # 续跑时 user_reply 会作为该 tool_call 的 tool response 填回去
                _save_session(AutonomousSession(
                    conversation_id=conversation_id,
                    messages=messages,
                    plan=plan,
                    steps=steps,
                    tools_called=tools_called,
                    rounds_used=round_idx + 1,
                    user_id=user_id,
                    document_id=document_id,
                    evidence_registry=evidence_registry,
                    grounding_required=grounding_required,
                    pending_ask_call_id=oc.call_id,
                ))
                steps.append(StepRecord(
                    round_index=round_idx, tool_name="ask_user", tool_args=fn_args,
                    observation_preview="(awaiting user reply)",
                ))
                logger.info(f"[autonomous] ask_user paused: cid={conversation_id} q={question[:50]}")
                return AutonomousResponse(
                    plan=plan, steps=steps, tools_called=list(dict.fromkeys(tools_called)),
                    rounds_used=round_idx + 1, truncated=False,
                    awaiting_user_input=True, user_question=question,
                    conversation_id=conversation_id,
                    grounding_status=(
                        GroundingStatus.PENDING
                        if grounding_required
                        else GroundingStatus.NOT_REQUESTED
                    ),
                )

            # ── 业务工具被白名单拦截（run_tool_round 已回灌错误 message）──
            if oc.kind == "blocked":
                steps.append(StepRecord(
                    round_index=round_idx, tool_name=fn_name, tool_args=fn_args,
                    blocked_reason=oc.blocked_reason,
                ))
                continue

            # ── 业务工具已 dispatch（run_tool_round 已回灌 tool message）──
            tools_called.append(fn_name)
            observation_preview = (oc.result or "")[:300]
            evidence_error = None
            if fn_name == "search_document":
                accepted = collect_search_evidence(oc.result or "", evidence_registry)
                if accepted == 0:
                    evidence_error = "citation_payload_invalid_or_empty"
            steps.append(StepRecord(
                round_index=round_idx, tool_name=fn_name, tool_args=fn_args,
                observation_preview=observation_preview,
                blocked_reason=oc.blocked_reason or evidence_error,
            ))

    # ── 达 max_rounds：强制收尾 ──
    truncated = True
    logger.warning(f"[autonomous] truncated at {MAX_AUTONOMOUS_ROUNDS} rounds")
    messages.append({
        "role": "user",
        "content": "已达到最大执行轮次。请基于已有 observation 给出最终回答（直接文字，无需调工具）。",
    })
    try:
        finish_resp = await llm_chat(messages, client=_client)
        final_answer = finish_resp.choices[0].message.content or "执行被截断"
    except Exception as e:
        logger.exception(f"[autonomous] finish call failed: {e}")
        final_answer = f"执行被截断（{MAX_AUTONOMOUS_ROUNDS} 轮）"

    return _build_response(
        plan, steps, tools_called, final_answer,
        MAX_AUTONOMOUS_ROUNDS, truncated, "max_rounds_truncated",
        evidence_registry=evidence_registry,
        grounding_required=grounding_required,
    )


def _build_response(
    plan, steps, tools_called, final_answer, rounds_used, truncated, finalize_reason,
    *,
    evidence_registry: EvidenceRegistry,
    grounding_required: bool,
    citation_ids: list[str] | None = None,
    abstained: bool = False,
) -> AutonomousResponse:
    resolution = resolve_citations(citation_ids, evidence_registry)
    if grounding_required and not abstained and not resolution.citations:
        final_answer = "现有检索证据不足，无法提供带可验证引用的回答。"
        finalize_reason = "grounding_required_without_valid_citation"
        abstained = True

    if abstained:
        grounding_status = GroundingStatus.ABSTAINED
    elif resolution.citations:
        grounding_status = GroundingStatus.CITATION_IDS_VALID
    else:
        grounding_status = GroundingStatus.NOT_REQUESTED

    return AutonomousResponse(
        plan=plan, steps=steps,
        tools_called=list(dict.fromkeys(tools_called)),
        rounds_used=rounds_used, truncated=truncated,
        final_answer=final_answer, finalize_reason=finalize_reason,
        citations=resolution.citations,
        invalid_citation_ids=resolution.invalid_ids,
        abstained=abstained,
        grounding_status=grounding_status,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════
@router.post("/agent/autonomous", response_model=AutonomousResponse)
async def autonomous_agent(req: AutonomousRequest) -> AutonomousResponse:
    """Autonomous ReAct Agent 端点（P2 升级版）。

    范式：真 ReAct（LLM 全权决策）
    可选 plan：短 query 跳过，长 query 生成 plan 作为 hint
    HITL：LLM 可调 ask_user 触发两段式协议
    """
    # 第 1 层：Prompt Injection 检测
    is_injection, reason = await check_injection(req.query)
    if is_injection:
        logger.warning(f"[autonomous] injection blocked: {reason}")
        return AutonomousResponse(final_answer=f"输入安全检查未通过：{reason}")

    context_hint = f"\n\n当前用户 ID: {req.user_id}"
    if req.document_id:
        context_hint += f"\n当前文档 ID: {req.document_id}"
    if req.grounding_required:
        context_hint += (
            "\n本次请求要求可验证引用：必须先调用 search_document，并在 finalize 的 "
            "citation_ids 中仅填写 observation 返回的真实 chunk_ids；证据不足时设置 abstained=true。"
        )

    # Plan 可选：短 query 跳过
    if len(req.query.strip()) < PLAN_SKIP_QUERY_LEN:
        plan: list[str] = []
        logger.info(f"[autonomous] skip plan (query len={len(req.query.strip())})")
    else:
        plan = await _generate_plan(req.query, context_hint)
        logger.info(f"[autonomous] plan ({len(plan)} steps): {plan}")

    # 构造初始 messages（去掉旧版的"强制调工具" coercion）
    messages: list = [
        {"role": "system", "content": _REACT_SYSTEM + context_hint},
        {"role": "user", "content": req.query},
    ]
    if plan:
        plan_summary = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
        messages.append({
            "role": "system",
            "content": f"参考 plan（hint，非强制）：\n{plan_summary}",
        })

    run_id = f"auto_{uuid.uuid4().hex[:12]}"        # 关联 audit log

    return await _run_react_loop(
        messages=messages, plan=plan, steps=[], tools_called=[],
        user_id=req.user_id, document_id=req.document_id,
        starting_round=0, run_id=run_id,
        evidence_registry={}, grounding_required=req.grounding_required,
    )


@router.post("/agent/autonomous/continue", response_model=AutonomousResponse)
async def continue_autonomous(req: ContinueRequest) -> AutonomousResponse:
    """ask_user 后用户回答的续跑端点。

    协议：把 user_reply 作为 ask_user 的 tool response 填回 messages，从下一轮继续 ReAct。
    """
    session = _sessions.get(req.conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail="conversation 不存在或已过期")

    # 注入检测
    is_injection, reason = await check_injection(req.user_reply)
    if is_injection:
        return AutonomousResponse(
            final_answer=f"用户回答安全检查未通过：{reason}",
            conversation_id=req.conversation_id,
        )

    # 把 user_reply 作为 ask_user 的 tool response
    session.messages.append({
        "role": "tool",
        "tool_call_id": session.pending_ask_call_id,
        "content": f"User replied: {req.user_reply}",
    })
    # 弹出 session（用过即销毁；如果再次 ask_user 会重新写入）
    _sessions.pop(req.conversation_id, None)

    run_id = f"auto_cont_{uuid.uuid4().hex[:12]}"
    return await _run_react_loop(
        messages=session.messages, plan=session.plan,
        steps=session.steps, tools_called=session.tools_called,
        user_id=session.user_id, document_id=session.document_id,
        starting_round=session.rounds_used, run_id=run_id,
        evidence_registry=session.evidence_registry,
        grounding_required=session.grounding_required,
    )
