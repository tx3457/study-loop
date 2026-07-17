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
3. Web HITL：共享数据库保存暂停点，支持重启与多 worker，替代文件 IPC
4. State summary injection：context engineering 让 LLM 不健忘
5. 复用 P1-2 的 ToolRegistry，dispatch_tool 自带超时 / 重试 / audit
"""
import logging
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from models.citation import CitationView, GroundingStatus
from services.citations import (
    EvidenceChunk,
    EvidenceRegistry,
    collect_search_evidence,
    resolve_citations,
)
from services.autonomous_sessions import (
    AutonomousSessionStore,
    SessionCapacityError,
    SessionPayloadTooLargeError,
)
from services.injection import check_injection, check_output_leak
from services.idempotency import normalize_idempotency_key, request_idempotency
from services.llm import _client as _client, llm_chat
from services.react_controls import (
    CONTROL_TOOL_NAMES,
    build_control_tools,
    build_react_decision_prompt,
    build_react_system_prompt,
)
from services.tools import get_tool_definitions
from services.tool_registry import SideEffectAmbiguousError, tool_registry
from services.tool_loop import run_tool_round

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_AUTONOMOUS_ROUNDS = 8
PLAN_SKIP_QUERY_LEN = 80               # 短于此长度的 query 跳过 plan 阶段

# 业务工具白名单由 run_tool_round 从 ToolRegistry 派生，不在路由中硬编码。


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


autonomous_sessions = AutonomousSessionStore.from_environment()


# ═══════════════════════════════════════════════════════════════════════════
# 请求 / 响应模型
# ═══════════════════════════════════════════════════════════════════════════
class AutonomousRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, max_length=8000, description="用户自然语言学习目标"
    )
    user_id: str = Field(
        default="default", min_length=1, max_length=128, description="用户 ID"
    )
    document_id: Optional[str] = Field(
        default=None, max_length=512, description="可选文档 ID"
    )
    grounding_required: bool = Field(
        default=False,
        description="为 true 时，文档回答必须返回本轮检索得到的有效 chunk ID，否则安全弃答。",
    )


class ContinueRequest(BaseModel):
    conversation_id: str = Field(
        ..., min_length=1, max_length=128,
        description="ask_user 时返回的 conversation_id",
    )
    user_reply: str = Field(
        ..., min_length=1, max_length=8000,
        description="用户对 ask_user 问题的回答",
    )


class StepRecord(BaseModel):
    """单步执行记录，Trajectory Eval 可消费"""
    round_index: int = Field(ge=0)
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    observation_preview: Optional[str] = None
    blocked_reason: Optional[str] = None


class _EvidenceChunkPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str
    rank: int = Field(ge=1)


class _AutonomousSessionPayload(BaseModel):
    """Versioned JSON boundary for durable pause snapshots."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    messages: list[dict[str, object]] = Field(min_length=3, max_length=256)
    plan: list[str] = Field(max_length=6)
    steps: list[StepRecord] = Field(min_length=1, max_length=128)
    tools_called: list[str] = Field(max_length=128)
    rounds_used: int = Field(ge=1, le=MAX_AUTONOMOUS_ROUNDS)
    user_id: str = Field(min_length=1, max_length=128)
    document_id: Optional[str] = Field(default=None, max_length=512)
    evidence_registry: dict[str, _EvidenceChunkPayload]
    grounding_required: bool
    pending_ask_call_id: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_pause_boundary(self):
        if (
            self.steps[-1].tool_name != "ask_user"
            or self.steps[-1].round_index != self.rounds_used - 1
            or any(step.round_index >= self.rounds_used for step in self.steps)
        ):
            raise ValueError("session trajectory does not end at the pending ask_user")

        if not any(message.get("role") == "system" for message in self.messages) or not any(
            message.get("role") == "user" for message in self.messages
        ):
            raise ValueError("session messages must retain system and user context")

        for message in self.messages:
            if message.get("role") != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if (
                    tool_call.get("id") == self.pending_ask_call_id
                    and isinstance(function, dict)
                    and function.get("name") == "ask_user"
                    and isinstance(function.get("arguments"), str)
                ):
                    return self
        raise ValueError("pending ask_user tool call is missing from session messages")


def _session_to_payload(session: AutonomousSession) -> dict:
    payload = _AutonomousSessionPayload(
        schema_version=1,
        messages=session.messages,
        plan=session.plan,
        steps=session.steps,
        tools_called=session.tools_called,
        rounds_used=session.rounds_used,
        user_id=session.user_id,
        document_id=session.document_id,
        evidence_registry={
            chunk_id: _EvidenceChunkPayload(
                chunk_id=evidence.chunk_id,
                document_id=evidence.document_id,
                text=evidence.text,
                rank=evidence.rank,
            )
            for chunk_id, evidence in session.evidence_registry.items()
        },
        grounding_required=session.grounding_required,
        pending_ask_call_id=session.pending_ask_call_id,
    )
    return payload.model_dump(mode="json")


def _session_from_payload(
    conversation_id: str, payload: dict
) -> AutonomousSession:
    snapshot = _AutonomousSessionPayload.model_validate(payload)
    evidence_registry: EvidenceRegistry = {}
    for chunk_id, value in snapshot.evidence_registry.items():
        if chunk_id != value.chunk_id:
            raise ValueError("evidence registry key does not match chunk_id")
        evidence_registry[chunk_id] = EvidenceChunk(**value.model_dump())
    return AutonomousSession(
        conversation_id=conversation_id,
        messages=snapshot.messages,
        plan=snapshot.plan,
        steps=snapshot.steps,
        tools_called=snapshot.tools_called,
        rounds_used=snapshot.rounds_used,
        user_id=snapshot.user_id,
        document_id=snapshot.document_id,
        evidence_registry=evidence_registry,
        grounding_required=snapshot.grounding_required,
        pending_ask_call_id=snapshot.pending_ask_call_id,
    )


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


async def _validate_replayed_response(response_payload: dict) -> AutonomousResponse:
    """Reject cached pause responses whose resumable snapshot is no longer live."""
    response = AutonomousResponse.model_validate(response_payload)
    if not response.awaiting_user_input or not response.conversation_id:
        return response
    session_status = await autonomous_sessions.status(response.conversation_id)
    if session_status is None:
        raise HTTPException(
            status_code=410,
            detail="暂停会话已过期；请使用新的 Idempotency-Key 重新开始",
        )
    if session_status == "in_flight":
        raise HTTPException(status_code=409, detail="conversation 正在续跑，请稍后重试")
    return response


async def _abort_receipt(key: Optional[str]) -> bool:
    return await request_idempotency.abort(key) if key else False


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
    idempotency_key: Optional[str] = None,
    on_before_tool_calls: Callable[[], Awaitable[None]] | None = None,
    pause_session_saver: Callable[[AutonomousSession], Awaitable[None]] | None = None,
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
        rr = await run_tool_round(
            messages,
            tools=get_tool_definitions() + _CONTROL_TOOLS,
            client=_client,
            control_tools=CONTROL_TOOL_NAMES,
            run_id=run_id,
            user_id=user_id,
            idempotency_key=idempotency_key,
            tool_choice="auto",
            extra_call_messages=state_msg,
            on_before_tool_calls=on_before_tool_calls,
        )

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
            if oc.kind == "control" and fn_name == "finalize":
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
            if oc.kind == "control" and fn_name == "ask_user":
                question = fn_args.get("question", "请提供更多信息。")
                conversation_id = f"conv_{uuid.uuid4().hex}"
                # 注意：messages 已含 assistant message（含 ask_user 的 tool_call）
                # 续跑时 user_reply 会作为该 tool_call 的 tool response 填回去
                steps.append(StepRecord(
                    round_index=round_idx, tool_name="ask_user", tool_args=fn_args,
                    observation_preview="(awaiting user reply)",
                ))
                session = AutonomousSession(
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
                )
                try:
                    if pause_session_saver is None:
                        await autonomous_sessions.save(
                            conversation_id, _session_to_payload(session)
                        )
                    else:
                        await pause_session_saver(session)
                except SessionCapacityError as exc:
                    raise HTTPException(
                        status_code=503,
                        detail="暂停会话容量已满，请稍后重试",
                    ) from exc
                except SessionPayloadTooLargeError as exc:
                    raise HTTPException(
                        status_code=413,
                        detail="暂停会话数据过大，无法继续保存",
                    ) from exc
                logger.info(
                    "[autonomous] ask_user paused: cid_prefix=%s",
                    conversation_id[:13],
                )
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
async def _execute_autonomous(
    req: AutonomousRequest,
    *,
    run_id: str,
    idempotency_key: Optional[str],
) -> AutonomousResponse:
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

    return await _run_react_loop(
        messages=messages, plan=plan, steps=[], tools_called=[],
        user_id=req.user_id, document_id=req.document_id,
        starting_round=0, run_id=run_id,
        evidence_registry={}, grounding_required=req.grounding_required,
        idempotency_key=idempotency_key,
    )


@router.post("/agent/autonomous", response_model=AutonomousResponse)
async def autonomous_agent(
    req: AutonomousRequest,
    idempotency_key: Optional[str] = Header(
        default=None, alias="Idempotency-Key"
    ),
) -> AutonomousResponse:
    """Run one request with an optional durable replay receipt."""
    key = normalize_idempotency_key(idempotency_key)
    if key:
        decision = await request_idempotency.begin(
            key, "agent.autonomous", req.model_dump(mode="json")
        )
        if decision.replayed:
            return await _validate_replayed_response(decision.response)

    run_id = f"auto_{uuid.uuid4().hex[:12]}"
    try:
        response = await _execute_autonomous(
            req, run_id=run_id, idempotency_key=key
        )
        if key:
            await request_idempotency.complete(
                key, response.model_dump(mode="json")
            )
        return response
    except BaseException as exc:
        durable_effect = await _abort_receipt(key)
        # 写工具成功后，下一轮 provider 仍可能失败。此时整个请求不可作为
        # 普通 503 自动重试；持久 receipt 不依赖可淘汰的进程内 audit。
        effect_attempted = durable_effect or tool_registry.has_effect_attempt(run_id)
        if (
            isinstance(exc, Exception)
            and effect_attempted
            and not isinstance(exc, SideEffectAmbiguousError)
        ):
            raise SideEffectAmbiguousError("autonomous_request") from exc
        raise


@router.post("/agent/autonomous/continue", response_model=AutonomousResponse)
async def continue_autonomous(
    req: ContinueRequest,
    idempotency_key: Optional[str] = Header(
        default=None, alias="Idempotency-Key"
    ),
) -> AutonomousResponse:
    """ask_user 后用户回答的续跑端点。

    协议：把 user_reply 作为 ask_user 的 tool response 填回 messages，从下一轮继续 ReAct。
    """
    key = normalize_idempotency_key(idempotency_key)
    if key:
        decision = await request_idempotency.begin(
            key, "agent.autonomous.continue", req.model_dump(mode="json")
        )
        if decision.replayed:
            return await _validate_replayed_response(decision.response)

    try:
        session_status = await autonomous_sessions.status(req.conversation_id)
    except BaseException:
        await _abort_receipt(key)
        raise
    if session_status is None:
        await _abort_receipt(key)
        raise HTTPException(status_code=404, detail="conversation 不存在或已过期")
    if session_status == "in_flight":
        await _abort_receipt(key)
        raise HTTPException(status_code=409, detail="conversation 正在续跑，请稍后重试")

    # 注入检测尚未改变 session 或执行工具；分类器失败时释放
    # receipt，让客户端可以用同一个 key 安全重试。
    try:
        is_injection, reason = await check_injection(req.user_reply)
    except BaseException:
        await _abort_receipt(key)
        raise
    if is_injection:
        response = AutonomousResponse(
            final_answer=f"用户回答安全检查未通过：{reason}",
            conversation_id=req.conversation_id,
        )
        if key:
            await request_idempotency.complete(
                key, response.model_dump(mode="json")
            )
        return response

    # 数据库 CAS 认领 session，跨进程也只允许一个续跑 owner。
    try:
        claim = await autonomous_sessions.claim(req.conversation_id)
    except BaseException:
        await _abort_receipt(key)
        raise
    if not claim.claimed:
        await _abort_receipt(key)
        if claim.reason == "in_progress":
            raise HTTPException(status_code=409, detail="conversation 正在续跑，请稍后重试")
        raise HTTPException(status_code=404, detail="conversation 不存在或已过期")
    if not claim.claim_token:
        raise RuntimeError("session claim returned without ownership data")

    claim_token = claim.claim_token
    if claim.payload is None:
        await autonomous_sessions.consume(req.conversation_id, claim_token)
        await _abort_receipt(key)
        logger.error(
            "[autonomous] unreadable persisted session: cid_prefix=%s reason=%s",
            req.conversation_id[:13],
            claim.reason,
        )
        raise HTTPException(
            status_code=410,
            detail="暂停会话数据无效，已安全终止；请重新开始",
        )

    try:
        session = _session_from_payload(req.conversation_id, claim.payload)
    except (TypeError, ValueError, ValidationError) as exc:
        await autonomous_sessions.consume(req.conversation_id, claim_token)
        await _abort_receipt(key)
        logger.exception(
            "[autonomous] invalid persisted session: cid_prefix=%s",
            req.conversation_id[:13],
        )
        raise HTTPException(
            status_code=410,
            detail="暂停会话数据无效，已安全终止；请重新开始",
        ) from exc

    # copy-on-resume：provider 临时失败时保留数据库中的原始 JSON 快照。
    resume_messages = list(session.messages)
    resume_steps = list(session.steps)
    resume_tools_called = list(session.tools_called)
    resume_evidence_registry = dict(session.evidence_registry)
    resume_messages.append({
        "role": "tool",
        "tool_call_id": session.pending_ask_call_id,
        "content": f"User replied: {req.user_reply}",
    })

    progress_started = False
    session_handed_off = False

    async def mark_progress_before_tool_calls() -> None:
        nonlocal progress_started
        if progress_started:
            return
        marked = await autonomous_sessions.mark_progress(
            req.conversation_id, claim_token
        )
        if not marked:
            raise RuntimeError("lost autonomous session claim before tool processing")
        progress_started = True

    async def handoff_to_next_pause(next_session: AutonomousSession) -> None:
        nonlocal session_handed_off
        handed_off = await autonomous_sessions.handoff(
            req.conversation_id,
            claim_token,
            next_session.conversation_id,
            _session_to_payload(next_session),
        )
        if not handed_off:
            raise RuntimeError("lost autonomous session claim during pause handoff")
        session_handed_off = True

    run_id = f"auto_cont_{uuid.uuid4().hex[:12]}"
    baseline = (
        len(resume_messages),
        len(resume_steps),
        len(resume_tools_called),
        len(resume_evidence_registry),
    )
    session_consumed = False
    try:
        response = await _run_react_loop(
            messages=resume_messages, plan=session.plan,
            steps=resume_steps, tools_called=resume_tools_called,
            user_id=session.user_id, document_id=session.document_id,
            starting_round=session.rounds_used, run_id=run_id,
            evidence_registry=resume_evidence_registry,
            grounding_required=session.grounding_required,
            idempotency_key=key,
            on_before_tool_calls=mark_progress_before_tool_calls,
            pause_session_saver=handoff_to_next_pause,
        )
        if not session_handed_off:
            if not await autonomous_sessions.consume(
                req.conversation_id, claim_token
            ):
                raise RuntimeError("lost autonomous session claim during completion")
        session_consumed = True
        if key:
            await request_idempotency.complete(
                key, response.model_dump(mode="json")
            )
        return response
    except BaseException as exc:
        trajectory_progressed = baseline != (
            len(resume_messages),
            len(resume_steps),
            len(resume_tools_called),
            len(resume_evidence_registry),
        )
        # Registry 在非幂等/未知 handler 前写 started marker。即使工具已开始而
        # trajectory 尚未来得及 append，也必须 fail closed，避免重放副作用。
        durable_effect = await _abort_receipt(key)
        effect_attempted = (
            durable_effect or tool_registry.has_effect_attempt(run_id)
        )
        progressed = progress_started or trajectory_progressed or effect_attempted

        if not session_consumed and not progressed:
            released = await autonomous_sessions.release(
                req.conversation_id, claim_token
            )
            if released:
                raise

        if not session_consumed:
            await autonomous_sessions.consume(req.conversation_id, claim_token)

        logger.warning(
            "[autonomous] continuation terminated fail-closed: cid_prefix=%s progressed=%s",
            req.conversation_id[:13],
            progressed,
        )
        if isinstance(exc, Exception):
            raise HTTPException(
                status_code=410,
                detail=(
                    "续跑已执行部分操作，无法安全重试；请重新开始"
                    if progressed
                    else "暂停会话已过期或状态已变化；请重新开始"
                ),
            ) from exc
        raise
