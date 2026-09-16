"""
Autonomous Agent 端点：ReAct + HITL

── 设计 ────────────────────────────────────────────────────────────────────
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

"""
import asyncio
import hashlib
import json
import logging
import re
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, Optional

from fastapi import Depends, APIRouter, Header, HTTPException, Path as ApiPath
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from models.citation import CitationView, GroundingStatus
from services.auth import require_user_id
from services.citations import (
    CitationResolution,
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
from services.idempotency import (
    IdempotencyConflictError,
    ReceiptLease,
    abort_idempotency_claim,
    normalize_idempotency_key,
    request_idempotency,
)
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
from services.tool_scope import build_business_tool_scope_guard

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_AUTONOMOUS_ROUNDS = 8
_SESSION_SCHEMA_VERSION = 3
_RESPONSE_SCHEMA_VERSION = 2
PLAN_SKIP_QUERY_LEN = 80               # 短于此长度的 query 跳过 plan 阶段
PLAN_TOTAL_TIMEOUT_SECONDS = 20.0
PLAN_TRANSPORT_RETRIES = 1
PLAN_SEMANTIC_ATTEMPTS = 2
_GROUNDING_ABSTENTION = "现有检索证据不足，无法提供满足完整引用约束的回答。"

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
    registry_sha256: str = ""
    semantic_reservation_digests: tuple[tuple[str, str], ...] = ()


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

    @model_validator(mode="after")
    def require_document_for_strict_grounding(self):
        if not self.query.strip():
            raise ValueError("query must not be blank")
        if not self.user_id.strip():
            raise ValueError("user_id must not be blank")
        if self.document_id is not None and not self.document_id.strip():
            raise ValueError("document_id must not be blank when supplied")
        if self.grounding_required and (
            self.document_id is None or not self.document_id.strip()
        ):
            raise ValueError("grounding_required requires a document_id")
        return self


class ContinueRequest(BaseModel):
    conversation_id: str = Field(
        ..., min_length=1, max_length=128,
        description="ask_user 时返回的 conversation_id",
    )
    user_reply: str = Field(
        ..., min_length=1, max_length=8000,
        description="用户对 ask_user 问题的回答",
    )

    @model_validator(mode="after")
    def require_nonblank_values(self):
        if not self.conversation_id.strip():
            raise ValueError("conversation_id must not be blank")
        if not self.user_reply.strip():
            raise ValueError("user_reply must not be blank")
        return self


class StepRecord(BaseModel):
    """单步执行记录，Trajectory Eval 可消费"""
    round_index: int = Field(ge=0)
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    observation_preview: Optional[str] = None
    blocked_reason: Optional[str] = None


_PUBLIC_REDACTION = "[敏感内容已隐藏]"


def _sanitize_public_value(value):
    """Recursively redact sensitive strings before trajectory data is public."""
    if isinstance(value, str):
        leaked, _ = check_output_leak(value)
        return _PUBLIC_REDACTION if leaked else value
    if isinstance(value, dict):
        return {
            _sanitize_public_value(key): _sanitize_public_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_public_value(item) for item in value]
    return value


def _public_steps(
    steps: list[StepRecord],
    *,
    redact_details: bool = False,
) -> list[StepRecord]:
    """Return a public trajectory with unchecked answers and secrets removed."""
    public: list[StepRecord] = []
    for step in steps:
        tool_name = step.tool_name
        if (
            tool_name is not None
            and tool_name not in CONTROL_TOOL_NAMES
            and tool_registry.get(tool_name) is None
        ):
            tool_name = "blocked_tool"
        tool_args = step.tool_args
        if redact_details or step.tool_name in CONTROL_TOOL_NAMES:
            tool_args = None
        elif tool_args is not None:
            tool_args = {
                key: _sanitize_public_value(value)
                for key, value in tool_args.items()
            }
        public.append(step.model_copy(update={
            "tool_name": tool_name,
            "tool_args": tool_args,
            "observation_preview": (
                None
                if redact_details
                else _sanitize_public_value(step.observation_preview)
            ),
        }))
    return public


def _public_plan(plan: list[str]) -> list[str]:
    return [_sanitize_public_value(item) for item in plan]


def _safe_snapshot_plan(plan: list[str]) -> list[str]:
    """Keep model-authored plan text out of durable state when it trips DLP."""

    return [] if check_output_leak("\n".join(plan))[0] else list(plan)


class _FinalizeArgs(BaseModel):
    """Strict runtime boundary for model-generated finalize arguments."""

    model_config = ConfigDict(extra="forbid", strict=True)

    final_answer: str = Field(max_length=20_000)
    reason: str = Field(default="explicit_finalize", max_length=1_000)
    citation_ids: list[str] = Field(default_factory=list, max_length=50)
    abstained: bool = False

    @model_validator(mode="after")
    def validate_citation_ids(self):
        if not self.final_answer.strip():
            raise ValueError("final_answer must not be blank")
        if any(
            not chunk_id.strip() or len(chunk_id) > 1_024
            for chunk_id in self.citation_ids
        ):
            raise ValueError("citation IDs must contain 1 to 1024 characters")
        return self


class _AskUserArgs(BaseModel):
    """Strict runtime boundary for model-generated HITL questions."""

    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def validate_question(self):
        if not self.question.strip():
            raise ValueError("question must not be blank")
        return self


def _pending_ask_question(
    messages: list[dict[str, object]], call_id: str
) -> str | None:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict) or tool_call.get("id") != call_id:
                continue
            function = tool_call.get("function")
            if (
                not isinstance(function, dict)
                or function.get("name") != "ask_user"
                or not isinstance(function.get("arguments"), str)
            ):
                return None
            try:
                return _AskUserArgs.model_validate_json(
                    function["arguments"]
                ).question
            except ValidationError:
                return None
    return None


class _EvidenceChunkPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str
    rank: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_nonblank_strings(self):
        for field_name in ("chunk_id", "document_id", "text"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be blank")
        return self


class _AutonomousSessionPayload(BaseModel):
    """Versioned JSON boundary for durable pause snapshots."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1, 2, 3]
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
    registry_sha256: str = Field(default="", pattern=r"^[0-9a-f]{64}$|^$")
    semantic_reservation_digests: list[tuple[str, str]] = Field(
        default_factory=list,
        max_length=256,
    )

    @model_validator(mode="after")
    def validate_pause_boundary(self):
        if self.schema_version == 3 and not self.registry_sha256:
            raise ValueError("v3 session is missing its registry fingerprint")
        if self.schema_version in {1, 2} and (
            self.registry_sha256 or self.semantic_reservation_digests
        ):
            raise ValueError("legacy session contains unsupported security state")
        for tool_name, digest in self.semantic_reservation_digests:
            if not tool_name or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("semantic reservation digest is invalid")

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

        if self.grounding_required and (
            self.document_id is None or not self.document_id.strip()
        ):
            raise ValueError("strict grounding snapshot is missing its document scope")

        if self.document_id is not None and any(
            evidence.document_id != self.document_id
            for evidence in self.evidence_registry.values()
        ):
            raise ValueError("evidence registry escapes the persisted document scope")

        question = _pending_ask_question(
            self.messages, self.pending_ask_call_id
        )
        if question is None:
            raise ValueError(
                "pending ask_user tool call is missing or invalid in session messages"
            )
        if check_output_leak(question)[0]:
            raise ValueError(
                "pending ask_user question failed the safety boundary"
            )
        return self


def _current_registry_sha256() -> str:
    """Fingerprint schemas and security metadata governing business dispatch."""
    contracts = []
    for name in sorted(tool_registry.list_tools()):
        tool = tool_registry.get(name)
        if tool is None:
            continue
        contracts.append({
            "name": tool.name,
            "parameters_schema": tool.parameters_schema,
            "metadata": tool.metadata.security_contract_payload(),
        })
    canonical = json.dumps(
        contracts,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _has_completed_profile_update(steps: list[StepRecord]) -> bool:
    return any(
        step.tool_name == "update_learning_profile"
        and step.blocked_reason is None
        for step in steps
    )


def _validate_profile_update_reservations(
    steps: list[StepRecord],
    reservations: list[tuple[str, str]],
) -> None:
    """Bind every durable update receipt to its persisted effective arguments."""
    expected: set[tuple[str, str]] = set()
    for step in steps:
        if (
            step.tool_name != "update_learning_profile"
            or step.blocked_reason is not None
        ):
            continue
        if not isinstance(step.tool_args, dict):
            raise ValueError(
                "completed profile update lacks restorable arguments"
            )
        try:
            expected.add(tool_registry.semantic_reservation(
                "update_learning_profile",
                step.tool_args,
            ))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "completed profile update receipt cannot be reconstructed"
            ) from exc

    persisted = {
        reservation
        for reservation in reservations
        if reservation[0] == "update_learning_profile"
    }
    if persisted != expected:
        raise ValueError("profile update semantic receipt mismatch")


def _sanitize_snapshot_messages(
    raw_messages: list[dict[str, object]],
    pending_ask_call_id: str,
) -> tuple[list[dict[str, object]], str]:
    """Return the minimum safe tool-call history needed for HITL resume."""

    messages = deepcopy(raw_messages)
    redacted_ids: dict[str, str] = {}
    unknown_call_ids: set[str] = set()
    safe_pending_ask_call_id = pending_ask_call_id
    redacted_id_index = 0
    used_call_ids = {
        tool_call.get("id")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
        for tool_call in (
            message.get("tool_calls")
            if isinstance(message.get("tool_calls"), list)
            else []
        )
        if isinstance(tool_call, dict) and isinstance(tool_call.get("id"), str)
    }
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "system":
            content = message.get("content")
            if (
                isinstance(content, str)
                and content.startswith("参考 plan（hint，非强制）：")
                and check_output_leak(content)[0]
            ):
                message["content"] = "参考 plan 已因安全检查移除。"
            continue
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        message["content"] = None
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            raw_call_id = tool_call.get("id")
            if (
                isinstance(raw_call_id, str)
                and check_output_leak(raw_call_id)[0]
            ):
                safe_call_id = redacted_ids.get(raw_call_id)
                if safe_call_id is None:
                    while True:
                        redacted_id_index += 1
                        safe_call_id = f"redacted_call_{redacted_id_index}"
                        if safe_call_id not in used_call_ids:
                            break
                    redacted_ids[raw_call_id] = safe_call_id
                    used_call_ids.add(safe_call_id)
                tool_call["id"] = safe_call_id
                if raw_call_id == pending_ask_call_id:
                    safe_pending_ask_call_id = safe_call_id
            function = tool_call.get("function")
            function_name = (
                function.get("name") if isinstance(function, dict) else None
            )
            is_pending_ask = (
                function_name == "ask_user"
                and raw_call_id == pending_ask_call_id
            )
            if (
                isinstance(function, dict)
                and function_name not in CONTROL_TOOL_NAMES
                and tool_registry.get(function_name) is None
            ):
                if isinstance(raw_call_id, str):
                    unknown_call_ids.add(raw_call_id)
                function["name"] = "blocked_tool"
                function["arguments"] = "{}"
                continue
            if (
                isinstance(function, dict)
                and function_name in CONTROL_TOOL_NAMES
                and not is_pending_ask
            ):
                function["arguments"] = '{"redacted":true}'
                continue
            if isinstance(function, dict):
                arguments = function.get("arguments")
                if (
                    isinstance(arguments, str)
                    and check_output_leak(arguments)[0]
                ):
                    function["arguments"] = '{"redacted":true}'

    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        raw_call_id = message.get("tool_call_id")
        if isinstance(raw_call_id, str) and raw_call_id in redacted_ids:
            message["tool_call_id"] = redacted_ids[raw_call_id]
        content = message.get("content")
        if (
            isinstance(raw_call_id, str)
            and raw_call_id in unknown_call_ids
        ) or (isinstance(content, str) and check_output_leak(content)[0]):
            message["content"] = '{"error":"blocked_tool"}'
    return messages, safe_pending_ask_call_id


def _session_to_payload(session: AutonomousSession) -> dict:
    messages, pending_ask_call_id = _sanitize_snapshot_messages(
        session.messages, session.pending_ask_call_id
    )

    snapshot_steps = _public_steps(session.steps)
    _validate_profile_update_reservations(
        snapshot_steps,
        list(session.semantic_reservation_digests),
    )
    payload = _AutonomousSessionPayload(
        schema_version=_SESSION_SCHEMA_VERSION,
        messages=messages,
        plan=_safe_snapshot_plan(session.plan),
        steps=snapshot_steps,
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
        pending_ask_call_id=pending_ask_call_id,
        registry_sha256=(session.registry_sha256 or _current_registry_sha256()),
        semantic_reservation_digests=list(
            session.semantic_reservation_digests
        ),
    )
    return payload.model_dump(mode="json")


def _session_from_payload(
    conversation_id: str, payload: dict
) -> AutonomousSession:
    snapshot = _AutonomousSessionPayload.model_validate(payload)
    source_schema_version = snapshot.schema_version
    completed_profile_update = _has_completed_profile_update(snapshot.steps)
    if completed_profile_update and source_schema_version in {1, 2}:
        raise ValueError(
            "legacy session contains an update without a semantic receipt"
        )
    if source_schema_version == 3:
        _validate_profile_update_reservations(
            snapshot.steps,
            snapshot.semantic_reservation_digests,
        )
    # V1 predated at-rest control-message redaction. Reapply the same
    # sanitization to every restored version so a tampered V2 row cannot turn
    # historical control text back into provider input.
    migrated = snapshot.model_dump(mode="json")
    migrated["schema_version"] = _SESSION_SCHEMA_VERSION
    if source_schema_version in {1, 2}:
        migrated["registry_sha256"] = _current_registry_sha256()
        migrated["semantic_reservation_digests"] = []
    migrated["messages"], migrated["pending_ask_call_id"] = _sanitize_snapshot_messages(
        snapshot.messages, snapshot.pending_ask_call_id
    )
    migrated["steps"] = [
        step.model_dump(mode="json") for step in _public_steps(snapshot.steps)
    ]
    migrated["plan"] = _safe_snapshot_plan(snapshot.plan)
    snapshot = _AutonomousSessionPayload.model_validate(migrated)
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
        registry_sha256=snapshot.registry_sha256,
        semantic_reservation_digests=tuple(
            snapshot.semantic_reservation_digests
        ),
    )


class AutonomousResponse(BaseModel):
    response_schema_version: int = Field(
        default=_RESPONSE_SCHEMA_VERSION,
        ge=1,
        le=_RESPONSE_SCHEMA_VERSION,
    )
    plan: list[str] = Field(default_factory=list, description="可选的 plan 步骤列表（短 query 时为空）")
    steps: list[StepRecord] = Field(default_factory=list, description="实际执行的步骤序列")
    final_answer: str = Field(default="", description="最终面向用户的回复（finalize 触发时填）")
    rounds_used: int = Field(default=0, description="实际执行轮数")
    truncated: bool = Field(default=False, description="是否触发 max_rounds 强制收尾")
    tools_called: list[str] = Field(default_factory=list, description="所有被调用的工具名（去重）")

    # HITL 字段
    awaiting_user_input: bool = Field(default=False, description="是否在等用户回答 ask_user")
    user_question: Optional[str] = Field(default=None, description="ask_user 的具体问题")
    conversation_id: Optional[str] = Field(default=None, description="续跑用 ID")

    # 范式标记
    finalize_reason: Optional[str] = Field(default=None, description="LLM 调用 finalize 时给的结束理由")

    # 可验证引用字段（向后兼容：旧客户端可忽略）
    citations: list[CitationView] = Field(default_factory=list)
    invalid_citation_ids: list[str] = Field(
        default_factory=list,
        description="兼容旧客户端的保留字段；服务端不回显模型生成的无效 ID 原文",
    )
    invalid_citation_count: int = Field(default=0, ge=0)
    abstained: bool = Field(default=False, description="是否因证据不足或安全策略而拒答")
    grounding_status: GroundingStatus = Field(default=GroundingStatus.NOT_REQUESTED)
    grounding_required: bool = Field(default=False, description="本轮实际采用的引用约束")
    grounding_document_id: Optional[str] = Field(
        default=None,
        description="本轮工具的 document_id 参数被锁定到此值；为空表示未指定单文档范围",
    )


async def _validate_replayed_response(response_payload: dict) -> AutonomousResponse:
    """Validate durable responses against the current public state machine."""
    raw_schema_version = (
        response_payload.get("response_schema_version", 1)
        if isinstance(response_payload, dict)
        else 1
    )
    if (
        not isinstance(raw_schema_version, int)
        or isinstance(raw_schema_version, bool)
        or raw_schema_version not in {1, _RESPONSE_SCHEMA_VERSION}
    ):
        raise HTTPException(status_code=410, detail="缓存响应版本无效；请重新开始")
    try:
        response = AutonomousResponse.model_validate(response_payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=410,
            detail="缓存响应格式无效；请重新开始",
        ) from exc
    citation_payload_invalid = any(
        not citation.chunk_id.strip()
        or not citation.document_id.strip()
        or not citation.snippet.strip()
        for citation in response.citations
    )
    legacy_unverifiable_citations = (
        raw_schema_version == 1 and bool(response.citations)
    )
    citation_scope_missing = bool(response.citations) and not (
        response.grounding_document_id
        and response.grounding_document_id.strip()
    )
    scope_mismatch = any(
        citation.document_id != response.grounding_document_id
        for citation in response.citations
    )
    strict_scope_missing = (
        response.grounding_required
        and (
            not response.grounding_document_id
            or not response.grounding_document_id.strip()
        )
    )
    if (
        citation_payload_invalid
        or legacy_unverifiable_citations
        or citation_scope_missing
        or scope_mismatch
        or strict_scope_missing
    ):
        raise HTTPException(
            status_code=410,
            detail="缓存引用无法通过当前范围校验；请重新开始",
        )
    if response.user_question is not None and check_output_leak(
        response.user_question
    )[0]:
        raise HTTPException(
            status_code=410,
            detail="缓存的暂停问题未通过当前安全检查；请重新开始",
        )
    legacy_invalid_ids = list(response.invalid_citation_ids)
    updates = {
        "response_schema_version": _RESPONSE_SCHEMA_VERSION,
        "plan": [] if response.abstained else _public_plan(response.plan),
        "steps": _public_steps(response.steps, redact_details=response.abstained),
        "invalid_citation_count": max(
            response.invalid_citation_count,
            len(response.invalid_citation_ids),
        ),
        "invalid_citation_ids": [],
    }
    replay_leaked = check_output_leak(
        f"{response.final_answer}\n{response.finalize_reason or ''}"
    )[0] or any(
        check_output_leak(citation.model_dump_json())[0]
        for citation in response.citations
    )
    if replay_leaked:
        updates.update({
            "final_answer": "输出包含敏感信息已拦截。",
            "finalize_reason": "output_leak_blocked",
            "abstained": True,
            "grounding_status": GroundingStatus.ABSTAINED,
            "citations": [],
            "plan": [],
            "steps": _public_steps(response.steps, redact_details=True),
        })
    elif legacy_invalid_ids:
        # Older receipts allowed mixed valid/invalid IDs. They cannot be safely
        # replayed under the current strict contract because the unsupported
        # claim-to-ID relationship was never recorded.
        updates.update({
            "final_answer": _GROUNDING_ABSTENTION,
            "finalize_reason": "legacy_invalid_citation_replay",
            "abstained": True,
            "grounding_status": GroundingStatus.ABSTAINED,
            "citations": [],
            "plan": [],
            "steps": _public_steps(response.steps, redact_details=True),
        })
    elif response.abstained:
        safe_reasons = {
            "input_safety_blocked",
            "output_leak_blocked",
            "model_abstained",
            "grounding_required_with_invalid_citation",
            "grounding_required_without_valid_citation",
            "legacy_invalid_citation_replay",
        }
        fixed_answers = {
            "input_safety_blocked": "输入安全检查未通过。请修改目标后重试。",
            "output_leak_blocked": "输出包含敏感信息已拦截。",
        }
        updates.update({
            "final_answer": fixed_answers.get(
                response.finalize_reason, _GROUNDING_ABSTENTION
            ),
            "finalize_reason": (
                response.finalize_reason
                if response.finalize_reason in safe_reasons
                else "legacy_abstained"
            ),
            "citations": [],
            "grounding_status": GroundingStatus.ABSTAINED,
        })
    response = response.model_copy(update=updates)

    if response.awaiting_user_input:
        pending_contract_invalid = (
            response.user_question is None
            or not response.user_question.strip()
            or response.conversation_id is None
            or not response.conversation_id.strip()
            or bool(response.final_answer.strip())
            or response.finalize_reason is not None
            or response.abstained
            or bool(response.citations)
            or response.invalid_citation_count != 0
            or bool(response.invalid_citation_ids)
        )
        if pending_contract_invalid:
            raise HTTPException(
                status_code=410,
                detail="缓存的暂停响应状态不一致；请重新开始",
            )

        inspection = await autonomous_sessions.inspect(response.conversation_id)
        if inspection is None:
            raise HTTPException(
                status_code=410,
                detail="暂停会话已过期；请使用新的 Idempotency-Key 重新开始",
            )
        if inspection.state == "in_flight":
            raise HTTPException(
                status_code=409, detail="conversation 正在续跑，请稍后重试"
            )
        if inspection.state != "paused":
            raise HTTPException(
                status_code=410,
                detail="暂停会话状态无效；请重新开始",
            )
        try:
            session = _session_from_payload(
                response.conversation_id, inspection.payload
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise HTTPException(
                status_code=410,
                detail="暂停会话快照无效；请重新开始",
            ) from exc

        pending_question = _pending_ask_question(
            session.messages, session.pending_ask_call_id
        )
        if response.user_question != pending_question:
            raise HTTPException(
                status_code=410,
                detail="缓存的暂停问题与会话快照不一致；请重新开始",
            )

        expected_status = (
            GroundingStatus.PENDING
            if session.grounding_required
            else GroundingStatus.NOT_REQUESTED
        )
        if raw_schema_version == 1:
            response = response.model_copy(update={
                "grounding_required": session.grounding_required,
                "grounding_document_id": session.document_id,
                "grounding_status": expected_status,
            })
        elif (
            response.grounding_required != session.grounding_required
            or response.grounding_document_id != session.document_id
            or response.grounding_status != expected_status
        ):
            raise HTTPException(
                status_code=410,
                detail="缓存的暂停响应与会话范围不一致；请重新开始",
            )
        return response

    if (
        response.user_question is not None
        or response.conversation_id is not None
        or response.grounding_status == GroundingStatus.PENDING
    ):
        raise HTTPException(
            status_code=410,
            detail="缓存的终态响应包含暂停字段；请重新开始",
        )

    if response.grounding_required:
        grounded_terminal = (
            not response.abstained
            and bool(response.citations)
            and response.invalid_citation_count == 0
            and response.grounding_status == GroundingStatus.CITATION_IDS_VALID
        )
        abstained_terminal = (
            response.abstained
            and not response.citations
            and response.grounding_status == GroundingStatus.ABSTAINED
        )
        if not (grounded_terminal or abstained_terminal):
            raise HTTPException(
                status_code=410,
                detail="缓存响应不满足严格引用契约；请重新开始",
            )
    elif response.abstained:
        if response.citations or response.grounding_status != GroundingStatus.ABSTAINED:
            raise HTTPException(
                status_code=410,
                detail="缓存的拒答状态不一致；请重新开始",
            )
    elif response.citations:
        if response.grounding_status != GroundingStatus.CITATION_IDS_VALID:
            raise HTTPException(
                status_code=410,
                detail="缓存的引用状态不一致；请重新开始",
            )
    elif response.grounding_status != GroundingStatus.NOT_REQUESTED:
        raise HTTPException(
            status_code=410,
            detail="缓存的引用状态无效；请重新开始",
        )
    return response


def _request_fingerprint(operation: str, payload: dict) -> str:
    canonical = json.dumps(
        {"operation": operation, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _initial_conversation_id_for_recovery_token(recovery_token: str) -> str:
    digest = hashlib.sha256(
        f"agent.autonomous:{recovery_token}".encode("utf-8")
    ).hexdigest()
    return f"conv_{digest}"


def _initial_conversation_id(receipt_lease: ReceiptLease | None) -> str | None:
    """Use a stable pause ID so an ACK-lost start can find its first snapshot."""
    if receipt_lease is None:
        return None
    return _initial_conversation_id_for_recovery_token(
        receipt_lease.recovery_token
    )


async def _abort_receipt(receipt_lease: ReceiptLease | None) -> bool:
    if receipt_lease is None:
        return False
    return await abort_idempotency_claim(request_idempotency, receipt_lease)


async def _recover_initial_pause_from_session(
    key: str,
    req: AutonomousRequest,
    *,
    recovery_token: str | None = None,
) -> AutonomousResponse | None:
    """Repair an ACK-lost start receipt from its deterministic pause snapshot."""
    payload = req.model_dump(mode="json")
    if recovery_token is None:
        recovery_token = await request_idempotency.recovery_token(
            key,
            "agent.autonomous",
            payload,
        )
    if recovery_token is None:
        return None
    conversation_id = _initial_conversation_id_for_recovery_token(
        recovery_token
    )
    inspection = await autonomous_sessions.inspect(conversation_id)
    if inspection is None or inspection.state != "paused":
        return None
    try:
        session = _session_from_payload(conversation_id, inspection.payload)
    except (TypeError, ValueError, ValidationError) as exc:
        logger.error(
            "[autonomous] invalid initial pause during recovery: "
            "cid_prefix=%s error_type=%s",
            conversation_id[:13],
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=410,
            detail="暂停会话数据无效；请重新开始",
        ) from exc
    response = await _validate_replayed_response(
        _paused_response(session).model_dump(mode="json")
    )
    await request_idempotency.reconcile_completed(
        key,
        "agent.autonomous",
        payload,
        response.model_dump(mode="json"),
    )
    return response


async def _recover_continue_outcome_from_session(
    key: str,
    req: ContinueRequest,
) -> AutonomousResponse | None:
    """Repair an ACK-lost continue receipt from its completed session outcome."""
    payload = req.model_dump(mode="json")
    expected_fingerprint = _request_fingerprint(
        "agent.autonomous.continue", payload
    )
    inspection = await autonomous_sessions.inspect(req.conversation_id)
    if (
        inspection is None
        or inspection.state != "completed"
        or inspection.outcome is None
        or inspection.continue_fingerprint != expected_fingerprint
    ):
        return None
    response = await _validate_replayed_response(inspection.outcome)
    await request_idempotency.reconcile_completed(
        key,
        "agent.autonomous.continue",
        payload,
        response.model_dump(mode="json"),
        allow_effect_started=True,
    )
    return response


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
_NUMBERED_PLAN_LINE = re.compile(r"^(\d+)\s*[.)）、]\s*(.+)$")
_NAMED_PLAN_LINE = re.compile(r"^步骤\s*(\d+)\s*[:：]\s*(.+)$")
_BULLET_PLAN_LINE = re.compile(r"^[-*+•]\s+(.+)$")
_BARE_TOOL_CALL = re.compile(
    r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*\([^\n]*\)\s*;?$"
)


def _strip_inline_markdown(text: str) -> str:
    value = text.strip()
    for marker in ("**", "__", "`"):
        if (
            value.startswith(marker)
            and value.endswith(marker)
            and len(value) > 2 * len(marker)
        ):
            value = value[len(marker):-len(marker)].strip()
    return value


def _parse_plan(plan_text: str) -> list[str]:
    """Parse only an explicit 2-5 item plan, never prose or tool payloads."""
    raw = plan_text.strip()
    if not raw or raw[0] in "[{":
        return []

    steps: list[str] = []
    numbered_indexes: list[int] = []
    marker_kinds: set[str] = set()
    for raw_line in raw.splitlines():
        line = re.sub(r"^#{1,6}\s+", "", raw_line.strip())
        if not line:
            continue

        match = _NUMBERED_PLAN_LINE.fullmatch(line)
        if match:
            marker_kinds.add("numbered")
            numbered_indexes.append(int(match.group(1)))
            body = match.group(2)
        else:
            match = _NAMED_PLAN_LINE.fullmatch(line)
            if match:
                marker_kinds.add("numbered")
                numbered_indexes.append(int(match.group(1)))
                body = match.group(2)
            else:
                match = _BULLET_PLAN_LINE.fullmatch(line)
                if not match:
                    return []
                marker_kinds.add("bullet")
                body = match.group(1)

        step = _strip_inline_markdown(body)
        if not step or step[0] in "[{" or _BARE_TOOL_CALL.fullmatch(step):
            return []
        steps.append(step)

    if not 2 <= len(steps) <= 5 or len(marker_kinds) != 1:
        return []
    if numbered_indexes and numbered_indexes != list(range(1, len(steps) + 1)):
        return []
    return steps


def _plan_diagnostic(finish_reason: object, steps: list[str]) -> str:
    if finish_reason != "stop":
        return "invalid_finish_reason"
    return "valid" if steps else "invalid_format"


async def _generate_plan_within_budget(query: str) -> list[str]:
    for attempt in range(1, PLAN_SEMANTIC_ATTEMPTS + 1):
        system_prompt = _PLAN_SYSTEM
        if attempt > 1:
            system_prompt += (
                "\n上次响应结构无效。重新输出完整的 2-5 步编号列表，"
                "不要输出解释、JSON 或裸工具调用。"
            )
        response = await llm_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            client=_client,
            max_retries=PLAN_TRANSPORT_RETRIES,
            total_timeout=PLAN_TOTAL_TIMEOUT_SECONDS,
        )
        choice = response.choices[0]
        raw_plan = choice.message.content or ""
        if check_output_leak(raw_plan)[0]:
            logger.warning("[autonomous] plan output blocked by safety policy")
            return []
        steps = _parse_plan(raw_plan)
        diagnostic = _plan_diagnostic(
            getattr(choice, "finish_reason", None), steps
        )
        if diagnostic == "valid":
            logger.info(
                "[autonomous] plan accepted: attempt=%s step_count=%s",
                attempt,
                len(steps),
            )
            return steps
        logger.warning(
            "[autonomous] plan rejected: attempt=%s diagnostic=%s step_count=%s",
            attempt,
            diagnostic,
            len(steps),
        )
    return []


async def _generate_plan(query: str) -> list[str]:
    """Generate an isolated plan under one total deadline."""
    try:
        return await asyncio.wait_for(
            _generate_plan_within_budget(query),
            timeout=PLAN_TOTAL_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.warning(
            "[autonomous] plan generation failed: error_type=%s",
            type(exc).__name__,
        )
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
    receipt_lease: ReceiptLease | None = None,
    on_before_round: Callable[[], Awaitable[None]] | None = None,
    on_before_tool_calls: Callable[[], Awaitable[None]] | None = None,
    on_before_tool_dispatch: Callable[[], Awaitable[None]] | None = None,
    pause_session_saver: Callable[
        [AutonomousSession, AutonomousResponse],
        Awaitable[AutonomousResponse | None],
    ] | None = None,
    initial_conversation_id: str | None = None,
) -> AutonomousResponse:
    """从 starting_round 开始跑 ReAct 循环。命中 finalize / ask_user / max_rounds 时返回。"""
    truncated = False
    final_answer = ""
    finalize_reason: Optional[str] = None

    guard_business_tool = build_business_tool_scope_guard(
        user_id=user_id,
        document_id=document_id,
        allow_unbound_document_selection=True,
    )

    for round_idx in range(starting_round, MAX_AUTONOMOUS_ROUNDS):
        if on_before_round is not None:
            await on_before_round()
        # ── 注入 [Current state] 让 LLM 不健忘（仅本次调用，不持久化）──
        state_summary = _build_state_summary(
            steps, tools_called, plan, round_idx, evidence_registry
        )
        state_msg = [
            {"role": "system", "content": build_react_decision_prompt(state_summary)}
        ]

        # ── 单轮：run_tool_round 处理业务工具，控制工具交回本函数处理。──
        rr = await run_tool_round(
            messages,
            tools=get_tool_definitions() + _CONTROL_TOOLS,
            client=_client,
            control_tools=CONTROL_TOOL_NAMES,
            run_id=run_id,
            user_id=user_id,
            idempotency_key=(receipt_lease.key if receipt_lease is not None else None),
            idempotency_lease=receipt_lease,
            tool_choice="auto",
            extra_call_messages=state_msg,
            on_before_tool_calls=on_before_tool_calls,
            on_before_tool_dispatch=on_before_tool_dispatch,
            business_tool_guard=guard_business_tool,
        )

        # ── 无 tool_calls：LLM 直接给文字（视为隐式 finalize）──
        if not rr.has_tool_calls:
            final_answer = rr.content or ""
            if final_answer:
                finalize_reason = "implicit_finalize_no_tool_calls"
            else:
                final_answer = "（LLM 未给出回复且未调用工具，循环结束）"
                finalize_reason = "empty_response"
            return _build_response(
                plan, steps, tools_called, final_answer,
                round_idx + 1, truncated, finalize_reason,
                evidence_registry=evidence_registry,
                grounding_required=grounding_required,
                grounding_document_id=document_id,
            )

        # ── 逐个处理本轮 outcomes（顺序与 LLM 给的 tool_calls 一致）──
        for oc in rr.outcomes:
            fn_name, fn_args = oc.name, oc.arguments

            # ── 控制工具：finalize ──
            if oc.kind == "control" and fn_name == "finalize":
                try:
                    finalize_args = _FinalizeArgs.model_validate(fn_args)
                except ValidationError:
                    reason = "invalid_finalize_arguments"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": oc.call_id,
                        "content": (
                            '{"error":"finalize 参数无效，请按 schema 重新调用",'
                            f'"reason":"{reason}"}}'
                        ),
                    })
                    steps.append(StepRecord(
                        round_index=round_idx,
                        tool_name="finalize",
                        tool_args=fn_args if isinstance(fn_args, dict) else None,
                        blocked_reason=reason,
                    ))
                    continue

                final_answer = finalize_args.final_answer
                finalize_reason = finalize_args.reason
                citation_ids = finalize_args.citation_ids
                abstained = finalize_args.abstained
                # 补一个 tool message 让 messages 完整（OpenAI 协议要求 tool_call 都有 response）
                messages.append({"role": "tool", "tool_call_id": oc.call_id, "content": "Acknowledged."})
                # final_answer 只允许通过经过安全与引用检查的顶层字段对外返回，
                # 不能在公开 trajectory 中保留一份未经检查的副本。
                public_tool_args = {
                    key: value for key, value in fn_args.items()
                    if key != "final_answer"
                }
                steps.append(StepRecord(
                    round_index=round_idx, tool_name="finalize", tool_args=public_tool_args,
                    observation_preview="(loop ended)",
                ))
                return _build_response(
                    plan, steps, tools_called, final_answer,
                    round_idx + 1, truncated, finalize_reason,
                    evidence_registry=evidence_registry,
                    grounding_required=grounding_required,
                    grounding_document_id=document_id,
                    citation_ids=citation_ids,
                    abstained=abstained,
                )

            # ── 控制工具：ask_user → 保存 session 并返回 ──
            if oc.kind == "control" and fn_name == "ask_user":
                try:
                    ask_args = _AskUserArgs.model_validate(fn_args)
                except ValidationError:
                    reason = "invalid_ask_user_arguments"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": oc.call_id,
                        "content": (
                            '{"error":"ask_user 参数无效，请按 schema 重新调用",'
                            f'"reason":"{reason}"}}'
                        ),
                    })
                    steps.append(StepRecord(
                        round_index=round_idx,
                        tool_name="ask_user",
                        tool_args=fn_args if isinstance(fn_args, dict) else None,
                        blocked_reason=reason,
                    ))
                    continue

                question = ask_args.question
                question_leaked, _ = check_output_leak(question)
                if question_leaked:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": oc.call_id,
                        "content": "Rejected: sensitive output.",
                    })
                    steps.append(StepRecord(
                        round_index=round_idx,
                        tool_name="ask_user",
                        tool_args=fn_args,
                        blocked_reason="output_leak_blocked",
                    ))
                    return _build_response(
                        plan, steps, tools_called, question,
                        round_idx + 1, truncated, "ask_user_output_leak",
                        evidence_registry=evidence_registry,
                        grounding_required=grounding_required,
                        grounding_document_id=document_id,
                        abstained=True,
                    )

                conversation_id = initial_conversation_id or f"conv_{uuid.uuid4().hex}"
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
                    registry_sha256=_current_registry_sha256(),
                    semantic_reservation_digests=(
                        tool_registry.snapshot_run_policy_state(run_id)
                    ),
                )
                response = AutonomousResponse(
                    plan=_public_plan(plan),
                    steps=_public_steps(steps),
                    tools_called=list(dict.fromkeys(tools_called)),
                    rounds_used=round_idx + 1,
                    truncated=False,
                    awaiting_user_input=True,
                    user_question=question,
                    conversation_id=conversation_id,
                    grounding_status=(
                        GroundingStatus.PENDING
                        if grounding_required
                        else GroundingStatus.NOT_REQUESTED
                    ),
                    grounding_required=grounding_required,
                    grounding_document_id=document_id,
                )
                try:
                    if pause_session_saver is None:
                        await autonomous_sessions.save(
                            conversation_id, _session_to_payload(session)
                        )
                    else:
                        canonical = await pause_session_saver(session, response)
                        if canonical is not None:
                            response = canonical
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
                return response

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
                accepted = collect_search_evidence(
                    oc.result or "",
                    evidence_registry,
                    expected_document_id=document_id,
                )
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
    if on_before_round is not None:
        await on_before_round()
    try:
        finish_resp = await llm_chat(messages, client=_client)
        final_answer = finish_resp.choices[0].message.content or "执行被截断"
    except Exception as e:
        logger.warning(
            "[autonomous] finish call failed: error_type=%s",
            type(e).__name__,
        )
        final_answer = f"执行被截断（{MAX_AUTONOMOUS_ROUNDS} 轮）"

    return _build_response(
        plan, steps, tools_called, final_answer,
        MAX_AUTONOMOUS_ROUNDS, truncated, "max_rounds_truncated",
        evidence_registry=evidence_registry,
        grounding_required=grounding_required,
        grounding_document_id=document_id,
    )


def _build_response(
    plan, steps, tools_called, final_answer, rounds_used, truncated, finalize_reason,
    *,
    evidence_registry: EvidenceRegistry,
    grounding_required: bool,
    grounding_document_id: Optional[str],
    citation_ids: list[str] | None = None,
    abstained: bool = False,
) -> AutonomousResponse:
    if not isinstance(final_answer, str):
        final_answer = "" if final_answer is None else str(final_answer)
    if finalize_reason is not None and not isinstance(finalize_reason, str):
        finalize_reason = str(finalize_reason)

    output_blocked, _ = check_output_leak(
        f"{final_answer}\n{finalize_reason or ''}"
    )
    if output_blocked:
        logger.warning("[autonomous] output leak blocked by safety policy")
        final_answer = "输出包含敏感信息已拦截。"
        finalize_reason = "output_leak_blocked"
        abstained = True

    resolution = resolve_citations(citation_ids, evidence_registry)
    if grounding_document_id is None:
        # Unscoped tool observations are useful to the model, but public
        # citations need a persisted document boundary that can be replayed
        # and independently revalidated later.
        resolution = CitationResolution(
            citations=[], invalid_ids=resolution.invalid_ids
        )
    if not output_blocked and any(
        check_output_leak(citation.model_dump_json())[0]
        for citation in resolution.citations
    ):
        output_blocked = True
        final_answer = "输出包含敏感信息已拦截。"
        finalize_reason = "output_leak_blocked"
        abstained = True

    if output_blocked:
        # Output safety is the highest-priority terminal reason; grounding
        # checks must not overwrite it with a less precise explanation.
        pass
    elif abstained:
        # `abstained` is model-controlled input. Never let a confident answer
        # survive next to an abstention status in the public response.
        final_answer = _GROUNDING_ABSTENTION
        finalize_reason = "model_abstained"
    elif grounding_required and resolution.invalid_ids:
        # In strict mode, one forged ID may represent an unsupported claim;
        # a different surviving citation cannot make that answer safe.
        final_answer = _GROUNDING_ABSTENTION
        finalize_reason = "grounding_required_with_invalid_citation"
        abstained = True
    elif grounding_required and not resolution.citations:
        final_answer = _GROUNDING_ABSTENTION
        finalize_reason = "grounding_required_without_valid_citation"
        abstained = True

    if abstained:
        grounding_status = GroundingStatus.ABSTAINED
    elif resolution.citations:
        grounding_status = GroundingStatus.CITATION_IDS_VALID
    else:
        grounding_status = GroundingStatus.NOT_REQUESTED

    return AutonomousResponse(
        plan=[] if abstained else _public_plan(plan),
        steps=_public_steps(steps, redact_details=abstained),
        tools_called=list(dict.fromkeys(tools_called)),
        rounds_used=rounds_used, truncated=truncated,
        final_answer=final_answer, finalize_reason=finalize_reason,
        citations=[] if abstained else resolution.citations,
        invalid_citation_ids=[],
        invalid_citation_count=len(resolution.invalid_ids),
        abstained=abstained,
        grounding_status=grounding_status,
        grounding_required=grounding_required,
        grounding_document_id=grounding_document_id,
    )


def _paused_response(session: AutonomousSession) -> AutonomousResponse:
    question = _pending_ask_question(
        session.messages, session.pending_ask_call_id
    )
    if not question:
        raise ValueError("persisted pause is missing its ask_user question")
    return AutonomousResponse(
        plan=_public_plan(session.plan),
        steps=_public_steps(session.steps),
        tools_called=list(dict.fromkeys(session.tools_called)),
        rounds_used=session.rounds_used,
        truncated=False,
        awaiting_user_input=True,
        user_question=question,
        conversation_id=session.conversation_id,
        grounding_status=(
            GroundingStatus.PENDING
            if session.grounding_required
            else GroundingStatus.NOT_REQUESTED
        ),
        grounding_required=session.grounding_required,
        grounding_document_id=session.document_id,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════
async def _execute_autonomous(
    req: AutonomousRequest,
    *,
    run_id: str,
    receipt_lease: ReceiptLease | None,
) -> AutonomousResponse:
    """Autonomous ReAct Agent 端点。

    范式：真 ReAct（LLM 全权决策）
    可选 plan：短 query 跳过，长 query 生成 plan 作为 hint
    HITL：LLM 可调 ask_user 触发两段式协议
    """
    # 第 1 层：Prompt Injection 检测
    is_injection, _ = await check_injection(req.query)
    if is_injection:
        logger.warning("[autonomous] input blocked by injection policy")
        return AutonomousResponse(
            final_answer="输入安全检查未通过。请修改目标后重试。",
            finalize_reason="input_safety_blocked",
            abstained=True,
            grounding_status=GroundingStatus.ABSTAINED,
            grounding_required=req.grounding_required,
            grounding_document_id=req.document_id,
        )

    context_hint = f"\n\n当前用户 ID: {req.user_id}"
    if req.document_id:
        context_hint += (
            f"\n当前文档 ID: {req.document_id}"
            "\n所有 search_document 调用都必须使用这个文档 ID。"
        )
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
        plan = await _generate_plan(req.query)
        logger.info("[autonomous] plan generated: steps=%d", len(plan))

    # 构造初始 messages；工具调用由 ReAct 自主决策
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

    async def renew_receipt() -> None:
        if receipt_lease is None:
            return
        if await request_idempotency.renew(receipt_lease) is None:
            raise IdempotencyConflictError("in_progress")

    async def save_initial_pause(
        session: AutonomousSession,
        response: AutonomousResponse,
    ) -> AutonomousResponse:
        # The provider call may outlive the receipt lease. Fence immediately
        # before publishing a pause so a stale worker cannot later reconcile
        # its response over the current owner.
        await renew_receipt()
        async def recover_committed_pause() -> AutonomousResponse | None:
            inspection = await autonomous_sessions.inspect(session.conversation_id)
            if inspection is None:
                return None
            if inspection.state == "in_flight":
                raise IdempotencyConflictError("in_progress")
            if inspection.state != "paused":
                raise IdempotencyConflictError("ambiguous")
            persisted = _session_from_payload(
                session.conversation_id, inspection.payload
            )
            canonical = await _validate_replayed_response(
                _paused_response(persisted).model_dump(mode="json")
            )
            if receipt_lease is not None:
                # Initial pauses are only safe to repair from a clean pending
                # receipt. An effect-started receipt may belong to a newer
                # takeover owner and must remain fail-closed.
                await request_idempotency.reconcile_completed(
                    receipt_lease.key,
                    "agent.autonomous",
                    req.model_dump(mode="json"),
                    canonical.model_dump(mode="json"),
                )
            return canonical

        try:
            await autonomous_sessions.save(
                session.conversation_id, _session_to_payload(session)
            )
            return response
        except BaseException as save_exc:
            # save() may commit and then lose its ACK. Drain a shielded lookup
            # (also under repeated cancellation) before deciding whether the
            # canonical pause exists. This prevents deleting the receipt and
            # generating a second recovery capability after a committed save.
            repair_task = asyncio.create_task(recover_committed_pause())
            cancellation = (
                save_exc if isinstance(save_exc, asyncio.CancelledError) else None
            )
            while not repair_task.done():
                try:
                    await asyncio.shield(repair_task)
                except asyncio.CancelledError as later_cancel:
                    cancellation = later_cancel
                    continue
            try:
                recovered = repair_task.result()
            except BaseException as repair_exc:
                if cancellation is not None:
                    raise cancellation
                raise repair_exc from save_exc
            if cancellation is not None:
                raise cancellation
            if recovered is not None:
                return recovered
            raise

    return await _run_react_loop(
        messages=messages, plan=plan, steps=[], tools_called=[],
        user_id=req.user_id, document_id=req.document_id,
        starting_round=0, run_id=run_id,
        evidence_registry={}, grounding_required=req.grounding_required,
        receipt_lease=receipt_lease,
        on_before_round=renew_receipt,
        pause_session_saver=(
            save_initial_pause if receipt_lease is not None else None
        ),
        initial_conversation_id=_initial_conversation_id(receipt_lease),
    )


@router.post("/agent/autonomous", response_model=AutonomousResponse)
async def autonomous_agent(
    req: AutonomousRequest,
    idempotency_key: Optional[str] = Header(
        default=None, alias="Idempotency-Key"
    ),
    subject: str = Depends(require_user_id),
) -> AutonomousResponse:
    """Run one request with an optional durable replay receipt."""
    # 请求体里自报的 user_id 不作数：身份只来自 Authorization 头解析出的主体。
    req.user_id = subject
    key = normalize_idempotency_key(idempotency_key)
    receipt_lease = None
    request_payload = req.model_dump(mode="json")
    if key:
        try:
            decision = await request_idempotency.begin(
                key, "agent.autonomous", request_payload
            )
        except IdempotencyConflictError as exc:
            if exc.reason in {"in_progress", "ambiguous"}:
                recovered = await _recover_initial_pause_from_session(key, req)
                if recovered is not None:
                    return recovered
            raise
        if decision.replayed:
            return await _validate_replayed_response(decision.response)
        receipt_lease = decision.lease
        if receipt_lease is None:
            raise RuntimeError("idempotency claim returned without ownership data")
        recovered = await _recover_initial_pause_from_session(
            key,
            req,
            recovery_token=receipt_lease.recovery_token,
        )
        if recovered is not None:
            return recovered

    run_id = f"auto_{uuid.uuid4().hex[:12]}"
    durable_pause = False
    try:
        response = await _execute_autonomous(
            req, run_id=run_id, receipt_lease=receipt_lease
        )
        durable_pause = response.awaiting_user_input
        if receipt_lease is not None:
            try:
                await request_idempotency.complete(
                    receipt_lease, response.model_dump(mode="json")
                )
            except Exception:
                if not durable_pause:
                    raise
                await request_idempotency.reconcile_completed(
                    key,
                    "agent.autonomous",
                    request_payload,
                    response.model_dump(mode="json"),
                )
        return response
    except BaseException as exc:
        if isinstance(exc, asyncio.CancelledError):
            # Cancellation is an uncertain network/process boundary. Keep the
            # bounded lease so an exact retry can take over or repair a pause;
            # deleting it here could orphan a commit whose ACK was lost.
            raise
        if (
            isinstance(exc, IdempotencyConflictError)
            and exc.reason in {"in_progress", "ambiguous"}
        ):
            raise
        if durable_pause:
            # The exact pending response is already represented by the durable
            # pause snapshot. Keep the browser's exact key bound for repair
            # instead of turning a receipt-ack failure into a terminal conflict.
            if isinstance(exc, Exception):
                raise IdempotencyConflictError("in_progress") from exc
            raise
        durable_effect = await _abort_receipt(receipt_lease)
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
    finally:
        tool_registry.clear_run_policy_state(run_id)


@router.delete("/agent/autonomous/{conversation_id}")
async def cancel_autonomous_session(
    conversation_id: str = ApiPath(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
    ),
) -> dict[str, str]:
    """Cancel only a currently paused HITL session.

    A running owner is never interrupted by this endpoint. Missing, expired,
    and already-completed capabilities are idempotent no-ops for the browser.
    """
    result = await autonomous_sessions.discard_paused(conversation_id)
    if result.reason == "in_progress":
        raise HTTPException(
            status_code=409,
            detail="conversation 正在续跑，当前不能取消",
        )
    if result.reason == "ambiguous":
        raise IdempotencyConflictError("ambiguous")
    return {
        "status": "canceled" if result.discarded else "missing",
    }


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
    receipt_lease = None
    if key:
        try:
            decision = await request_idempotency.begin(
                key, "agent.autonomous.continue", req.model_dump(mode="json")
            )
        except IdempotencyConflictError as exc:
            if exc.reason in {"in_progress", "ambiguous"}:
                recovered = await _recover_continue_outcome_from_session(
                    key, req
                )
                if recovered is not None:
                    return recovered
            raise
        if decision.replayed:
            return await _validate_replayed_response(decision.response)
        receipt_lease = decision.lease
        if receipt_lease is None:
            raise RuntimeError("idempotency claim returned without ownership data")

    # 数据库 CAS 认领 session，跨进程也只允许一个续跑 owner。
    continue_fingerprint = _request_fingerprint(
        "agent.autonomous.continue", req.model_dump(mode="json")
    )
    try:
        claim = await autonomous_sessions.claim(
            req.conversation_id, continue_fingerprint
        )
    except BaseException:
        await _abort_receipt(receipt_lease)
        raise
    if not claim.claimed:
        if claim.reason == "completed":
            if claim.outcome is None:
                await _abort_receipt(receipt_lease)
                raise HTTPException(
                    status_code=410,
                    detail="续跑结果记录无效；请重新开始",
                )
            response = await _validate_replayed_response(claim.outcome)
            if receipt_lease is not None:
                await request_idempotency.complete(
                    receipt_lease, response.model_dump(mode="json")
                )
            return response
        await _abort_receipt(receipt_lease)
        if claim.reason == "payload_mismatch":
            raise IdempotencyConflictError("payload_mismatch")
        if claim.reason == "ambiguous":
            raise IdempotencyConflictError("ambiguous")
        if claim.reason == "in_progress":
            raise IdempotencyConflictError("in_progress")
        raise HTTPException(status_code=404, detail="conversation 不存在或已过期")
    if not claim.claim_token:
        raise RuntimeError("session claim returned without ownership data")

    claim_token = claim.claim_token
    if claim.payload is None:
        await autonomous_sessions.consume(req.conversation_id, claim_token)
        await _abort_receipt(receipt_lease)
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
        await _abort_receipt(receipt_lease)
        logger.error(
            "[autonomous] invalid persisted session: cid_prefix=%s error_type=%s",
            req.conversation_id[:13],
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=410,
            detail="暂停会话数据无效，已安全终止；请重新开始",
        ) from exc

    # 注入检测尚未改变 session 或执行工具；失败时把 clean claim 原子退回暂停态。
    try:
        is_injection, _ = await check_injection(req.user_reply)
    except BaseException:
        released = await autonomous_sessions.release(
            req.conversation_id, claim_token
        )
        await _abort_receipt(receipt_lease)
        if not released:
            raise IdempotencyConflictError("in_progress")
        raise
    if is_injection:
        logger.warning("[autonomous] continuation blocked by injection policy")
        released = await autonomous_sessions.release(
            req.conversation_id, claim_token
        )
        await _abort_receipt(receipt_lease)
        if not released:
            raise IdempotencyConflictError("in_progress")
        raise HTTPException(
            status_code=422,
            detail="用户回答安全检查未通过，请修改后重试",
        )

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
    outcome_committed = False
    outcome_commit_attempted = False

    async def renew_ownership() -> None:
        if receipt_lease is not None:
            renewed = await request_idempotency.renew(receipt_lease)
            if renewed is None:
                raise IdempotencyConflictError("in_progress")
        if not await autonomous_sessions.renew(
            req.conversation_id, claim_token
        ):
            raise IdempotencyConflictError("in_progress")

    async def mark_progress_before_tool_calls() -> None:
        nonlocal progress_started
        marked = await autonomous_sessions.mark_progress(
            req.conversation_id, claim_token
        )
        if not marked:
            raise IdempotencyConflictError("in_progress")
        progress_started = True

    async def handoff_to_next_pause(
        next_session: AutonomousSession,
        response: AutonomousResponse,
    ) -> AutonomousResponse:
        nonlocal outcome_commit_attempted, outcome_committed
        outcome_commit_attempted = True
        handed_off = await autonomous_sessions.handoff(
            req.conversation_id,
            claim_token,
            next_session.conversation_id,
            _session_to_payload(next_session),
            response.model_dump(mode="json"),
        )
        if not handed_off:
            raise RuntimeError("lost autonomous session claim during pause handoff")
        outcome_committed = True
        return response

    run_id = f"auto_cont_{uuid.uuid4().hex[:12]}"
    try:
        if session.registry_sha256 != _current_registry_sha256():
            raise ValueError("persisted tool registry fingerprint is stale")
        tool_registry.restore_run_policy_state(
            run_id,
            session.semantic_reservation_digests,
        )
    except ValueError as exc:
        await autonomous_sessions.consume(req.conversation_id, claim_token)
        await _abort_receipt(receipt_lease)
        tool_registry.clear_run_policy_state(run_id)
        logger.warning(
            "[autonomous] invalid persisted tool policy state: cid_prefix=%s",
            req.conversation_id[:13],
        )
        raise HTTPException(
            status_code=410,
            detail="暂停会话安全状态无效，已终止；请重新开始",
        ) from exc
    baseline = (
        len(resume_messages),
        len(resume_steps),
        len(resume_tools_called),
        len(resume_evidence_registry),
    )
    # handoff_to_next_pause 会在 _run_react_loop 返回前提交结果并置位
    # outcome_committed，此时下面这个绑定还不存在。显式初始化让异常处理
    # 能够区分"已提交且拿得到响应"和"已提交但引用未绑定"。
    response: AutonomousResponse | None = None
    try:
        response = await _run_react_loop(
            messages=resume_messages, plan=session.plan,
            steps=resume_steps, tools_called=resume_tools_called,
            user_id=session.user_id, document_id=session.document_id,
            starting_round=session.rounds_used, run_id=run_id,
            evidence_registry=resume_evidence_registry,
            grounding_required=session.grounding_required,
            receipt_lease=receipt_lease,
            on_before_round=renew_ownership,
            on_before_tool_calls=renew_ownership,
            on_before_tool_dispatch=mark_progress_before_tool_calls,
            pause_session_saver=handoff_to_next_pause,
        )
        if not outcome_committed:
            outcome_commit_attempted = True
            if not await autonomous_sessions.finish(
                req.conversation_id,
                claim_token,
                response.model_dump(mode="json"),
            ):
                raise RuntimeError("lost autonomous session claim during completion")
            outcome_committed = True
        if receipt_lease is not None:
            await request_idempotency.complete(
                receipt_lease, response.model_dump(mode="json")
            )
        return response
    except BaseException as exc:
        if (
            outcome_commit_attempted
            and receipt_lease is not None
            and isinstance(exc, Exception)
        ):
            try:
                recovered = await _recover_continue_outcome_from_session(
                    receipt_lease.key,
                    req,
                )
            except Exception as recovery_exc:
                raise IdempotencyConflictError("in_progress") from recovery_exc
            if recovered is not None:
                return recovered
        if isinstance(exc, Exception):
            # A long provider call can outlive the session lease.  finish or
            # handoff then reports ordinary False/RuntimeError, while a round
            # fence reports IdempotencyConflictError.  Recheck ownership for
            # both shapes before deciding whether this is a normal provider
            # failure or a peer-owned/canonical state transition.
            session_still_owned = await autonomous_sessions.renew(
                req.conversation_id, claim_token
            )
        else:
            session_still_owned = True
        if (
            isinstance(exc, Exception)
            and (
                not session_still_owned
                or isinstance(exc, IdempotencyConflictError)
            )
        ):
            inspection = await autonomous_sessions.inspect(req.conversation_id)
            if (
                inspection is not None
                and inspection.state == "completed"
                and inspection.outcome is not None
                and inspection.continue_fingerprint == continue_fingerprint
            ):
                recovered_response = await _validate_replayed_response(
                    inspection.outcome
                )
                if receipt_lease is not None:
                    await request_idempotency.reconcile_completed(
                        receipt_lease.key,
                        "agent.autonomous.continue",
                        req.model_dump(mode="json"),
                        recovered_response.model_dump(mode="json"),
                        allow_effect_started=True,
                    )
                return recovered_response
            if inspection is not None and inspection.state == "ambiguous":
                await _abort_receipt(receipt_lease)
                raise IdempotencyConflictError("ambiguous") from exc
            if inspection is not None and inspection.state == "in_flight":
                if session_still_owned and not progress_started:
                    await autonomous_sessions.release(
                        req.conversation_id, claim_token
                    )
                raise IdempotencyConflictError("in_progress") from exc
            if (
                inspection is not None
                and inspection.state == "paused"
                and inspection.continue_fingerprint == continue_fingerprint
            ):
                # Inspect may reap our expired clean session lease back to the
                # same canonical pause. Drop only our fenced receipt claim and
                # let the exact browser request acquire both leases again.
                await _abort_receipt(receipt_lease)
                raise IdempotencyConflictError("in_progress") from exc
        trajectory_progressed = baseline != (
            len(resume_messages),
            len(resume_steps),
            len(resume_tools_called),
            len(resume_evidence_registry),
        )
        business_tool_dispatched = len(resume_tools_called) > baseline[2]
        if outcome_committed:
            if not isinstance(exc, Exception):
                raise
            if response is None:
                # 结果已经提交，但本地拿不到那份响应，无法用它对账收据。
                # 报 ambiguous 让调用方带原 key 重试，从会话快照取回结果。
                raise IdempotencyConflictError("in_progress") from exc
            if receipt_lease is not None:
                try:
                    await request_idempotency.reconcile_completed(
                        receipt_lease.key,
                        "agent.autonomous.continue",
                        req.model_dump(mode="json"),
                        response.model_dump(mode="json"),
                        allow_effect_started=True,
                    )
                except Exception:
                    # Keep the canonical session outcome and exact browser key;
                    # the next retry can repeat this repair without rerunning.
                    raise IdempotencyConflictError("in_progress") from exc
                return response
            raise

        # Registry 在非幂等/未知 handler 前写 started marker。即使工具已开始而
        # trajectory 尚未来得及 append，也必须 fail closed，避免重放副作用。
        durable_effect = await _abort_receipt(receipt_lease)
        effect_attempted = (
            durable_effect or tool_registry.has_effect_attempt(run_id)
        )
        # Model/control trajectory changes are still local until a terminal
        # outcome is journaled. Only an actually dispatched business tool (or
        # its durable side-effect marker) makes a retry unsafe.
        progressed = (
            progress_started or business_tool_dispatched or effect_attempted
        )

        if not progressed:
            released = await autonomous_sessions.release(
                req.conversation_id, claim_token
            )
            if released:
                raise

        await autonomous_sessions.consume(req.conversation_id, claim_token)

        logger.warning(
            "[autonomous] continuation terminated fail-closed: "
            "cid_prefix=%s progressed=%s trajectory_changed=%s",
            req.conversation_id[:13],
            progressed,
            trajectory_progressed,
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
    finally:
        tool_registry.clear_run_policy_state(run_id)
