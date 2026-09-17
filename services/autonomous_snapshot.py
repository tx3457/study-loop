"""Autonomous 会话快照的编解码与对外脱敏。

一份暂停中的会话有两条完全不同的出口：写进持久化快照供续跑，或者作为 HTTP
响应交给浏览器。两条路的危险是同一个——内部状态里带着工具原始参数、检索正文
和用户材料，任何一条泄漏出去都不可撤回。把两条出口放在一个模块里，脱敏规则
就只有一份，不会一边改了另一边忘了。

快照还承担恢复期的完整性校验：反序列化要求轨迹必须终止在那个待回答的
ask_user 上，并比对工具注册表指纹——工具定义变过就不允许旧会话续跑，否则模型
会在一套已经改写的工具语义下接着执行半场对话。
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from services.citations import EvidenceChunk, EvidenceRegistry
from services.injection import check_output_leak
from services.react_controls import CONTROL_TOOL_NAMES
from services.tool_registry import tool_registry

# 同一个上限同时约束"能执行几轮"和"快照能存几轮"。分开定义迟早会漂移，
# 变成跑得完却存不下的会话。
MAX_AUTONOMOUS_ROUNDS = 8

_SESSION_SCHEMA_VERSION = 3


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


def public_steps(
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


def public_plan(plan: list[str]) -> list[str]:
    return [_sanitize_public_value(item) for item in plan]


def _safe_snapshot_plan(plan: list[str]) -> list[str]:
    """Keep model-authored plan text out of durable state when it trips DLP."""

    return [] if check_output_leak("\n".join(plan))[0] else list(plan)


class FinalizeArgs(BaseModel):
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


class AskUserArgs(BaseModel):
    """Strict runtime boundary for model-generated HITL questions."""

    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def validate_question(self):
        if not self.question.strip():
            raise ValueError("question must not be blank")
        return self


def pending_ask_question(
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
                return AskUserArgs.model_validate_json(
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

        question = pending_ask_question(
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


def current_registry_sha256() -> str:
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


def session_to_payload(session: AutonomousSession) -> dict:
    messages, pending_ask_call_id = _sanitize_snapshot_messages(
        session.messages, session.pending_ask_call_id
    )

    snapshot_steps = public_steps(session.steps)
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
        registry_sha256=(session.registry_sha256 or current_registry_sha256()),
        semantic_reservation_digests=list(
            session.semantic_reservation_digests
        ),
    )
    return payload.model_dump(mode="json")


def session_from_payload(
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
        migrated["registry_sha256"] = current_registry_sha256()
        migrated["semantic_reservation_digests"] = []
    migrated["messages"], migrated["pending_ask_call_id"] = _sanitize_snapshot_messages(
        snapshot.messages, snapshot.pending_ask_call_id
    )
    migrated["steps"] = [
        step.model_dump(mode="json") for step in public_steps(snapshot.steps)
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
