"""
Tool Registry + Audit Trail

面向 StudyLoop 异步工具调用路径的统一注册表：
  - 超时机制：用 asyncio.wait_for（cooperative cancel），不用 ThreadPoolExecutor（伪超时）
  - 重试：复用 services/retry.py:with_retry
  - 元数据：每个 tool 单独声明 timeout / retry / permission / effect_mode
  - 审计：内存 LRU 记录每次 invoke 的输入/输出/耗时/状态，可按 run_id/user_id 查询

设计选择：
  1. 单例：ToolRegistry 全局唯一，避免 tool 重复注册和 audit 数据分裂
  2. 内存 LRU audit：1000 条上限，零依赖。生产里换 PostgreSQL 只改这一个类
     （Strategy Pattern + Open/Closed）
  3. invoke 把 retry + timeout + audit 三件套打包：业务侧 dispatch_tool 一行
     就能拿到全部能力，不用重复实现
  4. 只有只读或幂等工具允许自动重试；未知/非幂等工具开始执行后结果不明时
     fail closed，避免在同一进程内静默重放副作用
"""
import asyncio
import copy
import hashlib
import inspect
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Awaitable, Callable, Optional

from services.idempotency import (
    IdempotencyConflictError,
    ReceiptLease,
    request_idempotency,
)
from services.retry import RetryExhausted, with_retry

logger = logging.getLogger(__name__)


# ── 数据结构 ─────────────────────────────────────────────────────────────────
class EffectMode(str, Enum):
    """工具执行语义；未知工具按最保守的不可重放处理。"""

    READ_ONLY = "read_only"
    IDEMPOTENT = "idempotent"
    NON_IDEMPOTENT = "non_idempotent"
    UNKNOWN = "unknown"


_REPLAY_SAFE_EFFECTS = {EffectMode.READ_ONLY, EffectMode.IDEMPOTENT}
_NON_REPLAYABLE_EFFECTS = {EffectMode.NON_IDEMPOTENT, EffectMode.UNKNOWN}


class SideEffectAmbiguousError(RuntimeError):
    """A state-changing tool failed after execution began; replay is unsafe."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"工具 {tool_name} 的执行结果不确定，禁止自动重试")


class ToolPolicyViolation(RuntimeError):
    """A tool call rejected by a registry-enforced safety policy."""

    def __init__(self, reason: str, tool_name: Optional[str] = None):
        self.reason = reason
        self.tool_name = tool_name
        super().__init__(f"tool policy violation: {reason}")


@dataclass(frozen=True)
class ToolArgumentBinding:
    """Authoritatively bind an upstream tool result to a downstream argument."""

    source_tool: str
    source_path: str
    target_argument: str


@dataclass
class ToolMetadata:
    """工具元数据：每个 tool 自己声明 SLO、重试、权限和副作用语义。"""
    timeout_sec: float = 30.0       # per-call 超时
    max_retries: int = 2            # 重试次数（不含首次调用）
    base_delay: float = 1.0         # 重试 backoff 基础秒数
    permission: str = "public"      # 权限标签，预留扩展
    effect_mode: EffectMode = EffectMode.UNKNOWN
    owner_argument: Optional[str] = None
    dedupe_within_run: bool = False
    dedupe_argument_paths: tuple[str, ...] = ()
    dedupe_normalizer: Optional[Callable[[dict], object]] = field(
        default=None,
        repr=False,
        compare=False,
    )
    dedupe_normalizer_id: Optional[str] = None
    argument_bindings: tuple[ToolArgumentBinding, ...] = ()

    def security_contract_payload(self) -> dict:
        """Stable JSON payload for binding/fingerprinting enforced semantics."""
        return {
            "version": 1,
            "effect_mode": self.effect_mode.value,
            "owner_argument": self.owner_argument,
            "dedupe_within_run": self.dedupe_within_run,
            "dedupe_argument_paths": list(self.dedupe_argument_paths),
            "dedupe_normalizer_id": self.dedupe_normalizer_id,
            "argument_bindings": [
                asdict(binding) for binding in self.argument_bindings
            ],
        }


@dataclass
class Tool:
    """单个工具的完整声明"""
    name: str
    description: str
    parameters_schema: dict         # OpenAI Function Calling JSON Schema
    handler: Callable[..., Awaitable[str]]   # async callable, 返回 JSON 字符串
    metadata: ToolMetadata = field(default_factory=ToolMetadata)
    _handler_signature: inspect.Signature = field(init=False, repr=False)

    def __post_init__(self) -> None:
        has_normalizer = self.metadata.dedupe_normalizer is not None
        has_normalizer_id = bool(self.metadata.dedupe_normalizer_id)
        if has_normalizer != has_normalizer_id:
            raise ValueError(
                "dedupe_normalizer and dedupe_normalizer_id must be configured together"
            )
        if has_normalizer and self.metadata.dedupe_argument_paths:
            raise ValueError(
                "dedupe_normalizer cannot be combined with dedupe_argument_paths"
            )
        self._handler_signature = inspect.signature(self.handler)

    def to_openai_schema(self) -> dict:
        """转 OpenAI tools 参数格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


@dataclass
class ToolCallRecord:
    """单次调用的审计记录（结构化日志）"""
    tool_call_id: str
    tool_name: str
    arguments: dict
    status: str                     # started / ok / error / timeout / cancelled / ambiguous
    duration_ms: float
    timestamp: str                  # ISO format
    effect_mode: str = EffectMode.UNKNOWN.value
    run_id: Optional[str] = None
    user_id: Optional[str] = None
    output_preview: Optional[str] = None       # 截断 1000 字节避免爆内存
    error_message: Optional[str] = None
    retry_attempts: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ── 单例 Registry ────────────────────────────────────────────────────────────
class ToolRegistry:
    """工具注册器（单例）+ 内存 LRU audit。"""

    _instance: Optional["ToolRegistry"] = None
    _AUDIT_MAX = 1000               # audit 内存上限
    _POLICY_MAX_RESERVATIONS = 256
    _POLICY_TOOL_NAME_MAX_BYTES = 128

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: dict[str, Tool] = {}
            cls._instance._audit_log: list[ToolCallRecord] = []
            cls._instance._run_effect_digests: dict[
                str, set[tuple[str, str]]
            ] = {}
            cls._instance._run_effect_attempts: set[str] = set()
        return cls._instance

    # ── 注册 / 查询 ──────────────────────────────────────────────────────
    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            logger.warning(f"[tool_registry] tool '{tool.name}' already registered, overwriting")
        self._tools[tool.name] = tool
        logger.info(
            f"[tool_registry] registered: {tool.name} "
            f"(timeout={tool.metadata.timeout_sec}s, retries={tool.metadata.max_retries}, "
            f"effect={tool.metadata.effect_mode})"
        )

    def unregister(self, name: str, *, expected_tool: Tool | None = None) -> bool:
        """Remove a tool without deleting a newer owner that reused its name."""
        current = self._tools.get(name)
        if current is None:
            return False
        if expected_tool is not None and current is not expected_tool:
            return False
        del self._tools[name]
        logger.info(f"[tool_registry] unregistered: {name}")
        return True

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_openai_schemas(self) -> list[dict]:
        """返回所有 tool 的 OpenAI Function Calling schema 列表"""
        return [t.to_openai_schema() for t in self._tools.values()]

    # ── 核心：invoke 把 retry + timeout + audit 打包 ────────────────────
    async def invoke(
        self,
        name: str,
        arguments: dict,
        *,
        run_id: Optional[str] = None,
        user_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        idempotency_lease: ReceiptLease | None = None,
        on_before_handler: Callable[[], Awaitable[None]] | None = None,
    ) -> str:
        """调用工具：带 per-tool timeout、retry、audit。

        返回工具的 str 输出（通常是 JSON 字符串，喂给 LLM 的 tool message）。
        """
        tool = self._tools.get(name)
        if not tool:
            err = "unknown_tool"
            self._record(ToolCallRecord(
                tool_call_id=uuid.uuid4().hex,
                tool_name=name,
                arguments=arguments,
                status="error",
                duration_ms=0.0,
                timestamp=datetime.now().isoformat(),
                run_id=run_id, user_id=user_id,
                error_message=err,
            ))
            return '{"error": "未知工具"}'

        tool_call_id = uuid.uuid4().hex
        effect_mode = tool.metadata.effect_mode

        required = tool.parameters_schema.get("required", [])
        missing = [field for field in required if field not in arguments]
        declared = set(tool.parameters_schema.get("properties", {}))
        closed_schema = (
            tool.parameters_schema.get("additionalProperties") is False
        )
        unknown = (
            [field for field in arguments if field not in declared]
            if closed_schema
            else []
        )
        binding_error = None
        if missing:
            binding_error = "missing_required_arguments"
        elif unknown:
            binding_error = "unknown_arguments"
        else:
            try:
                tool._handler_signature.bind(**arguments)
            except TypeError:
                binding_error = "signature_mismatch"

        if binding_error:
            message = f"工具 {name} 参数无效"
            self._record(ToolCallRecord(
                tool_call_id=tool_call_id,
                tool_name=name,
                arguments=arguments,
                status="error",
                duration_ms=0.0,
                timestamp=datetime.now().isoformat(),
                effect_mode=effect_mode.value,
                run_id=run_id,
                user_id=user_id,
                error_message=binding_error,
                retry_attempts=0,
            ))
            return json.dumps(
                {"error": message, "reason": "invalid_tool_arguments"},
                ensure_ascii=False,
            )

        def reject_policy(reason: str) -> None:
            self._record(ToolCallRecord(
                tool_call_id=tool_call_id,
                tool_name=name,
                arguments=arguments,
                status="blocked",
                duration_ms=0.0,
                timestamp=datetime.now().isoformat(),
                effect_mode=effect_mode.value,
                run_id=run_id,
                user_id=user_id,
                error_message=reason,
                retry_attempts=0,
            ))
            raise ToolPolicyViolation(reason, name)

        owner_argument = tool.metadata.owner_argument
        if owner_argument:
            if user_id is None:
                reject_policy("owner_context_missing")
            if arguments.get(owner_argument) != user_id:
                reject_policy("owner_mismatch")

        reservation: tuple[str, str] | None = None
        if (
            tool.metadata.dedupe_within_run
            and effect_mode in _NON_REPLAYABLE_EFFECTS
            and run_id is not None
        ):
            try:
                reservation = self.semantic_reservation(tool, arguments)
            except ValueError:
                reject_policy("invalid_dedupe_arguments")
            run_reservations = self._run_effect_digests.setdefault(run_id, set())
            if reservation in run_reservations:
                reject_policy("duplicate_within_run")
            if len(run_reservations) >= self._POLICY_MAX_RESERVATIONS:
                reject_policy("run_policy_reservation_limit")
            # No await occurs between lookup and insertion, so concurrent tasks
            # in this event loop cannot both acquire the semantic effect marker.
            run_reservations.add(reservation)

        # The durable progress boundary belongs immediately before a validated
        # handler invocation.  Putting it in the outer tool loop would mark
        # schema/signature-invalid calls as having started a side effect even
        # though no handler can run.
        if on_before_handler is not None:
            try:
                await on_before_handler()
            except BaseException:
                if reservation is not None and run_id is not None:
                    self._release_effect_digest(run_id, reservation)
                raise

        # This marker is deliberately independent from the bounded audit LRU
        # and covers every potentially non-replayable tool, including UNKNOWN
        # tools that are not configured for semantic deduplication.
        if effect_mode in _NON_REPLAYABLE_EFFECTS and run_id is not None:
            self._run_effect_attempts.add(run_id)

        if idempotency_key and effect_mode in _NON_REPLAYABLE_EFFECTS:
            try:
                if (
                    idempotency_lease is None
                    or idempotency_lease.key != idempotency_key
                ):
                    raise IdempotencyConflictError("receipt_owner_missing")
                await request_idempotency.mark_effect_started(
                    idempotency_lease, name
                )
            except IdempotencyConflictError as exc:
                self._record(ToolCallRecord(
                    tool_call_id=tool_call_id,
                    tool_name=name,
                    arguments=arguments,
                    status="error",
                    duration_ms=0.0,
                    timestamp=datetime.now().isoformat(),
                    effect_mode=effect_mode.value,
                    run_id=run_id,
                    user_id=user_id,
                    error_message=exc.reason,
                    retry_attempts=0,
                ))
                raise SideEffectAmbiguousError(name) from exc

        start = time.monotonic()
        replay_safe = effect_mode in _REPLAY_SAFE_EFFECTS
        effective_retries = tool.metadata.max_retries if replay_safe else 0
        if not replay_safe and tool.metadata.max_retries:
            logger.warning(
                "[tool_registry] suppressing retries for non-replayable tool %s",
                name,
            )

        record = ToolCallRecord(
            tool_call_id=tool_call_id,
            tool_name=name,
            arguments=arguments,
            status="started",
            duration_ms=0.0,
            timestamp=datetime.now().isoformat(),
            effect_mode=effect_mode.value,
            run_id=run_id,
            user_id=user_id,
        )
        # 在 handler 前写 started marker。与同为进程内的 continuation session 配合，
        # 即使 trajectory 尚未 append，也能保守地阻止副作用重放。
        self._record(record)

        status = "started"
        output_preview: Optional[str] = None
        error_message: Optional[str] = None
        attempts = 0

        # 包一层用来计数 retry 次数（with_retry 内部不暴露，外层用闭包计）
        async def _wrapped():
            nonlocal attempts
            attempts += 1
            return await tool.handler(**arguments)

        try:
            result = await with_retry(
                _wrapped,
                max_retries=effective_retries,
                base_delay=tool.metadata.base_delay,
                timeout=tool.metadata.timeout_sec,
            )
            status = "ok"
            if isinstance(result, str):
                output_preview = result[:1000]
            else:
                output_preview = str(result)[:1000]
            return result
        except RetryExhausted as e:
            status = (
                "ambiguous"
                if effect_mode in _NON_REPLAYABLE_EFFECTS
                else "timeout" if "exceeded" in str(e) else "error"
            )
            cause = e.__cause__
            cause_type = type(cause).__name__ if cause is not None else type(e).__name__
            error_message = f"retry_exhausted:{cause_type}"
            logger.error(
                "[tool_registry] retries exhausted: tool=%s error_type=%s",
                name,
                cause_type,
            )
            if effect_mode in _NON_REPLAYABLE_EFFECTS:
                raise SideEffectAmbiguousError(name) from e
            raise
        except asyncio.CancelledError:
            status = (
                "ambiguous"
                if effect_mode in _NON_REPLAYABLE_EFFECTS
                else "cancelled"
            )
            error_message = "cancelled"
            raise
        except Exception as e:
            status = (
                "ambiguous"
                if effect_mode in _NON_REPLAYABLE_EFFECTS
                else "error"
            )
            error_type = type(e).__name__
            error_message = f"handler_error:{error_type}"
            logger.error(
                "[tool_registry] handler failed: tool=%s error_type=%s",
                name,
                error_type,
            )
            if effect_mode in _NON_REPLAYABLE_EFFECTS:
                raise SideEffectAmbiguousError(name) from e
            raise
        finally:
            record.duration_ms = (time.monotonic() - start) * 1000
            record.output_preview = output_preview
            record.error_message = error_message
            record.retry_attempts = attempts
            # status 最后写入：并发读取 audit 时，不会看到“已完成”却仍带旧字段。
            record.status = status

    # ── Audit 查询 ───────────────────────────────────────────────────────
    @staticmethod
    def _argument_path(arguments: dict, path: str) -> tuple[bool, object]:
        if path in {"", "$"}:
            return True, arguments
        current: object = arguments
        for segment in path.split("."):
            if isinstance(current, dict) and segment in current:
                current = current[segment]
            elif isinstance(current, list) and segment.isdigit():
                index = int(segment)
                if index >= len(current):
                    return False, None
                current = current[index]
            else:
                return False, None
        return True, current

    @classmethod
    def _semantic_arguments_payload(
        cls, arguments: dict, paths: tuple[str, ...]
    ) -> object:
        if not paths:
            return arguments
        return [
            {"path": path, "present": present, "value": value}
            for path in paths
            for present, value in [cls._argument_path(arguments, path)]
        ]

    def semantic_reservation(
        self,
        tool: Tool | str,
        arguments: dict,
    ) -> tuple[str, str]:
        """Return the stable digest-only reservation for a configured tool."""
        resolved = self._tools.get(tool) if isinstance(tool, str) else tool
        if resolved is None or self._tools.get(resolved.name) is not resolved:
            raise ValueError("tool is not registered")
        metadata = resolved.metadata
        if (
            not metadata.dedupe_within_run
            or metadata.effect_mode not in _NON_REPLAYABLE_EFFECTS
        ):
            raise ValueError("tool is not configured for semantic dedupe")
        try:
            if metadata.dedupe_normalizer is not None:
                payload = metadata.dedupe_normalizer(copy.deepcopy(arguments))
            else:
                payload = self._semantic_arguments_payload(
                    arguments,
                    metadata.dedupe_argument_paths,
                )
        except Exception as exc:
            raise ValueError("dedupe normalization failed") from exc
        if not isinstance(payload, (dict, list)):
            raise ValueError("dedupe normalizer must return an object or list")
        try:
            canonical = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("normalized dedupe payload is not canonical JSON") from exc
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return resolved.name, digest

    def _release_effect_digest(
        self,
        run_id: str,
        reservation: tuple[str, str],
    ) -> None:
        run_reservations = self._run_effect_digests.get(run_id)
        if run_reservations is None:
            return
        run_reservations.discard(reservation)
        if not run_reservations:
            self._run_effect_digests.pop(run_id, None)

    def snapshot_run_policy_state(
        self,
        run_id: str,
    ) -> tuple[tuple[str, str], ...]:
        """Return deterministic digest-only state safe for JSON persistence."""
        self._validate_policy_run_id(run_id)
        reservations = self._run_effect_digests.get(run_id, set())
        if len(reservations) > self._POLICY_MAX_RESERVATIONS:
            raise ValueError("run policy reservation limit exceeded")
        return tuple(sorted(reservations))

    def restore_run_policy_state(
        self,
        run_id: str,
        reservations: object,
    ) -> None:
        """Atomically union a strictly validated digest-only snapshot."""
        self._validate_policy_run_id(run_id)
        if not isinstance(reservations, (list, tuple)):
            raise ValueError("run policy reservations must be a sequence")
        if len(reservations) > self._POLICY_MAX_RESERVATIONS:
            raise ValueError("run policy reservation limit exceeded")

        validated: set[tuple[str, str]] = set()
        for entry in reservations:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError("invalid run policy reservation entry")
            tool_name, digest = entry
            if not isinstance(tool_name, str) or not tool_name:
                raise ValueError("invalid reservation tool name")
            if (
                len(tool_name.encode("utf-8"))
                > self._POLICY_TOOL_NAME_MAX_BYTES
            ):
                raise ValueError("reservation tool name too long")
            tool = self._tools.get(tool_name)
            if (
                tool is None
                or not tool.metadata.dedupe_within_run
                or tool.metadata.effect_mode not in _NON_REPLAYABLE_EFFECTS
            ):
                raise ValueError("reservation tool is not dedupe-enabled")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError("invalid semantic argument digest")
            reservation = (tool_name, digest)
            if reservation in validated:
                raise ValueError("duplicate reservation entry")
            validated.add(reservation)

        current = self._run_effect_digests.get(run_id, set())
        if len(current | validated) > self._POLICY_MAX_RESERVATIONS:
            raise ValueError("combined run policy reservation limit exceeded")
        self._run_effect_digests.setdefault(run_id, set()).update(validated)

    def clear_run_policy_state(self, run_id: str) -> None:
        self._run_effect_digests.pop(run_id, None)
        self._run_effect_attempts.discard(run_id)

    @staticmethod
    def _validate_policy_run_id(run_id: str) -> None:
        if (
            not isinstance(run_id, str)
            or not run_id
            or len(run_id.encode("utf-8")) > 256
        ):
            raise ValueError("invalid policy run id")

    def _record(self, record: ToolCallRecord) -> None:
        """写一条审计记录，超出上限丢最旧的（FIFO）。"""
        self._audit_log.append(record)
        if len(self._audit_log) > self._AUDIT_MAX:
            self._audit_log.pop(0)

    def get_audit(
        self,
        run_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        limit: int = 50,
    ) -> list[ToolCallRecord]:
        """按条件查 audit。多个条件 AND，倒序返回最新 limit 条。"""
        items = self._audit_log
        if run_id is not None:
            items = [r for r in items if r.run_id == run_id]
        if user_id is not None:
            items = [r for r in items if r.user_id == user_id]
        if tool_name is not None:
            items = [r for r in items if r.tool_name == tool_name]
        return items[-limit:][::-1]   # 倒序

    def has_effect_attempt(self, run_id: str) -> bool:
        """Whether a non-replayable tool started during this in-process run."""
        return run_id in self._run_effect_attempts

    def audit_summary(self) -> dict:
        """整体审计摘要：tool 调用计数、平均时长、错误率。"""
        from collections import Counter
        by_tool: dict[str, list[ToolCallRecord]] = {}
        for r in self._audit_log:
            by_tool.setdefault(r.tool_name, []).append(r)
        summary = {}
        for name, records in by_tool.items():
            durations = [r.duration_ms for r in records]
            statuses = Counter(r.status for r in records)
            summary[name] = {
                "calls": len(records),
                "avg_duration_ms": round(sum(durations) / len(durations), 1),
                "p95_duration_ms": round(sorted(durations)[int(len(durations) * 0.95)] if len(durations) >= 20 else max(durations), 1),
                "status_counts": dict(statuses),
                "error_rate": round((
                    statuses.get("error", 0)
                    + statuses.get("timeout", 0)
                    + statuses.get("ambiguous", 0)
                    + statuses.get("cancelled", 0)
                ) / len(records), 3),
            }
        return {
            "total_calls": len(self._audit_log),
            "by_tool": summary,
        }


# 全局单例
tool_registry = ToolRegistry()
