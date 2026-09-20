"""
可复用的单轮 tool-calling

run_tool_round 供 chat（N≤3）和 autonomous（N≤8 + 控制工具）共同调用：

  一轮 = 调一次 LLM → 若有 tool_calls 则逐个：
         白名单内业务工具 → dispatch_tool 回灌 tool message；
         白名单外 → 回灌错误 message（不 dispatch）；
         autonomous 的控制工具（finalize/ask_user）必须单独成批，交回调用方处理
         （见 control_tools）。

调用方拿到 ToolRoundResult 后自行决定继续下一轮、结束或暂停；
autonomous 专有的 finalize/ask_user 语义仍由调用方处理。

white-list 默认来自 services.tools.allowed_tool_names()（registry 派生）；
interrupt-capable 调用方可传更窄的 business_tool_allowlist。
LLM 调用统一走 services.llm.llm_chat（带 retry 横切）。
"""
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from services.llm import llm_chat
from services.idempotency import IdempotencyConflictError
from services.tool_registry import (
    EffectMode,
    SideEffectAmbiguousError,
    ToolPolicyViolation,
    ToolRegistry,
    tool_registry,
)
from services.tools import allowed_tool_names, dispatch_tool

logger = logging.getLogger(__name__)

_MISSING = object()


def _field(value, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _path_parts(path: str) -> list[str]:
    if path == "$":
        return []
    normalized = path[2:] if path.startswith("$.") else path
    return [part for part in normalized.split(".") if part]


def _path_get(value, path: str):
    current = value
    for part in _path_parts(path):
        if isinstance(current, dict):
            if part not in current:
                return _MISSING
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return _MISSING
            current = current[index]
        else:
            return _MISSING
    return current


def _path_set(target: dict, path: str, value) -> bool:
    parts = _path_parts(path)
    if not parts:
        return False
    current = target
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = copy.deepcopy(value)
    return True


def _successful_tool_results(messages: list) -> dict[str, object]:
    """Resolve successful JSON results in transcript order, failing closed."""
    call_tools: dict[str, str] = {}
    invalid_call_ids: set[str] = set()
    results: dict[str, object] = {}
    for message in messages:
        role = _field(message, "role")
        if role == "assistant":
            for call in _field(message, "tool_calls", []) or []:
                call_id = _field(call, "id")
                function = _field(call, "function", {})
                name = _field(function, "name")
                if not call_id or not name:
                    continue
                if call_id in call_tools or call_id in invalid_call_ids:
                    previous_name = call_tools.pop(call_id, None)
                    invalid_call_ids.add(call_id)
                    if previous_name:
                        results.pop(previous_name, None)
                    results.pop(name, None)
                else:
                    call_tools[call_id] = name
            continue
        if role != "tool":
            continue

        name = call_tools.get(_field(message, "tool_call_id"))
        if not name:
            continue
        content = _field(message, "content")
        if not isinstance(content, str):
            results.pop(name, None)
            continue
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            results.pop(name, None)
            continue
        if isinstance(parsed, dict) and "error" in parsed:
            results.pop(name, None)
            continue
        results[name] = copy.deepcopy(parsed)
    return results


def _apply_argument_bindings(args: dict, metadata, results: dict[str, object]) -> bool:
    """Apply authoritative bindings; the first conflicting target path wins."""
    bound_paths: list[tuple[str, ...]] = []
    for binding in metadata.argument_bindings:
        source = results.get(binding.source_tool, _MISSING)
        if source is _MISSING:
            continue
        value = _path_get(source, binding.source_path)
        if value is _MISSING:
            continue
        target_parts = tuple(_path_parts(binding.target_argument))
        if not target_parts or any(
            target_parts[:len(bound)] == bound
            or bound[:len(target_parts)] == target_parts
            for bound in bound_paths
        ):
            continue
        if _path_set(args, binding.target_argument, value):
            bound_paths.append(target_parts)
    return bool(bound_paths)


@dataclass
class ToolCallOutcome:
    """单个 tool_call 的处理结果，供调用方记录 trajectory / 决策。"""
    call_id: str
    name: str
    arguments: dict
    # control / dispatched / blocked / bad_args
    kind: str
    result: Optional[str] = None         # dispatched 时的工具输出
    blocked_reason: Optional[str] = None


@dataclass
class ToolRoundResult:
    """一轮的结果。"""
    assistant_message: object                    # LLM 返回的 message 对象
    has_tool_calls: bool
    content: Optional[str] = None                # 无 tool_calls 时的纯文字
    outcomes: list[ToolCallOutcome] = field(default_factory=list)


def _mark_untrusted_observation(run_id: str, result: str, *, registry=None) -> None:
    """Taint the run when a tool result flags suspicious retrieved content.

    Parsing is best-effort: a tool may legitimately return non-JSON, and a
    malformed payload must never break the dispatch loop.
    """
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return
    if isinstance(payload, dict) and payload.get("injection_flagged") is True:
        logger.warning(
            "[tool_loop] untrusted content flagged; blocking further writes in this run"
        )
        (registry if registry is not None else tool_registry).mark_run_untrusted_content(run_id)


def _block_tool_call(
    messages: list,
    outcomes: list,
    *,
    call_id: str,
    name: str,
    arguments: dict,
    error: str,
    reason: str,
) -> None:
    """记录一次被拒绝的工具调用。

    transcript 和 outcomes 必须一起写：只写 outcomes，模型下一轮看不到自己
    被拒绝，会原样重试同一个调用；只写 messages，调用方拿不到 blocked_reason。
    两边的 reason 也必须是同一个值，否则审计记录与模型看到的理由对不上。
    这个函数存在就是为了让上面三件事不可能被写漏。
    """
    result = json.dumps({"error": error, "reason": reason}, ensure_ascii=False)
    messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
    outcomes.append(ToolCallOutcome(
        call_id=call_id,
        name=name,
        arguments=arguments,
        kind="blocked",
        result=result,
        blocked_reason=reason,
    ))


async def run_tool_round(
    messages: list,
    *,
    tools: list,
    client=None,
    control_tools: Optional[set[str]] = None,
    business_tool_allowlist: Optional[set[str]] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    idempotency_lease=None,
    tool_choice: Optional[str] = None,
    max_retries: int = 2,
    extra_call_messages: Optional[list] = None,
    on_before_tool_calls: Callable[[], Awaitable[None]] | None = None,
    on_before_tool_dispatch: Callable[[], Awaitable[None]] | None = None,
    business_tool_guard: Callable[[str, dict], Optional[str]] | None = None,
    registry: ToolRegistry | None = None,
    **llm_kwargs,
) -> ToolRoundResult:
    """跑单轮 tool-calling 并把结果回灌进 messages（原地 append）。

    Args:
        messages: OpenAI messages，会被原地追加 assistant / tool 消息。
        tools: 传给 LLM 的 tools schema（业务 + 可选控制工具）。
        client: 注入的 AsyncOpenAI（默认用 llm 模块级 client）。
        control_tools: 控制工具名集合（如 {"finalize","ask_user"}）。命中时
                       记为 kind="control" 交回调用方处理，本函数不 dispatch、
                       不追加 tool message（由调用方决定如何补全协议）。若同一批
                       还有其他调用，则整批拒绝，让模型下一轮重新选择单一动作。
        business_tool_allowlist: 可选的业务工具白名单；不传时使用全局 registry。
                       interrupt-capable 节点应只传可安全重放的工具集合。
        tool_choice: 透传给 LLM（"auto" 等）。None 则不传。
        extra_call_messages: 仅用于本次 LLM 调用、不持久化进 messages 的临时消息
                       （如 autonomous 每轮注入的 [Current state] 摘要）。
        on_before_tool_calls: provider 返回工具调用后、修改消息或执行工具前的
                        ownership 续租/围栏；不会把调用标记为已开始副作用。
        on_before_tool_dispatch: 参数校验成功后、每个实际 handler 调用前的
                        ownership/progress 屏障；多工具批次和长轮次用它重新验证
                        fencing token。无效参数不会跨过该边界。
        business_tool_guard: 可选的调用级范围检查。返回 reason 时，本轮工具调用
                        会在 dispatch 前被拒绝并把结构化错误回灌给模型。
        其余 llm_kwargs 透传给 llm_chat（temperature/max_tokens...）。

    Returns:
        ToolRoundResult。调用方据此决定继续 / 结束 / 暂停。
    """
    control_tools = control_tools or set()
    active_registry = registry if registry is not None else tool_registry
    allowed = (
        (allowed_tool_names() if registry is None else set(active_registry.list_tools()))
        if business_tool_allowlist is None
        else set(business_tool_allowlist)
    )
    # Snapshot lineage before the provider request. Current-batch tool results
    # must never become implicit inputs to later calls in that parallel batch.
    prior_tool_results = _successful_tool_results(list(messages))
    replay_safe_bindings = None
    if business_tool_allowlist is not None:
        safe_modes = {EffectMode.READ_ONLY, EffectMode.IDEMPOTENT}
        replay_safe_bindings = {}
        for name in allowed:
            tool = active_registry.get(name)
            if tool is not None and tool.metadata.effect_mode in safe_modes:
                replay_safe_bindings[name] = (
                    tool,
                    tool.handler,
                    tool.metadata.effect_mode,
                )

    kwargs = dict(llm_kwargs)
    kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    call_messages = list(messages) + list(extra_call_messages) if extra_call_messages else messages
    response = await llm_chat(call_messages, client=client, max_retries=max_retries, **kwargs)
    msg = response.choices[0].message

    if not msg.tool_calls:
        return ToolRoundResult(
            assistant_message=msg, has_tool_calls=False, content=msg.content or "",
        )

    current_call_ids = [_field(tc, "id") for tc in msg.tool_calls]
    if (
        any(
            not isinstance(call_id, str) or not call_id
            for call_id in current_call_ids
        )
        or len(current_call_ids) != len(set(current_call_ids))
    ):
        reason = "duplicate_tool_call_id"
        result = json.dumps({
            "error": "同一批工具调用必须使用唯一且非空的 call_id",
            "reason": reason,
        }, ensure_ascii=False)
        logger.warning("[tool_loop] rejected batch with invalid tool call ids")
        return ToolRoundResult(
            assistant_message=msg,
            has_tool_calls=True,
            outcomes=[
                ToolCallOutcome(
                    call_id=(
                        _field(tc, "id")
                        if isinstance(_field(tc, "id"), str)
                        else ""
                    ),
                    name=_field(_field(tc, "function", {}), "name", ""),
                    arguments={},
                    kind="blocked",
                    result=result,
                    blocked_reason=reason,
                )
                for tc in msg.tool_calls
            ],
        )

    parsed_calls = []
    for tc in msg.tool_calls:
        name = tc.function.name
        args_error = False
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(
                "[tool_loop] invalid JSON tool arguments: error_type=%s",
                type(e).__name__,
            )
            args = {}
            args_error = True
        if not isinstance(args, dict):
            logger.warning("[tool_loop] non-object tool arguments blocked")
            args = {}
            args_error = True
        parsed_calls.append((tc, name, args, args_error, None, None))

    outcomes: list[ToolCallOutcome] = []

    # OpenAI 的同轮 tool_calls 是一个并行决策批次，数组顺序不表示执行依赖。
    # control 与任何其他调用混用时整批拒绝，避免把 [write, ask_user/finalize]
    # 误解为“先写后暂停/结束”，也避免最终回答声称未实际发生的副作用。
    has_control = any(
        name in control_tools for _, name, _, _, _, _ in parsed_calls
    )
    guarded_calls = []
    for tc, name, args, args_error, _, _ in parsed_calls:
        binding_reason = None
        metadata_tool = None
        bindings_applied = False
        can_bind = (
            not args_error
            and name not in control_tools
            and name in allowed
        )
        if can_bind and replay_safe_bindings is not None:
            replay_binding = replay_safe_bindings.get(name)
            current = active_registry.get(name)
            if (
                replay_binding is None
                or current is not replay_binding[0]
                or current.handler is not replay_binding[1]
                or current.metadata.effect_mode is not replay_binding[2]
            ):
                binding_reason = "replay_safety_binding_changed"
            else:
                metadata_tool = replay_binding[0]
        elif can_bind:
            metadata_tool = active_registry.get(name)

        guard_reason = None
        can_reach_guard = (
            not (has_control and len(parsed_calls) > 1)
            and not args_error
            and name not in control_tools
            and name in allowed
            and binding_reason is None
            and business_tool_guard is not None
        )
        if can_reach_guard:
            try:
                # Check the model-authored scope before authoritative bindings
                # can replace an attempted cross-user/document argument.
                guard_reason = business_tool_guard(
                    name,
                    copy.deepcopy(args),
                )
            except Exception as exc:
                logger.warning(
                    "[tool_loop] business tool guard failed closed: "
                    "tool=%s error_type=%s",
                    name,
                    type(exc).__name__,
                )
                guard_reason = "business_tool_guard_error"
        if guard_reason is None and metadata_tool is not None:
            bindings_applied = _apply_argument_bindings(
                args,
                metadata_tool.metadata,
                prior_tool_results,
            )
        if can_reach_guard and guard_reason is None and bindings_applied:
            try:
                # Re-check the effective server-bound arguments so neither the
                # model nor transcript lineage can bypass request scope.
                guard_reason = business_tool_guard(
                    name,
                    copy.deepcopy(args),
                )
            except Exception as exc:
                logger.warning(
                    "[tool_loop] effective business tool guard failed closed: "
                    "tool=%s error_type=%s",
                    name,
                    type(exc).__name__,
                )
                guard_reason = "business_tool_guard_error"
        guarded_calls.append((
            tc,
            name,
            args,
            args_error,
            guard_reason,
            binding_reason,
        ))
    parsed_calls = guarded_calls
    dispatch_may_start = (
        not (has_control and len(parsed_calls) > 1)
        and any(
            not args_error
            and name not in control_tools
            and name in allowed
            and guard_reason is None
            and binding_reason is None
            for _, name, _, args_error, guard_reason, binding_reason in parsed_calls
        )
    )
    if dispatch_may_start and on_before_tool_calls is not None:
        await on_before_tool_calls()

    # Only after the durable dispatch barrier has succeeded may the in-memory
    # protocol advance to the assistant tool-call message.
    messages.append({
        "role": "assistant",
        "content": msg.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ],
    })

    if has_control and len(parsed_calls) > 1:
        reason = "mixed_control_batch_rejected"
        logger.warning("[tool_loop] rejected multi-call batch containing control tool")
        for tc, name, args, _, _, _ in parsed_calls:
            _block_tool_call(
                messages, outcomes,
                call_id=tc.id, name=name, arguments=args,
                error="控制工具必须单独调用，本轮所有工具均未执行",
                reason=reason,
            )
        return ToolRoundResult(
            assistant_message=msg,
            has_tool_calls=True,
            outcomes=outcomes,
        )

    for (
        tc,
        name,
        args,
        args_error,
        precomputed_guard_reason,
        precomputed_binding_reason,
    ) in parsed_calls:
        if args_error:
            reason = "invalid_tool_arguments"
            _block_tool_call(
                messages, outcomes,
                call_id=tc.id, name=name, arguments={},
                error="工具参数必须是 JSON 对象",
                reason=reason,
            )
            continue

        # 单独出现的控制工具交回调用方处理（不 dispatch，由调用方补 tool message）
        if name in control_tools:
            outcomes.append(ToolCallOutcome(
                call_id=tc.id, name=name, arguments=args, kind="control",
            ))
            continue

        # 白名单拦截：不 dispatch，回灌错误 message
        if name not in allowed:
            logger.warning("[tool_loop] blocked unknown tool")
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": json.dumps({"error": "工具不在允许列表"}, ensure_ascii=False),
            })
            outcomes.append(ToolCallOutcome(
                call_id=tc.id, name=name, arguments=args,
                kind="blocked", blocked_reason="not_in_whitelist",
            ))
            continue

        if precomputed_binding_reason:
            _block_tool_call(
                messages, outcomes,
                call_id=tc.id, name=name, arguments=args,
                error=f"工具 {name} 的安全绑定已变化，本轮未执行",
                reason=precomputed_binding_reason,
            )
            continue

        # 调用级范围约束必须发生在 dispatch 前，避免模型先看到越界工具输出，
        # 再由上层在响应阶段被动丢弃。
        if business_tool_guard is not None:
            guard_reason = precomputed_guard_reason
            if guard_reason:
                _block_tool_call(
                    messages, outcomes,
                    call_id=tc.id, name=name, arguments=args,
                    error="工具调用超出当前请求允许范围，本轮未执行",
                    reason=guard_reason,
                )
                continue

        # interrupt-capable 调用方在 LLM await 前绑定 Tool 对象、handler 与
        # effect_mode。若等待期间同名工具被覆盖或原对象被改写，拒绝 dispatch，
        # 避免仅凭名称白名单执行新的副作用实现。
        if replay_safe_bindings is not None:
            binding = replay_safe_bindings.get(name)
            current = active_registry.get(name)
            if (
                binding is None
                or current is not binding[0]
                or current.handler is not binding[1]
                or current.metadata.effect_mode is not binding[2]
            ):
                reason = "replay_safety_binding_changed"
                logger.warning(f"[tool_loop] blocked changed tool binding: {name}")
                _block_tool_call(
                    messages, outcomes,
                    call_id=tc.id, name=name, arguments=args,
                    error=f"工具 {name} 的安全绑定已变化，本轮未执行",
                    reason=reason,
                )
                continue

        # 业务工具：dispatch（自带参数校验/超时/重试/audit）并回灌。真正的
        # progress 屏障由 registry 在参数与 handler 签名校验通过后调用。
        logger.info(
            "[tool_loop] dispatch tool=%s arg_count=%d",
            name,
            len(args),
        )
        try:
            dispatcher = dispatch_tool if registry is None else active_registry.invoke
            result = await dispatcher(
                name,
                args,
                run_id=run_id,
                user_id=user_id,
                idempotency_key=idempotency_key,
                idempotency_lease=idempotency_lease,
                on_before_handler=on_before_tool_dispatch,
            )
            blocked_reason = None
            # observation 即将被回灌进 messages。若工具报告这次取回的是可疑的
            # 不可信正文，就在回灌之前给本次 run 打 taint：之后模型的任何决策
            # 都可能是被那段正文操纵的，注册表据此拒绝非幂等写入。
            if run_id is not None:
                _mark_untrusted_observation(run_id, result, registry=active_registry)
        except IdempotencyConflictError:
            # Ownership loss is a request-level fencing event, not a tool
            # result that may be fed back to the model and ignored.
            raise
        except ToolPolicyViolation as exc:
            logger.warning(
                "[tool_loop] policy blocked: tool=%s reason=%s",
                name,
                exc.reason,
            )
            _block_tool_call(
                messages, outcomes,
                call_id=tc.id, name=name, arguments=args,
                error="工具调用违反安全策略",
                reason=exc.reason,
            )
            continue
        except SideEffectAmbiguousError:
            logger.warning(
                "[tool_loop] side effect result ambiguous: tool=%s", name
            )
            raise
        except Exception as e:
            logger.warning(
                "[tool_loop] dispatch failed: tool=%s error_type=%s",
                name,
                type(e).__name__,
            )
            result = json.dumps({
                "error": "工具执行失败",
                "error_type": type(e).__name__,
            }, ensure_ascii=False)
            blocked_reason = f"dispatch_exception: {type(e).__name__}"
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        outcomes.append(ToolCallOutcome(
            call_id=tc.id, name=name, arguments=args,
            kind="dispatched", result=result, blocked_reason=blocked_reason,
        ))

    return ToolRoundResult(assistant_message=msg, has_tool_calls=True, outcomes=outcomes)
