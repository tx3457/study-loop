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
import json
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from services.llm import llm_chat
from services.tool_registry import EffectMode, SideEffectAmbiguousError, tool_registry
from services.tools import allowed_tool_names, dispatch_tool

logger = logging.getLogger(__name__)


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
    tool_choice: Optional[str] = None,
    max_retries: int = 2,
    extra_call_messages: Optional[list] = None,
    on_before_tool_calls: Callable[[], Awaitable[None]] | None = None,
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
                       持久化屏障。用于 interrupt 续跑路径记录不可重放进展。
        其余 llm_kwargs 透传给 llm_chat（temperature/max_tokens...）。

    Returns:
        ToolRoundResult。调用方据此决定继续 / 结束 / 暂停。
    """
    control_tools = control_tools or set()
    allowed = (
        allowed_tool_names()
        if business_tool_allowlist is None
        else set(business_tool_allowlist)
    )
    replay_safe_bindings = None
    if business_tool_allowlist is not None:
        safe_modes = {EffectMode.READ_ONLY, EffectMode.IDEMPOTENT}
        replay_safe_bindings = {}
        for name in allowed:
            tool = tool_registry.get(name)
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

    if on_before_tool_calls is not None:
        await on_before_tool_calls()

    # 追加 assistant 消息（含 tool_calls），保证 OpenAI 协议完整
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

    parsed_calls = []
    for tc in msg.tool_calls:
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError as e:
            logger.warning(f"[tool_loop] bad tool args for {name}: {e}")
            args = {}
        parsed_calls.append((tc, name, args))

    outcomes: list[ToolCallOutcome] = []

    # OpenAI 的同轮 tool_calls 是一个并行决策批次，数组顺序不表示执行依赖。
    # control 与任何其他调用混用时整批拒绝，避免把 [write, ask_user/finalize]
    # 误解为“先写后暂停/结束”，也避免最终回答声称未实际发生的副作用。
    has_control = any(name in control_tools for _, name, _ in parsed_calls)
    if has_control and len(parsed_calls) > 1:
        reason = "mixed_control_batch_rejected"
        logger.warning("[tool_loop] rejected multi-call batch containing control tool")
        for tc, name, args in parsed_calls:
            result = json.dumps({
                "error": "控制工具必须单独调用，本轮所有工具均未执行",
                "reason": reason,
            }, ensure_ascii=False)
            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": result,
            })
            outcomes.append(ToolCallOutcome(
                call_id=tc.id,
                name=name,
                arguments=args,
                kind="blocked",
                result=result,
                blocked_reason=reason,
            ))
        return ToolRoundResult(
            assistant_message=msg,
            has_tool_calls=True,
            outcomes=outcomes,
        )

    for tc, name, args in parsed_calls:
        # 单独出现的控制工具交回调用方处理（不 dispatch，由调用方补 tool message）
        if name in control_tools:
            outcomes.append(ToolCallOutcome(
                call_id=tc.id, name=name, arguments=args, kind="control",
            ))
            continue

        # 白名单拦截：不 dispatch，回灌错误 message
        if name not in allowed:
            logger.warning(f"[tool_loop] blocked tool: {name}")
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": json.dumps({"error": f"工具 {name} 不在允许列表"}, ensure_ascii=False),
            })
            outcomes.append(ToolCallOutcome(
                call_id=tc.id, name=name, arguments=args,
                kind="blocked", blocked_reason="not_in_whitelist",
            ))
            continue

        # interrupt-capable 调用方在 LLM await 前绑定 Tool 对象、handler 与
        # effect_mode。若等待期间同名工具被覆盖或原对象被改写，拒绝 dispatch，
        # 避免仅凭名称白名单执行新的副作用实现。
        if replay_safe_bindings is not None:
            binding = replay_safe_bindings.get(name)
            current = tool_registry.get(name)
            if (
                binding is None
                or current is not binding[0]
                or current.handler is not binding[1]
                or current.metadata.effect_mode is not binding[2]
            ):
                reason = "replay_safety_binding_changed"
                logger.warning(f"[tool_loop] blocked changed tool binding: {name}")
                result = json.dumps({
                    "error": f"工具 {name} 的安全绑定已变化，本轮未执行",
                    "reason": reason,
                }, ensure_ascii=False)
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": result,
                })
                outcomes.append(ToolCallOutcome(
                    call_id=tc.id,
                    name=name,
                    arguments=args,
                    kind="blocked",
                    result=result,
                    blocked_reason=reason,
                ))
                continue

        # 业务工具：dispatch（自带超时/重试/audit）并回灌
        logger.info(f"[tool_loop] dispatch {name}({args})")
        try:
            result = await dispatch_tool(
                name,
                args,
                run_id=run_id,
                user_id=user_id,
                idempotency_key=idempotency_key,
            )
            blocked_reason = None
        except SideEffectAmbiguousError:
            logger.exception(f"[tool_loop] side effect result ambiguous for {name}")
            raise
        except Exception as e:
            logger.exception(f"[tool_loop] dispatch failed for {name}: {e}")
            result = json.dumps({"error": str(e)}, ensure_ascii=False)
            blocked_reason = f"dispatch_exception: {type(e).__name__}"
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        outcomes.append(ToolCallOutcome(
            call_id=tc.id, name=name, arguments=args,
            kind="dispatched", result=result, blocked_reason=blocked_reason,
        ))

    return ToolRoundResult(assistant_message=msg, has_tool_calls=True, outcomes=outcomes)
