"""
可复用的单轮 tool-calling（D4）

chat（N≤3）和 autonomous（N≤8 + 控制工具）此前各写一份「LLM→tool_calls→dispatch→回灌」
循环，内容高度重复且白名单各硬编码一份。这里抽出单轮逻辑 run_tool_round：

  一轮 = 调一次 LLM → 若有 tool_calls 则逐个：
         白名单内业务工具 → dispatch_tool 回灌 tool message；
         白名单外 → 回灌错误 message（不 dispatch）；
         autonomous 的控制工具（finalize/ask_user）交回调用方处理（见 control_tools）。

调用方拿到 ToolRoundResult 后自己决定：继续下一轮 / 结束 / 暂停。
这样既消除重复，又不把 autonomous 专有的 finalize/ask_user 语义塞进 chat。

white-list 单一数据源 = services.tools.allowed_tool_names()（registry 派生）。
LLM 调用统一走 services.llm.llm_chat（带 retry 横切）。
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from services.llm import llm_chat
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
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tool_choice: Optional[str] = None,
    max_retries: int = 2,
    extra_call_messages: Optional[list] = None,
    **llm_kwargs,
) -> ToolRoundResult:
    """跑单轮 tool-calling 并把结果回灌进 messages（原地 append）。

    Args:
        messages: OpenAI messages，会被原地追加 assistant / tool 消息。
        tools: 传给 LLM 的 tools schema（业务 + 可选控制工具）。
        client: 注入的 AsyncOpenAI（默认用 llm 模块级 client）。
        control_tools: 控制工具名集合（如 {"finalize","ask_user"}）。命中时
                       记为 kind="control" 交回调用方处理，本函数不 dispatch、
                       不追加 tool message（由调用方决定如何补全协议）。
        tool_choice: 透传给 LLM（"auto" 等）。None 则不传。
        extra_call_messages: 仅用于本次 LLM 调用、不持久化进 messages 的临时消息
                       （如 autonomous 每轮注入的 [Current state] 摘要）。
        其余 llm_kwargs 透传给 llm_chat（temperature/max_tokens...）。

    Returns:
        ToolRoundResult。调用方据此决定继续 / 结束 / 暂停。
    """
    control_tools = control_tools or set()
    allowed = allowed_tool_names()

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

    outcomes: list[ToolCallOutcome] = []
    for tc in msg.tool_calls:
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError as e:
            logger.warning(f"[tool_loop] bad tool args for {name}: {e}")
            args = {}

        # 控制工具：交回调用方处理（不 dispatch、不补 tool message）
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

        # 业务工具：dispatch（自带超时/重试/audit）并回灌
        logger.info(f"[tool_loop] dispatch {name}({args})")
        try:
            result = await dispatch_tool(name, args, run_id=run_id, user_id=user_id)
            blocked_reason = None
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
