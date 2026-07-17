"""
Assistant Worker：把 autonomous.py 的 ReAct 自由问答能力收编为 supervisor 麾下的图节点（Phase 4）

和 routers/autonomous.py 的关系（autonomous.py 源码不改，灰度并存）：
  - autonomous.py：独立端点，HITL 用内存 _sessions dict + /continue 两段式 HTTP
  - assistant_agent.py：作为 tutor_graph 的 assistant 节点，HITL 改用 LangGraph interrupt
    （checkpointer 持久化中断点，崩溃可恢复），由 supervisor 在 mode="assist" 时调度。

复用而非重写：
  - run_tool_round（services.tool_loop）：单轮 LLM→tool_calls→dispatch→回灌（白名单/重试/audit）
  - get_tool_definitions / allowed_tool_names（services.tools）：业务工具单一数据源
  - _build_state_summary 思想（防健忘）：每轮注入 [Current state] 摘要

ReAct 协议（与 autonomous 一致）：
  - finalize(final_answer, reason)：LLM 主动声明结束并给最终答案 → 写 final_answer 回 state
  - ask_user(question)：LLM 缺关键信息 → interrupt 暂停，resume 注入 user_reply 续跑

interrupt 重放语义（langgraph：节点从头重跑，已解决的 interrupt 按序返回缓存值）：
  messages 持久化进 TutorState；节点每次进入从 state["messages"] 重建循环。
  ask_user 命中 → interrupt({"question"})；resume 时 interrupt 返回 user_reply，
  作为该 ask_user tool_call 的 tool response 回灌 messages 后续跑。
  该节点只向模型暴露 READ_ONLY/IDEMPOTENT 工具，并把同一集合传给 dispatch
  allowlist；未知 MCP 与画像写入不会进入可重放的 interrupt 路径。若未来开放
  副作用工具，仍需先提供持久幂等键，不能只依赖进程内 audit。
"""
import logging
import os
import uuid

from langgraph.types import interrupt

from agents.state import TutorState
from services.injection import check_output_leak
from services.memory_context import build_memory_context_block, build_profile_card
from services.react_controls import (
    CONTROL_TOOL_NAMES,
    build_control_tools,
    build_react_decision_prompt,
    build_react_system_prompt,
)
from services.tool_loop import run_tool_round
from services.tool_registry import SideEffectAmbiguousError
from services.tools import (
    get_replay_safe_tool_definitions,
    replay_safe_tool_names,
)

logger = logging.getLogger(__name__)

# ReAct 轮次上限（对齐 autonomous.MAX_AUTONOMOUS_ROUNDS，防 LLM 失控）
MAX_ASSIST_ROUNDS = 8


_CONTROL_TOOLS = build_control_tools(ask_user_resume_hint="调用后循环会暂停，等用户回答后续跑。")
_ASSIST_SYSTEM = build_react_system_prompt(
    include_review_loop=True,
    replay_safe_only=True,
)


def _client_of(state: TutorState):
    """测试可经 state["_client"] 注入 mock；默认 None 用 llm 模块级 client。"""
    return state.get("_client")


def _build_state_summary(tools_called: list[str], round_idx: int) -> str:
    """浓缩当前进度，让 LLM 一眼看清（对齐 autonomous._build_state_summary 思想）。"""
    used = list(dict.fromkeys(tools_called))
    remaining = [t for t in (replay_safe_tool_names() - set(used))]
    return (
        f"轮次：{round_idx + 1}/{MAX_ASSIST_ROUNDS}\n"
        f"已调用工具（按顺序）：{used or '无'}\n"
        f"可用业务工具：{remaining or '无'}"
    )


def _initial_messages(state: TutorState) -> list:
    """首次进入构造初始 messages：system + 用户 query（goal/description）。"""
    query = (state.get("goal") or state.get("description") or "").strip()
    user_id = state.get("user_id", "default")
    document_id = state.get("document_id")
    context_hint = f"\n\n当前用户 ID: {user_id}"
    if document_id:
        context_hint += f"\n当前文档 ID: {document_id}"
    return [
        {"role": "system", "content": _ASSIST_SYSTEM + context_hint},
        {"role": "user", "content": query},
    ]


async def assistant_agent(state: TutorState) -> dict:
    """assistant 节点：ReAct 多轮自由问答 worker。

    输入（读 TutorState）：goal/description(query)、user_id、document_id、messages、tools_called。
    输出（写回 state）：final_answer、messages（持久化供 interrupt 重放/续跑）、tools_called、
      assistant_done（finalize/截断时 True，供 supervisor 收尾）。
    ask_user → interrupt({"question"}) 暂停；resume 注入 user_reply 续跑。
    """
    messages: list = list(state.get("messages") or [])
    if not messages:
        messages = _initial_messages(state)
        # 注入跨会话画像卡（栅栏防注入）：assist 模式不走 diagnostic，这里现场构建，让 assistant 也"记得你"
        try:
            block = build_memory_context_block(await build_profile_card(state.get("user_id", "")))
            if block:
                messages.insert(1, {"role": "system", "content": block})
        except SideEffectAmbiguousError:
            logger.exception("[assistant_agent] side-effect result is ambiguous")
            raise
        except Exception as e:
            logger.warning(f"[assistant_agent] 注入画像卡失败（忽略）: {e}")

    tools_called: list[str] = list(state.get("tools_called") or [])
    client = _client_of(state)
    run_id = state.get("thread_id") or f"assist_{uuid.uuid4().hex[:12]}"

    for round_idx in range(MAX_ASSIST_ROUNDS):
        # 每轮注入 [Current state] 摘要防健忘（仅本次调用，不持久化）
        state_msg = [{
            "role": "system",
            "content": build_react_decision_prompt(_build_state_summary(tools_called, round_idx)),
        }]
        try:
            rr = await run_tool_round(
                messages,
                tools=get_replay_safe_tool_definitions() + _CONTROL_TOOLS,
                client=client,
                control_tools=CONTROL_TOOL_NAMES,
                business_tool_allowlist=replay_safe_tool_names(),
                run_id=run_id,
                user_id=state.get("user_id"),
                tool_choice="auto",
                extra_call_messages=state_msg,
            )
        except SideEffectAmbiguousError:
            logger.exception("[assistant_agent] side-effect result is ambiguous")
            raise
        except Exception as e:
            logger.exception(f"[assistant_agent] LLM call failed round {round_idx}: {e}")
            return {
                "final_answer": f"助理调用失败：{e}",
                "messages": messages, "tools_called": list(dict.fromkeys(tools_called)),
                "assistant_done": True,
            }

        # 无 tool_calls：LLM 直接给文字（隐式 finalize）
        if not rr.has_tool_calls:
            final_answer = rr.content or "（助理未给出回复且未调用工具，结束）"
            is_leak, reason = check_output_leak(final_answer)
            if is_leak:
                logger.warning(f"[assistant_agent] output leak blocked: {reason}")
                final_answer = "输出包含敏感信息已拦截。"
            return {
                "final_answer": final_answer, "messages": messages,
                "tools_called": list(dict.fromkeys(tools_called)), "assistant_done": True,
            }

        # 逐个处理本轮 outcomes
        for oc in rr.outcomes:
            fn_name, fn_args = oc.name, oc.arguments

            # 控制工具：finalize → 写 final_answer 回 state，收尾
            if oc.kind == "control" and fn_name == "finalize":
                final_answer = fn_args.get("final_answer", "")
                is_leak, reason = check_output_leak(final_answer)
                if is_leak:
                    logger.warning(f"[assistant_agent] output leak in finalize: {reason}")
                    final_answer = "输出包含敏感信息已拦截。"
                # 补 tool message 让 OpenAI 协议完整（每个 tool_call 都要有 response）
                messages.append({"role": "tool", "tool_call_id": oc.call_id, "content": "Acknowledged."})
                logger.info(f"[assistant_agent] finalize: {fn_args.get('reason', '')[:50]}")
                return {
                    "final_answer": final_answer, "messages": messages,
                    "tools_called": list(dict.fromkeys(tools_called)), "assistant_done": True,
                }

            # 控制工具：ask_user → interrupt 暂停，resume 注入 user_reply 续跑
            if oc.kind == "control" and fn_name == "ask_user":
                question = fn_args.get("question", "请提供更多信息。")
                logger.info(f"[assistant_agent] ask_user interrupt: {question[:50]}")
                # interrupt 暂停：把问题暴露给前端；resume 时返回 Command(resume=user_reply)
                user_reply = interrupt({"question": question, "kind": "ask_user"})
                # 恢复后：把 user_reply 作为 ask_user 的 tool response 回灌，续跑下一轮
                messages.append({
                    "role": "tool", "tool_call_id": oc.call_id,
                    "content": f"User replied: {user_reply}",
                })
                continue

            # 业务工具被白名单拦截（run_tool_round 已回灌错误 message）
            if oc.kind == "blocked":
                continue

            # 业务工具已 dispatch（run_tool_round 已回灌 tool message）
            tools_called.append(fn_name)

    # 达 MAX_ASSIST_ROUNDS：强制收尾（让 LLM 基于已有 observation 给最终答案）
    logger.warning(f"[assistant_agent] truncated at {MAX_ASSIST_ROUNDS} rounds")
    messages.append({
        "role": "user",
        "content": "已达到最大执行轮次。请基于已有信息给出最终回答（直接文字，无需调工具）。",
    })
    try:
        rr = await run_tool_round(
            messages,
            tools=get_replay_safe_tool_definitions(),
            client=client,
            business_tool_allowlist=replay_safe_tool_names(),
            run_id=run_id,
        )
        final_answer = rr.content or "执行被截断。"
    except SideEffectAmbiguousError:
        logger.exception("[assistant_agent] truncated finish has ambiguous side effect")
        raise
    except Exception as e:
        logger.exception(f"[assistant_agent] finish call failed: {e}")
        final_answer = f"执行被截断（{MAX_ASSIST_ROUNDS} 轮）"
    return {
        "final_answer": final_answer, "messages": messages,
        "tools_called": list(dict.fromkeys(tools_called)), "assistant_done": True,
    }
