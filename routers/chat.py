import logging
import uuid

from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse

from models.chat import ChatRequest, ChatResponse, HistoryRequest, ToolChatRequest, ToolChatResponse
from services.llm import _client as _client, chat, chat_structured, chat_stream, chat_history
from services.compression import compress_chat_history, COMPRESS_THRESHOLD
from services.tools import get_tool_definitions
from services.tool_loop import run_tool_round
from services.tool_registry import SideEffectAmbiguousError, tool_registry
from services.injection import check_injection, check_output_leak
from services.idempotency import normalize_idempotency_key, request_idempotency

conversations: dict[str, list] = {}
router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat")
async def llm_service(request:ChatRequest):
    return await chat(request.message)

@router.post("/chat/structured")
async def llm_service_structured(request:ChatRequest):
    return await chat_structured(request.message)

@router.post("/chat/stream")
async def llm_service_stream(request:ChatRequest):
    stream = await chat_stream(request.message)
    return StreamingResponse(stream, media_type="text/event-stream")

@router.post("/chat/history")
async def llm_service_history(request:HistoryRequest):
    conversations.setdefault(request.conversation_id, [])
    conversations[request.conversation_id].append({"role":"user","content":request.message})
    response = await chat_history(conversations[request.conversation_id])
    conversations[request.conversation_id].append({"role":"assistant","content":response})

    # 超过阈值时自动压缩旧消息，保持 context 窗口可控
    if len(conversations[request.conversation_id]) > COMPRESS_THRESHOLD:
        conversations[request.conversation_id] = await compress_chat_history(
            conversations[request.conversation_id]
        )

    return response


# ═══════════════════════════════════════════════════════════════════════════
# Function Calling 端点
# ═══════════════════════════════════════════════════════════════════════════

_TOOL_SYSTEM = (
    "你是一个智能学习助手，可以调用工具帮助用户完成学习任务。\n"
    "你的能力：\n"
    "- search_document：搜索文档内容\n"
    "- generate_quiz：根据文档出题\n"
    "- grade_answer：批改单道复习题答案（轻量 deterministic 批改）\n"
    "- update_learning_profile：把批改结果写回学习画像\n"
    "- plan_next_step：根据画像和最近结果规划下一步\n"
    "- get_user_profile：查看用户学习画像\n"
    "- get_learning_path：生成学习路径\n\n"
    "根据用户的自然语言请求，自主判断需要调用哪些工具。"
    "调用工具获取结果后，用友好的中文回复用户。"
)

MAX_TOOL_ROUNDS = 3  # 防止无限 tool calling 循环

# ── 第 3 层：工具白名单（权限隔离）──
# 白名单单一数据源：从 ToolRegistry 派生，不在端点中硬编码。
# run_tool_round 内部也用 allowed_tool_names()，端点侧无需再各维护一份。


async def _execute_chat_with_tools(
    req: ToolChatRequest,
    *,
    run_id: str,
    idempotency_key: str | None,
) -> ToolChatResponse:
    """Function Calling 聊天端点。

    LLM 根据用户自然语言自主决定调用哪些工具，执行后生成最终回复。
    支持多轮 tool calling（最多 3 轮），覆盖需要多步推理的场景。

    示例：
      用户: "帮我出 5 道关于 Transformer 的选择题"
      LLM → tool_call: generate_quiz(document_id=..., topic="Transformer", count=5)
      → 执行工具 → LLM 生成包含题目的自然语言回复
    """
    # ── 第 1+2 层：Prompt Injection 检测 ──
    is_injection, reason = await check_injection(req.message)
    if is_injection:
        return ToolChatResponse(response=f"输入安全检查未通过：{reason}", tools_called=[])

    # 构建 context：注入 user_id 和 document_id 供 LLM 填充 tool 参数
    context_hint = f"\n当前用户 ID: {req.user_id}"
    if req.document_id:
        context_hint += f"\n当前文档 ID: {req.document_id}"

    messages = [
        {"role": "system", "content": _TOOL_SYSTEM + context_hint},
        {"role": "user", "content": req.message},
    ]

    tools_called: list[str] = []
    for _ in range(MAX_TOOL_ROUNDS):
        # run_tool_round 负责单轮 LLM→tool_calls→dispatch→回灌。
        # 传 client=_client 保留测试注入；白名单从 registry 派生。
        round_result = await run_tool_round(
            messages,
            tools=get_tool_definitions(),
            client=_client,
            run_id=run_id,
            user_id=req.user_id,
            idempotency_key=idempotency_key,
        )

        # 无 tool_calls → LLM 直接回复，执行第 4 层输出检查后返回
        if not round_result.has_tool_calls:
            content = round_result.content or ""
            is_leak, leak_reason = check_output_leak(content)
            if is_leak:
                logger.warning(f"[chat/tools] output leak blocked: {leak_reason}")
                content = "回复内容包含敏感信息，已拦截。请重新提问。"
            return ToolChatResponse(response=content, tools_called=tools_called)

        # 只记录实际派发的白名单工具
        for oc in round_result.outcomes:
            if oc.kind == "dispatched":
                tools_called.append(oc.name)

    # 超过最大轮次，取最后一条 assistant 内容
    last_content = next(
        (m["content"] for m in reversed(messages)
         if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content")),
        "处理轮次超限，请简化请求后重试。",
    )
    return ToolChatResponse(response=last_content, tools_called=tools_called)


@router.post("/chat/tools", response_model=ToolChatResponse)
async def chat_with_tools(
    req: ToolChatRequest,
    idempotency_key: str | None = Header(
        default=None, alias="Idempotency-Key"
    ),
) -> ToolChatResponse:
    """Run tool chat with an optional durable replay receipt."""
    key = normalize_idempotency_key(idempotency_key)
    if key:
        decision = await request_idempotency.begin(
            key, "chat.tools", req.model_dump(mode="json")
        )
        if decision.replayed:
            return ToolChatResponse.model_validate(decision.response)

    run_id = f"chat_tools_{uuid.uuid4().hex[:12]}"
    try:
        response = await _execute_chat_with_tools(
            req, run_id=run_id, idempotency_key=key
        )
        if key:
            await request_idempotency.complete(
                key, response.model_dump(mode="json")
            )
        return response
    except BaseException as exc:
        durable_effect = await request_idempotency.abort(key) if key else False
        effect_attempted = durable_effect or tool_registry.has_effect_attempt(run_id)
        if (
            isinstance(exc, Exception)
            and effect_attempted
            and not isinstance(exc, SideEffectAmbiguousError)
        ):
            raise SideEffectAmbiguousError("chat_tools_request") from exc
        raise
