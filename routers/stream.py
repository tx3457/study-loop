"""
Agent 流式端点

POST /agent/stream：SSE 流式响应，实时推送多 Agent 执行进度。

── 为什么需要流式响应 ──────────────────────────────────────────────────────────
Multi-Agent 流水线耗时较长（检索 + 多轮 LLM），同步等待用户体验差。
SSE 让前端在每个 Agent 节点完成后立即收到通知，实现"步骤可视化"效果。

── SSE vs WebSocket ─────────────────────────────────────────────────────────
SSE（Server-Sent Events）：单向推送，HTTP 长连接，适合服务端 → 客户端进度通知。
WebSocket：双向通信，适合聊天等需要双向交互的场景。
Agent 流水线是单向的，选 SSE 更简单，不需要额外依赖。

── SSE 消息格式（每条 `data: <json>\\n\\n`）──────────────────────────────────
  {"type": "node",  "node": "adapt_reader", "label": "读取用户画像"}  # 节点完成
  {"type": "done",  "result": {"quiz": {...}}}                        # 全部完成
  {"type": "error", "detail": "..."}                                  # 出错

── 实现原理 ─────────────────────────────────────────────────────────────────
使用 LangGraph `astream(stream_mode="updates")`：
  每个节点执行完成后 yield {node_name: {state_updates}}
  优于 `astream_events` 的原因：不依赖 LangChain LLM wrapper，更简单可靠。

注：Token 级别流（打字机效果）需要将 LLM 替换为 LangChain ChatOpenAI wrapper，
    这样 astream_events 能捕获 on_chat_model_stream 事件并 yield 每个 token。
    当前项目使用原生 AsyncOpenAI，只支持节点级别进度推送。
"""
import json
import logging
from fastapi import Depends, APIRouter
from fastapi.responses import StreamingResponse
from agents.orchestrator import orchestrator
from routers.orchestrator import RunRequest
from services.auth import require_user_id
from services.tracing import build_trace_config
from services.request_context import current_request_id, safe_sse_error

router = APIRouter(prefix="/agent", tags=["agent"])
logger = logging.getLogger(__name__)

# 节点名 → 用户可见标签
# 包含所有层级的节点名（Orchestrator 外层 + 各 subgraph 内层）
_NODE_LABELS: dict[str, str] = {
    # ── Orchestrator 层 ───────────────────────────────────────────────
    "input_guard":   "输入校验",
    "adapt_reader":  "读取用户画像",
    "quiz_agent":    "题目生成流水线",
    "grader_agent":  "AI 批改流水线",
    "adapt_writer":  "更新用户画像",
    "planner_agent": "学习路径流水线",
    "output_guard":  "输出格式校验",
    # ── AdaptAgent 内层 ───────────────────────────────────────────────
    "read_profile":  "读取用户画像",
    "write_profile": "写回用户画像",
    # ── QuizAgent 内层 ────────────────────────────────────────────────
    "retrieve":         "检索相关文档（Hybrid BM25+向量）",
    "sufficiency_check": "检索充分性判定（前置质量门）",
    "rewrite_query":    "改写检索 query 提升召回",
    "degrade":          "证据不足，降级出题参数",
    "generate":         "LLM 生成题目",
    "review":           "审核题目质量",
    # ── GraderAgent 内层 ──────────────────────────────────────────────
    "grade":         "执行 AI 批改",
    # ── PlannerAgent 内层 ─────────────────────────────────────────────
    "plan":          "LLM 规划学习路径",
}


def _sse(payload: dict) -> str:
    """将 dict 序列化为 SSE 消息行。格式规范：`data: <json>\\n\\n`"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/stream")
async def stream_agent(req: RunRequest, subject: str = Depends(require_user_id)):
    """Multi-Agent 流式执行入口。

    与 POST /agent/run 功能完全相同，但以 SSE 流式返回每个节点的执行进度。

    前端接入示例（JavaScript）：
        // EventSource 不支持 POST body，需用 fetch + ReadableStream
        const res = await fetch('/agent/stream', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({action: 'quiz', document_id: '...', count: 5}),
        });
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            const lines = decoder.decode(value).split('\\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const event = JSON.parse(line.slice(6));
                    console.log(event);  // {type, node, label} or {type, result}
                }
            }
        }
    """
    # 请求体里自报的 user_id 不作数：身份只来自 Authorization 头解析出的主体。
    req.user_id = subject
    async def _event_generator():
        final_state: dict = {}

        try:
            # astream(stream_mode="updates")：
            #   每个节点执行完成后 yield {node_name: {改变的 state 字段}}
            #   包含 Orchestrator 层和所有 subgraph 内层节点
            async for chunk in orchestrator.astream(
                req.model_dump(),
                stream_mode="updates",
                config=build_trace_config(
                    user_id=req.user_id,
                    session_id=req.session_id or req.document_id or None,
                    tags=[f"action:{req.action}", "endpoint:agent_stream"],
                ),
            ):
                # chunk 格式：{"node_name": {"field": value, ...}}
                node_name = next(iter(chunk))        # 当前完成的节点名
                node_updates = chunk[node_name]      # 该节点写入的 state 字段

                # 更新 final_state，收集所有节点写入的字段（用于最终结果）
                if isinstance(node_updates, dict):
                    final_state.update(node_updates)

                # 仅推送有语义标签的节点（过滤 LangGraph 内部匿名节点）
                if node_name in _NODE_LABELS:
                    yield _sse({
                        "type":  "node",
                        "node":  node_name,
                        "label": _NODE_LABELS[node_name],
                    })

            # ── 所有节点完成，推送最终结果 ─────────────────────────────────────
            if req.action == "quiz":
                result = {
                    "quiz":             final_state.get("quiz"),
                    "difficulty_score": final_state.get("difficulty_score"),
                    "weak_points":      final_state.get("weak_points"),
                }
            elif req.action == "grade":
                result = {"grading_report": final_state.get("grading_report")}
            else:  # plan
                result = {"learning_path": final_state.get("learning_path")}

            yield _sse({"type": "done", "result": result})

        except Exception as exc:
            logger.error(
                "[stream_agent] execution failed request_id=%s error_type=%s",
                current_request_id(),
                type(exc).__name__,
            )
            yield safe_sse_error(
                detail="Agent 执行失败，请稍后重试",
                code="agent_stream_failed",
            )

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "Connection":       "keep-alive",
            "X-Accel-Buffering": "no",   # 关闭 Nginx 缓冲，保证实时推送
        },
    )
