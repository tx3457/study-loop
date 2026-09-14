"""
GraderAgent：AI 批改 + 个性化讲解

直接复用 services/grader.py::grade_session，该函数从内存 sessions 读取答案并调用 LLM。
GraderAgent 的职责是将批改结果写入共享状态（grading_report），
供后续 AdaptAgent(adapt_writer) 更新用户画像使用。

失败处理：
  Transient error → with_retry 指数退避重试（最多 3 次）
  RetryExhausted  → 原样上抛，由 main.py 的 retry_exhausted_handler 归类为
                    provider 故障（429 / 504 / 503）。不要转成 ValueError：
                    那会把模型服务不可用伪装成调用方的错误。
"""
import logging
from langgraph.graph import StateGraph, START, END
from services.grader import grade_session
from services.retry import with_retry, RetryExhausted
from services.tracing import traceable
from agents.state import OrchestratorState

logger = logging.getLogger(__name__)


@traceable(name="grader_agent.grade", run_type="llm")
async def _grade(state: OrchestratorState) -> dict:
    """调用 AI 批改服务（带重试），将结果序列化为 dict 写入共享状态。"""
    try:
        report = await with_retry(lambda: grade_session(state["session_id"]))
    except RetryExhausted:
        # 不要把 session_id 写进日志：它由调用方提供，带换行就能伪造日志条目。
        logger.error("[grader_agent] grade retries exhausted")
        raise
    return {"grading_report": report.model_dump()}


_builder = StateGraph(OrchestratorState)
_builder.add_node("grade", _grade)
_builder.add_edge(START, "grade")
_builder.add_edge("grade", END)
grader_agent = _builder.compile()
