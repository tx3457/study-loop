"""
Agent Guardrails：Prompt Injection 防御

Resilience 三件套（容错维度的 defense-in-depth）：
  input_guard（输入校验 + 注入检测）→ Agent 执行 → output_guard（输出校验 + 泄露检测）
  这里只覆盖输入与输出校验，不构成完整的 Agent 运行时。

设计原则：
  Fail-fast：input_guard 在任何 LLM 调用之前运行，校验失败直接抛错 → 节省 token
  防御深度：output_guard 兜底，防止 LLM 返回格式异常数据或泄露敏感信息
  纵深防御：4 层防御叠加（正则 → LLM 检测 → 权限隔离 → 输出检查）

GuardrailError 继承 ValueError，由 main.py 的异常处理器统一转为 HTTP 400 响应。
"""
import logging

from agents.state import OrchestratorState
from services.injection import check_injection, check_output_leak

logger = logging.getLogger(__name__)


class GuardrailError(ValueError):
    """Guardrail 校验失败。main.py 的 ValueError handler 会转为 400。"""
    pass


# ── 输入校验规则（纯函数，便于单独测试）──────────────────────────────────────────

def _check_action(state: OrchestratorState) -> None:
    valid = {"quiz", "grade", "plan"}
    action = state.get("action", "quiz")
    if action not in valid:
        raise GuardrailError(f"action 必须是 {valid}，当前值：{action!r}")


def _check_count(state: OrchestratorState) -> None:
    """count 只在 quiz 时校验，范围 1-20。"""
    if state.get("action", "quiz") != "quiz":
        return
    count = state.get("count", 5)
    if not isinstance(count, int) or not (1 <= count <= 20):
        raise GuardrailError(f"count 必须是 1-20 的整数，当前值：{count!r}")


def _check_grade_needs_session(state: OrchestratorState) -> None:
    if state.get("action") == "grade" and not state.get("session_id"):
        raise GuardrailError("grade action 必须提供 session_id")


def _check_document_id(state: OrchestratorState) -> None:
    """quiz / plan 必须提供非空 document_id。"""
    if state.get("action") in ("quiz", "plan") and not (state.get("document_id") or "").strip():
        raise GuardrailError("quiz / plan action 必须提供 document_id")


# ── 输出校验规则 ───────────────────────────────────────────────────────────────

def _check_quiz_output(state: OrchestratorState) -> None:
    """检查 quiz 输出：题数匹配，每题必须有 question 和 answer 字段。"""
    if state.get("action") != "quiz":
        return
    quiz = state.get("quiz")
    if not quiz:
        raise GuardrailError("quiz action 结束后 quiz 输出为空")
    questions = quiz.get("questions", [])
    expected = state.get("count", 5)
    # 题数校验放宽:空→失败;超出请求数→失败;少于请求数是降级/重试耗尽的合理结果,
    # 放行可用题而非让整个请求 400(降级路径下 generate 可能用 min(count,3) 重出)
    if not questions:
        raise GuardrailError("quiz action 结束后题目列表为空")
    if len(questions) > expected:
        raise GuardrailError(f"生成题数 {len(questions)} 超过请求的 {expected} 题")
    if len(questions) < expected:
        logger.warning(f"[guardrail] 题数少于预期(期望 {expected} 实际 {len(questions)})，降级路径放行")
    for i, q in enumerate(questions):
        if not q.get("question") or not q.get("answer"):
            raise GuardrailError(f"第 {i + 1} 题缺少 question 或 answer 字段")


def _check_plan_output(state: OrchestratorState) -> None:
    """检查 plan 输出：必须有 stages 且不为空列表。"""
    if state.get("action") != "plan":
        return
    path = state.get("learning_path")
    if not path or not path.get("stages"):
        raise GuardrailError("plan action 输出为空或缺少 stages 字段")


# ── LangGraph 节点 ─────────────────────────────────────────────────────────────

async def _check_injection(state: OrchestratorState) -> None:
    """Prompt Injection 检测（第 1 层正则 + 第 2 层 LLM）。

    扫描所有用户可控的文本字段：description、document_id。
    检测到注入时抛 GuardrailError，阻止进入 Agent 流水线。
    """
    fields_to_check = [
        ("description", state.get("description", "")),
        ("document_id", state.get("document_id", "")),
    ]
    for field_name, value in fields_to_check:
        if not value:
            continue
        is_injection, reason = await check_injection(value)
        if is_injection:
            logger.warning(
                "[guardrail] injection detected: field=%s",
                field_name,
            )
            raise GuardrailError(f"输入安全检查未通过（{field_name}）：{reason}")


def _check_output_leak(state: OrchestratorState) -> None:
    """输出泄露检测（第 4 层）。检查 LLM 输出是否包含敏感信息。"""
    # 检查 quiz 输出
    quiz = state.get("quiz")
    if quiz:
        for i, q in enumerate(quiz.get("questions", [])):
            for field in ("question", "answer", "explanation"):
                text = q.get(field, "")
                is_leak, reason = check_output_leak(text)
                if is_leak:
                    logger.warning(
                        "[guardrail] output leak in question: index=%d field=%s",
                        i + 1,
                        field,
                    )
                    raise GuardrailError(f"输出安全检查未通过：第 {i+1} 题 {field} {reason}")

    # 检查 learning_path 输出
    path = state.get("learning_path")
    if path:
        text = str(path)
        is_leak, reason = check_output_leak(text)
        if is_leak:
            logger.warning("[guardrail] output leak in learning_path")
            raise GuardrailError(f"输出安全检查未通过：学习路径 {reason}")


async def input_guard(state: OrchestratorState) -> dict:
    """输入 Guardrail 节点。校验失败时抛 GuardrailError，不进入任何 Agent。

    检查顺序：参数校验（快）→ 注入检测（慢）。快速失败优先。
    """
    _check_action(state)
    _check_count(state)
    _check_grade_needs_session(state)
    _check_document_id(state)
    await _check_injection(state)
    return {}  # 只做校验，不修改状态


async def output_guard(state: OrchestratorState) -> dict:
    """输出 Guardrail 节点。校验数据格式 + 敏感信息泄露检测。"""
    _check_quiz_output(state)
    _check_plan_output(state)
    _check_output_leak(state)
    return {}
