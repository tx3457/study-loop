"""
Function Calling 工具定义与分发

每个 tool 注册到 ToolRegistry，invoke 时自动带 per-tool timeout + retry + audit。

向后兼容：
  - TOOL_DEFINITIONS 仍然导出（legacy import-time snapshot）
  - get_tool_definitions() 从 registry 动态生成；interrupt 节点另用 replay-safe 子集
  - dispatch_tool 的 run_id/user_id 为可选参数

"""
import json
import logging
from typing import Any, Awaitable, Callable, Optional

from services.learning_path import generate_learning_path
from services.injection import scan_untrusted_content
from services.memory import (
    append_weak_points,
    get_user_profile,
    persist_memory_snapshot,
    update_mastery,
)
from services.rag import generate_question
from services.tool_registry import (
    EffectMode,
    Tool,
    ToolArgumentBinding,
    ToolMetadata,
    tool_registry,
)
from services.vectorstore import retrieve_with_rewrite

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Tool Handler（业务实现，保持 async + 返回 str 的契约）
# ═══════════════════════════════════════════════════════════════════════════

async def _search_document(document_id: str, query: str, user_id: str) -> str:
    # 走统一生产检索入口（含 HyDE/Multi-query 改写），与出题流保持一致。
    # user_id 由注册表按 owner_argument 对照可信上下文校验过，检索因此
    # 只可能命中该属主自己的文档。
    result = await retrieve_with_rewrite(document_id, query, owner_id=user_id)
    chunks = result["documents"][0][:3]    # 避免 token 爆炸
    chunk_ids = result["ids"][0][:3]
    if len(chunks) != len(chunk_ids):
        raise ValueError("retrieval returned misaligned documents and ids")

    # chunks 是用户上传文件的原文，会被 tool_loop 原样回灌给模型。它是数据，
    # 不是指令：显式标注来源可信度，并在命中注入模式时打标，让 tool_loop 给
    # 本次 run 上 taint（随后禁止非幂等写入）。这里不拦截，理由见
    # services/injection.py::scan_untrusted_content。
    suspicious, reason = scan_untrusted_content("\n".join(chunks))
    payload = {
        "document_id": document_id,
        "chunks": chunks,
        "chunk_ids": chunk_ids,
        "content_trust": "untrusted_document_text",
    }
    if suspicious:
        payload["injection_flagged"] = True
        payload["injection_reason"] = reason
    return json.dumps(payload, ensure_ascii=False)


async def _generate_quiz(
    document_id: str,
    topic: str = "",
    count: int = 3,
    difficulty: str = "medium",
    type: str = "choice",
) -> str:
    effective_topic = str(topic or "").strip() or "文档综合内容"
    quiz = await generate_question(
        document_id=document_id,
        description=effective_topic,
        count=count,
        difficulty=difficulty,
        type=type,
    )
    return quiz.model_dump_json(ensure_ascii=False)


async def _get_user_profile(
    user_id: str,
    document_id: str | None = None,
) -> str:
    profile = await get_user_profile(user_id)
    if not isinstance(profile, dict):
        return json.dumps(profile, ensure_ascii=False, default=str)

    topic_mastery = profile.get("topic_mastery")
    mastery_values = {
        str(topic): float(value)
        for topic, value in (
            topic_mastery.items() if isinstance(topic_mastery, dict) else []
        )
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if document_id and document_id in mastery_values:
        mastery = mastery_values[document_id]
        mastery_scope = document_id
    elif mastery_values:
        mastery = sum(mastery_values.values()) / len(mastery_values)
        mastery_scope = "all_topics"
    else:
        mastery = None
        mastery_scope = document_id or "all_topics"

    return json.dumps(
        {
            **profile,
            "mastery": round(mastery, 3) if mastery is not None else None,
            "mastery_scope": mastery_scope,
        },
        ensure_ascii=False,
        default=str,
    )


async def _get_learning_path(document_id: str, user_id: str) -> str:
    path = await generate_learning_path(document_id, owner_id=user_id)
    return path.model_dump_json(ensure_ascii=False)


def _coerce_dict(value: Any) -> dict:
    """Tool args may arrive as JSON strings or objects from function calling."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _norm_answer(value: str) -> str:
    return "".join(str(value or "").strip().lower().split())


def _profile_update_values(grade_result: dict | str) -> tuple[dict, float, list[str]]:
    """Return the exact score/gap semantics applied by the profile handler."""
    result = _coerce_dict(grade_result)
    raw_score = result.get("score")
    if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
        score = float(raw_score)
    else:
        score = 1.0 if result.get("is_correct") else 0.0

    if score == 0:
        score = 0.0

    gaps: list[str] = []
    gap = result.get("knowledge_gap")
    if gap:
        gaps.append(str(gap))
    for item in result.get("knowledge_gaps", []) or []:
        if item:
            gaps.append(str(item))
    return result, score, gaps


def _normalize_profile_update_event(arguments: dict) -> dict:
    """Canonical identity for one learner-profile mutation event."""
    result, score, gaps = _profile_update_values(arguments.get("grade_result", {}))
    return {
        "user_id": arguments.get("user_id"),
        "document_id": arguments.get("document_id"),
        "grade_result": {
            "question": result.get("question"),
            "user_answer": result.get("user_answer"),
            "correct_answer": result.get("correct_answer"),
            "score": score,
            "knowledge_gaps": gaps,
        },
    }


async def _grade_answer(
    question: str,
    answer: str,
    correct_answer: str,
    evidence: str = "",
    explanation: str = "",
    question_type: str = "short_answer",
) -> str:
    """Deterministic single-answer grader for the ReAct tutor.

    The full production session grader remains services.grader.grade_session.
    This tool does not require a model call.
    """
    ans_norm = _norm_answer(answer)
    correct_norm = _norm_answer(correct_answer)
    if question_type in {"choice", "true_false"}:
        is_correct = ans_norm == correct_norm
    else:
        is_correct = bool(
            ans_norm
            and correct_norm
            and (
                ans_norm == correct_norm
                or (len(correct_norm) >= 4 and correct_norm in ans_norm)
                or (len(ans_norm) >= 4 and ans_norm in correct_norm)
            )
        )

    gap = None if is_correct else f"需要复习：{(evidence or explanation or question)[:80]}"
    payload = {
        "question": question,
        "user_answer": answer,
        "correct_answer": correct_answer,
        "is_correct": is_correct,
        "score": 1.0 if is_correct else 0.0,
        "feedback": "回答正确，可以进入下一步。" if is_correct else (
            f"答案还不准确。参考答案：{correct_answer}。"
            + (f" 依据：{explanation}" if explanation else "")
        ),
        "knowledge_gap": gap,
        "graded_by": "deterministic_demo_grader",
    }
    return json.dumps(payload, ensure_ascii=False)


async def _update_learning_profile(user_id: str, document_id: str, grade_result: dict | str) -> str:
    """Write a single-question grade result into the learner memory banks."""
    _, score, gaps = _profile_update_values(grade_result)
    mastery = await update_mastery(user_id, document_id, score)
    if gaps:
        await append_weak_points(user_id, gaps, document_id)
    await persist_memory_snapshot()

    return json.dumps({
        "user_id": user_id,
        "document_id": document_id,
        "score": round(float(score), 3),
        "mastery": mastery,
        "weak_points_added": gaps,
        "mode": "memory_bank_update",
    }, ensure_ascii=False)


async def _plan_next_step(
    profile: dict | str | None = None,
    last_result: dict | str | None = None,
) -> str:
    """Deterministic next-step planner for trace display."""
    profile_data = _coerce_dict(profile)
    result = _coerce_dict(last_result)
    score = result.get("score")
    if not isinstance(score, (int, float)):
        score = 1.0 if result.get("is_correct") else 0.0
    weak_points = profile_data.get("weak_points") or []
    if result.get("knowledge_gap"):
        weak_points = [result["knowledge_gap"], *weak_points]

    if score >= 0.8:
        action = "advance"
        recommendation = "提高一个难度，做 2-3 道迁移应用题。"
    elif score >= 0.5:
        action = "practice"
        focus = "、".join(str(p) for p in weak_points[:2]) or "刚才题目的关键概念"
        recommendation = f"围绕 {focus} 再练一组同难度题。"
    else:
        action = "remediate"
        focus = "、".join(str(p) for p in weak_points[:2]) or "材料中的基础定义"
        recommendation = f"先回看 {focus}，用一道基础题复测后再进阶。"

    return json.dumps({
        "action": action,
        "recommendation": recommendation,
        "focus": weak_points[:3],
        "based_on_score": round(float(score), 3),
    }, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# 2. 注册到 Registry（每个 tool 单独声明 SLO）
# ═══════════════════════════════════════════════════════════════════════════

def _register_all() -> None:
    """模块加载时注册所有工具。幂等（registry 重复注册会 warn 但不出错）。"""
    tool_registry.register(Tool(
        name="search_document",
        description="搜索已上传文档的相关段落（Hybrid BM25 + 向量 + RRF + 改写）",
        parameters_schema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "用户 ID"},
                "document_id": {"type": "string", "description": "文档 ID"},
                "query": {"type": "string", "description": "搜索关键词或问题"},
            },
            "required": ["user_id", "document_id", "query"],
        },
        handler=_search_document,
        # 检索：embed + ChromaDB + BM25，应该快。给 20s 应付偶发慢
        metadata=ToolMetadata(
            timeout_sec=20.0,
            max_retries=2,
            effect_mode=EffectMode.READ_ONLY,
        ),
    ))

    tool_registry.register(Tool(
        name="generate_quiz",
        description=(
            "根据文档内容生成测验题目。topic 可选；用户未指定主题时直接覆盖"
            "文档综合内容，不要先调用 search_document 来补造 topic。"
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "文档 ID"},
                "topic": {
                    "type": "string",
                    "description": "可选的出题主题；省略时覆盖文档综合内容",
                    "default": "",
                },
                "count": {"type": "integer", "description": "题目数量", "default": 3},
                "difficulty": {
                    "type": "string",
                    "enum": ["easy", "medium", "hard"],
                    "description": "难度等级",
                    "default": "medium",
                },
                "type": {
                    "type": "string",
                    "enum": ["choice", "true_false", "short_answer"],
                    "description": "题型",
                    "default": "choice",
                },
            },
            "required": ["document_id"],
        },
        handler=_generate_quiz,
        # 生成涉及 LLM thinking + 结构化输出，慢；重试一次防止累积成本
        metadata=ToolMetadata(
            timeout_sec=60.0,
            max_retries=1,
            effect_mode=EffectMode.READ_ONLY,
        ),
    ))

    tool_registry.register(Tool(
        name="get_user_profile",
        description=(
            "获取当前会话用户的学习画像：总体/指定文档掌握度、薄弱知识点、历史会话数。"
            "只能读取当前用户，禁止指定或访问其他用户。"
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "用户 ID"},
                "document_id": {
                    "type": "string",
                    "description": "可选文档 ID；提供时优先返回该文档的 mastery",
                },
            },
            "required": ["user_id"],
        },
        handler=_get_user_profile,
        # 首次读取可能迁移多个 legacy bank；部分迁移后的失败不满足安全重试条件。
        metadata=ToolMetadata(
            timeout_sec=5.0,
            max_retries=0,
            effect_mode=EffectMode.UNKNOWN,
            owner_argument="user_id",
        ),
    ))

    tool_registry.register(Tool(
        name="get_learning_path",
        description="为用户生成基于文档内容的多阶段学习路径规划",
        parameters_schema={
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "文档 ID"},
            },
            "required": ["document_id"],
        },
        handler=_get_learning_path,
        # 全文 LLM 规划，最慢
        metadata=ToolMetadata(
            timeout_sec=90.0,
            max_retries=1,
            effect_mode=EffectMode.READ_ONLY,
        ),
    ))

    tool_registry.register(Tool(
        name="grade_answer",
        description=(
            "批改单道复习题的用户答案，返回 score/feedback/knowledge_gap。"
            "这是无模型 key 可运行的轻量 deterministic 批改工具；完整会话批改走 grader_agent。"
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "题目文本；数学表达式和字段值必须保持为一个原子参数",
                },
                "answer": {"type": "string", "description": "用户答案"},
                "correct_answer": {"type": "string", "description": "参考答案"},
                "evidence": {"type": "string", "description": "检索证据或课程材料片段", "default": ""},
                "explanation": {"type": "string", "description": "参考解析", "default": ""},
                "question_type": {
                    "type": "string",
                    "enum": ["choice", "true_false", "short_answer"],
                    "description": "题型",
                    "default": "short_answer",
                },
            },
            "required": ["question", "answer", "correct_answer"],
        },
        handler=_grade_answer,
        metadata=ToolMetadata(
            timeout_sec=5.0,
            max_retries=0,
            effect_mode=EffectMode.READ_ONLY,
        ),
    ))

    tool_registry.register(Tool(
        name="update_learning_profile",
        description=(
            "把单题批改结果写回当前会话用户的学习画像 memory bank，更新 mastery "
            "和 weak_points。只能写当前用户；同一批改结果在一次运行中最多写入一次。"
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "用户 ID"},
                "document_id": {"type": "string", "description": "文档 ID 或课程材料 ID"},
                "grade_result": {
                    "type": "object",
                    "description": "grade_answer 返回的结果对象",
                    "additionalProperties": True,
                },
            },
            "required": ["user_id", "document_id", "grade_result"],
        },
        handler=_update_learning_profile,
        # EMA 写入不是幂等操作；提交后的超时无法判断是否已生效，禁止自动重放。
        metadata=ToolMetadata(
            timeout_sec=5.0,
            max_retries=0,
            effect_mode=EffectMode.NON_IDEMPOTENT,
            owner_argument="user_id",
            dedupe_within_run=True,
            dedupe_normalizer=_normalize_profile_update_event,
            dedupe_normalizer_id="update_learning_profile_event_v1",
            argument_bindings=(
                ToolArgumentBinding(
                    source_tool="grade_answer",
                    source_path="$",
                    target_argument="grade_result",
                ),
            ),
        ),
    ))

    tool_registry.register(Tool(
        name="plan_next_step",
        description=(
            "根据最近结果给出下一步复习建议；已有画像时可以结合，但不要为了填写"
            "可选 profile 单独调用 get_user_profile。服务器会补齐已产生的可信工具结果。"
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "profile": {
                    "type": "object",
                    "description": "可选的用户画像对象，可来自 get_user_profile 或 update_learning_profile",
                    "additionalProperties": True,
                    "default": {},
                },
                "last_result": {
                    "type": "object",
                    "description": "最近一次 grade_answer 或 update_learning_profile 的结果",
                    "additionalProperties": True,
                },
            },
            "required": ["last_result"],
        },
        handler=_plan_next_step,
        metadata=ToolMetadata(
            timeout_sec=5.0,
            max_retries=0,
            effect_mode=EffectMode.READ_ONLY,
            argument_bindings=(
                ToolArgumentBinding(
                    source_tool="update_learning_profile",
                    source_path="$",
                    target_argument="profile",
                ),
                ToolArgumentBinding(
                    source_tool="get_user_profile",
                    source_path="$",
                    target_argument="profile",
                ),
                ToolArgumentBinding(
                    source_tool="grade_answer",
                    source_path="$",
                    target_argument="last_result",
                ),
                ToolArgumentBinding(
                    source_tool="get_user_profile",
                    source_path="mastery",
                    target_argument="last_result.score",
                ),
            ),
        ),
    ))


_register_all()


# ═══════════════════════════════════════════════════════════════════════════
# 3. 兼容接口
# ═══════════════════════════════════════════════════════════════════════════

# 保留 TOOL_DEFINITIONS import 兼容快照。
# standalone agent 与带收据的 Chat 可用 get_tool_definitions() 动态发现全量工具；
# 无收据 Chat 仅使用严格只读快照，interrupt-capable 节点则使用下面的
# replay-safe schema + allowlist 双重约束。
TOOL_DEFINITIONS = tool_registry.get_openai_schemas()


def get_tool_definitions() -> list[dict]:
    """Return current OpenAI tool schemas from the live registry."""
    return tool_registry.get_openai_schemas()


def allowed_tool_names() -> set[str]:
    """业务工具白名单的单一数据源，从 registry 派生，不在各端点硬编码。

    增删业务工具只改注册块，chat 与 autonomous 的白名单自动同步。
    控制工具（finalize / ask_user）不在 registry，由 autonomous 端点自行处理。
    """
    return set(tool_registry.list_tools())


def get_read_only_tool_capabilities() -> tuple[list[dict], set[str]]:
    """Return one consistent schema/name snapshot of strictly read-only tools.

    This is narrower than ``replay_safe_tool_names``: an idempotent tool is
    safe only when the same arguments are replayed, while a receipt-less HTTP
    retry may ask the model to generate different arguments.
    """
    tools = []
    for name in tool_registry.list_tools():
        tool = tool_registry.get(name)
        if tool is not None and tool.metadata.effect_mode is EffectMode.READ_ONLY:
            tools.append(tool)
    return [tool.to_openai_schema() for tool in tools], {tool.name for tool in tools}


def replay_safe_tool_names() -> set[str]:
    """Return tools safe to re-run when an interrupt-capable node restarts."""
    safe_modes = {EffectMode.READ_ONLY, EffectMode.IDEMPOTENT}
    return {
        name
        for name in tool_registry.list_tools()
        if (tool := tool_registry.get(name))
        and tool.metadata.effect_mode in safe_modes
    }


def get_replay_safe_tool_definitions() -> list[dict]:
    """Return schemas for tools whose declared effects are replay-safe."""
    safe_names = replay_safe_tool_names()
    return [
        tool.to_openai_schema()
        for name in tool_registry.list_tools()
        if name in safe_names
        and (tool := tool_registry.get(name)) is not None
    ]


async def dispatch_tool(
    name: str,
    arguments: dict,
    *,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    idempotency_lease=None,
    on_before_handler: Callable[[], Awaitable[None]] | None = None,
) -> str:
    """根据 tool_call 名字调用工具，返回 JSON 字符串。

    可选 run_id / user_id 用于 audit 关联，省略时按无关联标识执行。
    """
    logger.info("[tools] dispatch tool=%s arg_count=%d", name, len(arguments))
    return await tool_registry.invoke(
        name,
        arguments,
        run_id=run_id,
        user_id=user_id,
        idempotency_key=idempotency_key,
        idempotency_lease=idempotency_lease,
        on_before_handler=on_before_handler,
    )
