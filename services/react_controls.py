"""
Shared ReAct control-tool schemas and prompt fragments.

The control tools are intentionally not registered in ToolRegistry because they
are loop controls, not business tools. Callers still handle their effects
locally: `finalize` ends the loop and `ask_user` pauses/resumes through the
caller-specific HITL mechanism.
"""
from copy import deepcopy


CONTROL_TOOL_NAMES = {"finalize", "ask_user"}


_FINALIZE_TOOL = {
    "type": "function",
    "function": {
        "name": "finalize",
        "description": (
            "当你认为已经收集到足够信息可以回答用户时，调用此工具结束循环。"
            "必须给出 final_answer（面向用户的最终回复）和 reason（结束理由）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "final_answer": {
                    "type": "string",
                    "description": "面向用户的最终自然语言回复，要简洁可读。",
                },
                "reason": {
                    "type": "string",
                    "description": "为什么这里可以结束？例如 '已完成 plan 所有步骤' / '已获取学习路径'",
                },
                "citation_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "回答所依据的 search_document observation 中真实 chunk_ids。",
                    "default": [],
                },
                "abstained": {
                    "type": "boolean",
                    "description": "检索证据不足、选择不作答时为 true。",
                    "default": False,
                },
            },
            "required": ["final_answer"],
        },
    },
}


def build_control_tools(*, ask_user_resume_hint: str) -> list[dict]:
    """Return fresh control-tool schemas so callers cannot mutate globals."""
    ask_user_tool = {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "当你缺少关键信息（如未指定文档、目标模糊）无法继续时，调用此工具向用户提问。"
                f"{ask_user_resume_hint}"
                "只问真正必要的问题，避免每步都打扰用户。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "你想问用户的具体问题。例如 '请告诉我你想学习的文档 ID'",
                    },
                },
                "required": ["question"],
            },
        },
    }
    return [deepcopy(_FINALIZE_TOOL), ask_user_tool]


def build_react_system_prompt(
    *,
    include_live_mcp: bool = False,
    include_review_loop: bool = False,
) -> str:
    """Build the shared ReAct tutor prompt with endpoint-specific hints."""
    tool_lines = [
        "业务工具：search_document / generate_quiz / grade_answer / update_learning_profile / "
        "plan_next_step / get_user_profile / get_learning_path",
    ]
    if include_live_mcp:
        tool_lines.append(
            "联网工具（若已接入）：mcp_ddg_search（联网搜索）/ "
            "mcp_ddg_fetch_content（抓取网页正文）——"
            "仅当本地文档库无法回答、需要文档外的最新/外部信息时才用；"
            "学习与出题材料优先用已上传文档。"
        )
    tool_lines.append("控制工具：finalize（结束并给最终答案）/ ask_user（向用户提问）")

    principles = [
        "你是唯一的决策者：根据当前已有信息，自主决定下一步",
        "若已能回答用户，立即调用 finalize 给出 final_answer + reason，不要继续调工具",
        "若缺关键信息（如未指定文档 ID），调用 ask_user 求助而不是瞎猜",
    ]
    if include_review_loop:
        principles.append(
            "复习闭环优先顺序：search_document → generate_quiz → grade_answer → "
            "update_learning_profile → plan_next_step → finalize"
        )
    principles.extend([
        "如果使用 search_document 的内容回答，finalize 时必须把实际 observation 中的 chunk_ids 放入 citation_ids；不得编造 ID",
        "简单概念问题（如『什么是 RAG』）如果你知道答案，直接 finalize 给答案，不需要调工具",
        "不要输出空 message 或 '执行完毕' 这种废话——要么调工具要么调 finalize",
        "同一个工具不要短时间重复调用（除非参数明显不同）",
    ])

    numbered_principles = "\n".join(f"{i}. {text}" for i, text in enumerate(principles, 1))
    return (
        "你是 ReAct Agent，可以调用工具完成用户的学习任务。\n\n"
        "【可用工具】\n"
        + "\n".join(tool_lines)
        + "\n\n【工作原则】\n"
        + numbered_principles
        + "\n"
    )


def build_react_decision_prompt(state_summary: str) -> str:
    """Per-round state reminder used only for the next LLM call."""
    return (
        f"[Current state]\n{state_summary}\n\n"
        "请决定下一步：调用业务工具 / finalize / ask_user。"
    )
