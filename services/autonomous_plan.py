"""Autonomous Agent 的规划阶段。

把用户目标拆成 2-5 步的显式计划，作为方向提示喂给 ReAct 循环。这一步是可选
的：调用方决定要不要生成（短 query 直接跳过），生成失败也只返回空列表，不中断
整轮执行。

解析刻意严格——只接受编号或项目符号的显式列表，散文、JSON 和裸工具调用一律判
为无效并重试一次。模型在这里输出的内容会进入下一轮上下文，宽松解析等于让它自
己往上下文里塞任意东西。
"""

from __future__ import annotations

import asyncio
import logging
import re

from services.injection import check_output_leak
from services.llm import _client as _client, llm_chat

logger = logging.getLogger(__name__)

PLAN_TOTAL_TIMEOUT_SECONDS = 20.0
PLAN_TRANSPORT_RETRIES = 1
PLAN_SEMANTIC_ATTEMPTS = 2

_PLAN_SYSTEM = (
    "你是 ReAct Agent 的规划助手。任务：把用户的学习目标拆解为 2-5 个可执行步骤。\n\n"
    "输出格式严格按以下编号列表：\n"
    "1. <第一步描述>\n2. <第二步描述>\n...\n"
    "不要输出任何解释，只输出编号列表。"
)


_NUMBERED_PLAN_LINE = re.compile(r"^(\d+)\s*[.)）、]\s*(.+)$")
_NAMED_PLAN_LINE = re.compile(r"^步骤\s*(\d+)\s*[:：]\s*(.+)$")
_BULLET_PLAN_LINE = re.compile(r"^[-*+•]\s+(.+)$")
_BARE_TOOL_CALL = re.compile(
    r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*\([^\n]*\)\s*;?$"
)

def _strip_inline_markdown(text: str) -> str:
    value = text.strip()
    for marker in ("**", "__", "`"):
        if (
            value.startswith(marker)
            and value.endswith(marker)
            and len(value) > 2 * len(marker)
        ):
            value = value[len(marker):-len(marker)].strip()
    return value


def _parse_plan(plan_text: str) -> list[str]:
    """Parse only an explicit 2-5 item plan, never prose or tool payloads."""
    raw = plan_text.strip()
    if not raw or raw[0] in "[{":
        return []

    steps: list[str] = []
    numbered_indexes: list[int] = []
    marker_kinds: set[str] = set()
    for raw_line in raw.splitlines():
        line = re.sub(r"^#{1,6}\s+", "", raw_line.strip())
        if not line:
            continue

        match = _NUMBERED_PLAN_LINE.fullmatch(line)
        if match:
            marker_kinds.add("numbered")
            numbered_indexes.append(int(match.group(1)))
            body = match.group(2)
        else:
            match = _NAMED_PLAN_LINE.fullmatch(line)
            if match:
                marker_kinds.add("numbered")
                numbered_indexes.append(int(match.group(1)))
                body = match.group(2)
            else:
                match = _BULLET_PLAN_LINE.fullmatch(line)
                if not match:
                    return []
                marker_kinds.add("bullet")
                body = match.group(1)

        step = _strip_inline_markdown(body)
        if not step or step[0] in "[{" or _BARE_TOOL_CALL.fullmatch(step):
            return []
        steps.append(step)

    if not 2 <= len(steps) <= 5 or len(marker_kinds) != 1:
        return []
    if numbered_indexes and numbered_indexes != list(range(1, len(steps) + 1)):
        return []
    return steps


def _plan_diagnostic(finish_reason: object, steps: list[str]) -> str:
    if finish_reason != "stop":
        return "invalid_finish_reason"
    return "valid" if steps else "invalid_format"


async def _generate_plan_within_budget(query: str) -> list[str]:
    for attempt in range(1, PLAN_SEMANTIC_ATTEMPTS + 1):
        system_prompt = _PLAN_SYSTEM
        if attempt > 1:
            system_prompt += (
                "\n上次响应结构无效。重新输出完整的 2-5 步编号列表，"
                "不要输出解释、JSON 或裸工具调用。"
            )
        response = await llm_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            client=_client,
            max_retries=PLAN_TRANSPORT_RETRIES,
            total_timeout=PLAN_TOTAL_TIMEOUT_SECONDS,
        )
        choice = response.choices[0]
        raw_plan = choice.message.content or ""
        if check_output_leak(raw_plan)[0]:
            logger.warning("[autonomous] plan output blocked by safety policy")
            return []
        steps = _parse_plan(raw_plan)
        diagnostic = _plan_diagnostic(
            getattr(choice, "finish_reason", None), steps
        )
        if diagnostic == "valid":
            logger.info(
                "[autonomous] plan accepted: attempt=%s step_count=%s",
                attempt,
                len(steps),
            )
            return steps
        logger.warning(
            "[autonomous] plan rejected: attempt=%s diagnostic=%s step_count=%s",
            attempt,
            diagnostic,
            len(steps),
        )
    return []


async def generate_plan(query: str) -> list[str]:
    """Generate an isolated plan under one total deadline."""
    try:
        return await asyncio.wait_for(
            _generate_plan_within_budget(query),
            timeout=PLAN_TOTAL_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.warning(
            "[autonomous] plan generation failed: error_type=%s",
            type(exc).__name__,
        )
    return []
