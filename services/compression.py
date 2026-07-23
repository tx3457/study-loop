"""
历史压缩服务

Context Engineering — Compress 策略：
  对话历史超过阈值时，将旧消息 LLM 摘要压缩为单条 system 消息，
  保留最近 N 条作为精确上下文，其余替换为摘要。
  压缩后减少历史消息占用的 token。

触发条件：消息数 > COMPRESS_THRESHOLD（默认 20 条）
保留精确：最近 KEEP_RECENT 条（默认 6 条）
"""
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from services.llm import (
    llm_parse,
    structured_client as _client,
    structured_model as _model,
)

load_dotenv(Path(__file__).parent.parent / ".env")

COMPRESS_THRESHOLD = 20   # 超过多少条消息触发压缩
KEEP_RECENT = 6           # 始终保留最近 N 条精确消息


class CompressionResult(BaseModel):
    """压缩摘要结构化输出（替代自由文本）。"""
    summary: str           # 对话核心内容概述（200 字以内）
    key_points: list[str]  # 关键结论 / 决策（3-5 条）


async def _summarize_messages(messages: list[dict]) -> str:
    """用 LLM 将旧消息列表压缩为结构化摘要。

    使用 structured output 替代自由文本，确保摘要格式一致，
    并提取 key_points 便于后续语义检索或 UI 展示。
    """
    formatted = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )
    response = await llm_parse(
        messages=[
            {
                "role": "system",
                "content": (
                    "你是对话历史摘要助手。将下方对话历史压缩为结构化摘要，"
                    "保留关键信息和结论，去除冗余内容。摘要用中文，200字以内。"
                ),
            },
            {"role": "user", "content": formatted},
        ],
        response_format=CompressionResult,
        client=_client,
        model=_model,
    )
    result = response.choices[0].message.parsed
    # 拼接：摘要 + 关键结论，保持与下游兼容
    points = " | ".join(result.key_points) if result.key_points else ""
    return f"{result.summary}\n关键结论：{points}" if points else result.summary


async def compress_chat_history(messages: list[dict]) -> list[dict]:
    """
    超过 COMPRESS_THRESHOLD 时触发压缩：
      旧消息（前 len-KEEP_RECENT 条）→ LLM 摘要 → 单条 system message
      最近 KEEP_RECENT 条保持原样不变

    未超过阈值时原样返回，不做任何处理。
    """
    if len(messages) <= COMPRESS_THRESHOLD:
        return messages

    old = messages[:-KEEP_RECENT]
    recent = messages[-KEEP_RECENT:]

    summary_text = await _summarize_messages(old)
    summary_msg = {
        "role": "system",
        "content": f"[对话历史摘要（已压缩 {len(old)} 条消息）]\n{summary_text}",
    }
    return [summary_msg] + recent
