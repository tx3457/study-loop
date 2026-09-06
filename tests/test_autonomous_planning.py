import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import routers.autonomous as autonomous


def _completion(content: str, *, finish_reason: str = "stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(content=content),
        )]
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1. 检索资料\n2. 总结证据", ["检索资料", "总结证据"]),
        ("1) 检索资料\n2) 总结证据", ["检索资料", "总结证据"]),
        ("1）检索资料\n2）总结证据", ["检索资料", "总结证据"]),
        ("步骤1：检索资料\n步骤2：总结证据", ["检索资料", "总结证据"]),
        ("- 检索资料\n- 总结证据", ["检索资料", "总结证据"]),
        ("* **检索资料**\n* **总结证据**", ["检索资料", "总结证据"]),
    ],
)
def test_parse_plan_accepts_supported_lists(text, expected):
    assert autonomous._parse_plan(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        '["检索资料", "总结证据"]',
        '{"steps": ["检索资料", "总结证据"]}',
        '1. {"tool": "search_document"}\n2. {"tool": "finalize"}',
        "- `search_document(document_id='d')`\n- `finalize()`",
        "先检索资料，然后总结证据。",
        "1. 只有一步",
        "1. 一\n2. 二\n3. 三\n4. 四\n5. 五\n6. 六",
        "1. 一\n这是一段额外解释\n2. 二",
        "2. 二\n1. 一",
    ],
)
def test_parse_plan_rejects_non_plan_or_out_of_bounds(text):
    assert autonomous._parse_plan(text) == []


def test_generate_plan_isolates_context_and_bounds_transport():
    mocked_chat = AsyncMock(return_value=_completion("1. 检索资料\n2. 总结证据"))

    with patch.object(autonomous, "llm_chat", mocked_chat):
        result = asyncio.run(autonomous._generate_plan("制定学习计划"))

    assert result == ["检索资料", "总结证据"]
    joined = "\n".join(
        message["content"] for message in mocked_chat.await_args.args[0]
    )
    assert "当前用户 ID" not in joined
    assert "当前文档 ID" not in joined
    assert mocked_chat.await_args.kwargs["max_retries"] == 1
    assert (
        mocked_chat.await_args.kwargs["total_timeout"]
        == autonomous.PLAN_TOTAL_TIMEOUT_SECONDS
    )


def test_generate_plan_retries_invalid_semantics_once_then_accepts():
    mocked_chat = AsyncMock(side_effect=[
        _completion("我建议先检索再总结。"),
        _completion("1. 检索资料\n2. 总结证据"),
    ])
    with patch.object(autonomous, "llm_chat", mocked_chat):
        result = asyncio.run(autonomous._generate_plan("制定学习计划"))
    assert result == ["检索资料", "总结证据"]
    assert mocked_chat.await_count == 2


def test_generate_plan_requires_stop_and_two_to_five_steps():
    mocked_chat = AsyncMock(side_effect=[
        _completion("1. 检索资料\n2. 总结证据", finish_reason="length"),
        _completion("1. 仍然只有一步"),
    ])
    with patch.object(autonomous, "llm_chat", mocked_chat):
        result = asyncio.run(autonomous._generate_plan("制定学习计划"))
    assert result == []
    assert mocked_chat.await_count == 2


def test_generate_plan_enforces_one_deadline_across_attempts():
    async def never_returns(*args, **kwargs):
        await asyncio.Event().wait()

    mocked_chat = AsyncMock(side_effect=never_returns)
    with patch.object(autonomous, "llm_chat", mocked_chat), patch.object(
        autonomous, "PLAN_TOTAL_TIMEOUT_SECONDS", 0.01
    ):
        result = asyncio.run(autonomous._generate_plan("制定学习计划"))
    assert result == []
    assert mocked_chat.await_count == 1


def test_generate_plan_propagates_cancellation():
    mocked_chat = AsyncMock(side_effect=asyncio.CancelledError())
    with patch.object(autonomous, "llm_chat", mocked_chat):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(autonomous._generate_plan("制定学习计划"))


def test_generate_plan_logs_only_structural_diagnostics(caplog):
    secret = "不要记录 SECRET-PLAN-TEXT"
    mocked_chat = AsyncMock(side_effect=[_completion(secret), _completion(secret)])
    with caplog.at_level(logging.WARNING), patch.object(
        autonomous, "llm_chat", mocked_chat
    ):
        result = asyncio.run(autonomous._generate_plan("制定学习计划"))
    assert result == []
    assert "SECRET-PLAN-TEXT" not in caplog.text
    assert "invalid_format" in caplog.text
