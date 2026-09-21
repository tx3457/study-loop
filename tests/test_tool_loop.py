import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from services.tool_loop import run_tool_round


def _plain_response(content="done"):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content=content, tool_calls=None),
    )])


def test_ephemeral_system_suffix_merges_into_provider_copy_only():
    messages = [
        {"role": "system", "content": "base policy"},
        {"role": "user", "content": "question"},
    ]
    original = deepcopy(messages)
    captured = {}

    async def provider(call_messages, **_kwargs):
        captured["messages"] = call_messages
        return _plain_response()

    with patch("services.tool_loop.llm_chat", AsyncMock(side_effect=provider)):
        result = asyncio.run(run_tool_round(
            messages,
            tools=[],
            ephemeral_system_suffix="trusted state",
        ))

    assert result.content == "done"
    assert captured["messages"] == [
        {"role": "system", "content": "base policy\n\ntrusted state"},
        {"role": "user", "content": "question"},
    ]
    assert captured["messages"] is not messages
    assert captured["messages"][0] is not messages[0]
    assert messages == original


def test_ephemeral_system_suffix_rejects_other_ephemeral_messages():
    with pytest.raises(ValueError, match="mutually exclusive"):
        asyncio.run(run_tool_round(
            [{"role": "system", "content": "base"}],
            tools=[],
            extra_call_messages=[{"role": "system", "content": "old"}],
            ephemeral_system_suffix="new",
        ))


@pytest.mark.parametrize(
    ("messages", "suffix"),
    [
        ([], "state"),
        ([{"role": "user", "content": "question"}], "state"),
        ([{"role": "system", "content": 123}], "state"),
        ([{"role": "system", "content": "base"}], 123),
    ],
)
def test_ephemeral_system_suffix_requires_leading_string_system(messages, suffix):
    with pytest.raises(ValueError, match="leading system"):
        asyncio.run(run_tool_round(
            messages,
            tools=[],
            ephemeral_system_suffix=suffix,
        ))
