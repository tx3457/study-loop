"""
统一 LLM 入口测试（D1）

验证 services/llm.py 的 llm_chat / llm_parse：
  1. client 注入：传入的 mock client 被使用（不碰模块级 _client）
  2. kwargs 透传：tools / response_format / model override 正确传递
  3. 重试横切：transient 错误（RateLimitError）会触发 with_retry 重试后成功
  4. max_retries=0 退回「只调一次」旧语义
  5. 4xx 客户端错误不重试，直接抛出

全程 mock，无网络。跑：
  python -m pytest test/test_llm_entrypoint.py -q
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
from openai import APIStatusError, RateLimitError

sys.path.insert(0, str(Path(__file__).parent.parent))

import services.llm as llm


def _resp(content="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, parsed=content))]
    )


def _rate_limit_error():
    req = httpx.Request("POST", "https://x/v1/chat")
    resp = httpx.Response(429, request=req)
    return RateLimitError("rate limited", response=resp, body=None)


def _bad_request_error():
    req = httpx.Request("POST", "https://x/v1/chat")
    resp = httpx.Response(400, request=req)
    return APIStatusError("bad request", response=resp, body=None)


class TestLlmChat(unittest.IsolatedAsyncioTestCase):

    async def test_uses_injected_client_and_passes_kwargs(self):
        mock_client = MagicMock()
        create = AsyncMock(return_value=_resp("hello"))
        mock_client.chat.completions.create = create

        tools = [{"type": "function", "function": {"name": "x"}}]
        out = await llm.llm_chat(
            [{"role": "user", "content": "hi"}],
            client=mock_client,
            tools=tools,
            tool_choice="auto",
        )
        self.assertEqual(out.choices[0].message.content, "hello")
        create.assert_awaited_once()
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(kwargs["tools"], tools)
        self.assertEqual(kwargs["tool_choice"], "auto")
        self.assertEqual(kwargs["model"], llm.model)   # 默认填充 model

    async def test_model_override(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_resp())
        await llm.llm_chat([{"role": "user", "content": "hi"}],
                           client=mock_client, model="custom-model")
        self.assertEqual(mock_client.chat.completions.create.call_args.kwargs["model"],
                         "custom-model")

    async def test_retry_on_transient_then_success(self):
        mock_client = MagicMock()
        # 第一次 429（transient），第二次成功
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[_rate_limit_error(), _resp("recovered")]
        )
        out = await llm.llm_chat([{"role": "user", "content": "hi"}],
                                 client=mock_client, base_delay=0.0)
        self.assertEqual(out.choices[0].message.content, "recovered")
        self.assertEqual(mock_client.chat.completions.create.await_count, 2)

    async def test_max_retries_zero_calls_once(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=_rate_limit_error())
        from services.retry import RetryExhausted
        with self.assertRaises(RetryExhausted):
            await llm.llm_chat([{"role": "user", "content": "hi"}],
                               client=mock_client, max_retries=0, base_delay=0.0)
        self.assertEqual(mock_client.chat.completions.create.await_count, 1)

    async def test_client_error_4xx_not_retried(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=_bad_request_error())
        with self.assertRaises(APIStatusError):
            await llm.llm_chat([{"role": "user", "content": "hi"}],
                               client=mock_client, base_delay=0.0)
        self.assertEqual(mock_client.chat.completions.create.await_count, 1)


class TestLlmParse(unittest.IsolatedAsyncioTestCase):

    async def test_uses_injected_client_and_response_format(self):
        mock_client = MagicMock()
        parse = AsyncMock(return_value=_resp("parsed"))
        mock_client.beta.chat.completions.parse = parse

        out = await llm.llm_parse(
            [{"role": "user", "content": "hi"}],
            response_format=dict,
            client=mock_client,
            max_tokens=128,
        )
        self.assertEqual(out.choices[0].message.parsed, "parsed")
        kwargs = parse.call_args.kwargs
        self.assertEqual(kwargs["response_format"], dict)
        self.assertEqual(kwargs["max_tokens"], 128)
        self.assertEqual(kwargs["model"], llm.structured_model)

    async def test_retry_on_transient(self):
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse = AsyncMock(
            side_effect=[_rate_limit_error(), _resp("ok")]
        )
        out = await llm.llm_parse([{"role": "user", "content": "hi"}],
                                  response_format=dict, client=mock_client, base_delay=0.0)
        self.assertEqual(out.choices[0].message.parsed, "ok")
        self.assertEqual(mock_client.beta.chat.completions.parse.await_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
