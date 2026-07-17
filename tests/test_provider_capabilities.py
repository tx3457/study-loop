"""Explicit live-capability probe tests (all provider calls are mocked)."""

from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace

from services.provider_capabilities import ProviderCapabilityChecker
from services.provider_config import load_provider_configs


def _provider_env() -> dict[str, str]:
    return {
        "LLM_API_KEY": "secret-chat-key",
        "LLM_BASE_URL": "https://provider.example/v1",
        "LLM_MODEL": "chat-model",
        "STRUCTURED_API_KEY": "secret-structured-key",
        "STRUCTURED_BASE_URL": "https://structured.example/v1",
        "STRUCTURED_MODEL": "structured-model",
        "EMBEDDING_API_KEY": "secret-embedding-key",
        "EMBEDDING_BASE_URL": "https://embedding.example/v1",
        "LLM_EMBEDDING_MODEL": "embedding-model",
    }


def _chat_response(content: str = "OK", *, tool_calls: list | None = None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls or [])
            )
        ]
    )


class _FakeClient:
    def __init__(self, mode: str = "ready") -> None:
        self.mode = mode
        self.closed = False
        self.tool_attempts = 0
        self.calls: list[tuple[str, dict]] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_chat)
        )
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(parse=self._parse_structured)
            )
        )
        self.embeddings = SimpleNamespace(create=self._create_embedding)

    async def _create_chat(self, **kwargs):
        self.calls.append(("chat", kwargs))
        if self.mode == "error":
            raise RuntimeError("secret-chat-key https://internal.provider/v1")
        if self.mode == "hang":
            await asyncio.Event().wait()
        if "tools" in kwargs:
            self.tool_attempts += 1
            if self.mode == "invalid" or (
                self.mode == "tool_second_attempt" and self.tool_attempts == 1
            ):
                return _chat_response("I will not call a tool")
            tool_call = SimpleNamespace(
                function=SimpleNamespace(
                    name="capability_probe", arguments='{"value":"ok"}'
                )
            )
            return _chat_response("", tool_calls=[tool_call])
        if kwargs.get("response_format") == {"type": "json_object"}:
            return _chat_response(
                "not-json" if self.mode == "invalid" else '{"ok":true}'
            )
        if self.mode == "invalid":
            return _chat_response("")
        return _chat_response("NOT OK" if self.mode == "wrong_chat" else "OK")

    async def _parse_structured(self, **kwargs):
        self.calls.append(("structured", kwargs))
        if self.mode == "error":
            raise RuntimeError("secret-structured-key")
        if self.mode == "invalid":
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=None))]
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(parsed=SimpleNamespace(ok=True))
                )
            ]
        )

    async def _create_embedding(self, **kwargs):
        self.calls.append(("embedding", kwargs))
        if self.mode == "error":
            raise RuntimeError("secret-embedding-key")
        vector = [] if self.mode == "invalid" else [0.25, -0.5]
        return SimpleNamespace(data=[SimpleNamespace(embedding=vector)])

    async def close(self):
        self.closed = True


class _RecordingFactory:
    def __init__(self, mode: str = "ready") -> None:
        self.mode = mode
        self.calls: list[tuple[object, dict]] = []
        self.clients: list[_FakeClient] = []

    def __call__(self, config, **kwargs):
        self.calls.append((config, kwargs))
        client = _FakeClient(self.mode)
        self.clients.append(client)
        return client


class TestProviderCapabilityChecker(unittest.IsolatedAsyncioTestCase):
    async def test_verifies_all_runtime_capabilities_without_exposing_credentials(self):
        factory = _RecordingFactory()
        checker = ProviderCapabilityChecker(
            config_loader=lambda: load_provider_configs(_provider_env()),
            client_factory=factory,
            timeout_seconds=1,
        )

        result = await checker.check()

        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["billable_requests_possible"])
        self.assertEqual(
            set(result["capabilities"]),
            {
                "chat",
                "chat_json_mode",
                "tool_calling_auto",
                "structured_output",
                "embedding",
            },
        )
        self.assertTrue(
            all(item["status"] == "ready" for item in result["capabilities"].values())
        )
        self.assertEqual(len(factory.calls), 5)
        self.assertTrue(all(kwargs["max_retries"] == 0 for _, kwargs in factory.calls))
        self.assertTrue(all(client.closed for client in factory.clients))

        tool_kwargs = next(
            kwargs
            for client in factory.clients
            for kind, kwargs in client.calls
            if kind == "chat" and "tools" in kwargs
        )
        self.assertEqual(tool_kwargs["tool_choice"], "auto")
        completion_calls = [
            kwargs
            for client in factory.clients
            for kind, kwargs in client.calls
            if kind in {"chat", "structured"}
        ]
        self.assertTrue(
            all(kwargs["max_tokens"] >= 128 for kwargs in completion_calls),
            "reasoning-capable providers need enough budget to reach the answer/tool call",
        )
        serialized = json.dumps(result)
        self.assertNotIn("secret-", serialized)
        self.assertNotIn("provider.example", serialized)

    async def test_skips_invalid_configuration_without_contacting_that_provider(self):
        env = _provider_env()
        env["LLM_EMBEDDING_MODEL"] = ""
        factory = _RecordingFactory()
        checker = ProviderCapabilityChecker(
            config_loader=lambda: load_provider_configs(env),
            client_factory=factory,
            timeout_seconds=1,
        )

        result = await checker.check()

        self.assertEqual(result["status"], "degraded")
        self.assertEqual(
            result["capabilities"]["embedding"]["code"],
            "configuration_invalid",
        )
        self.assertFalse(result["capabilities"]["embedding"]["attempted"])
        self.assertEqual(len(factory.calls), 4)

    async def test_rejects_transport_success_with_invalid_capability_responses(self):
        factory = _RecordingFactory(mode="invalid")
        checker = ProviderCapabilityChecker(
            config_loader=lambda: load_provider_configs(_provider_env()),
            client_factory=factory,
            timeout_seconds=1,
        )

        result = await checker.check()

        self.assertEqual(result["status"], "degraded")
        self.assertTrue(
            all(
                item["code"] == "invalid_response"
                for item in result["capabilities"].values()
            )
        )
        self.assertTrue(all(client.closed for client in factory.clients))

    async def test_auto_tool_behavior_gets_one_bounded_semantic_retry(self):
        factory = _RecordingFactory(mode="tool_second_attempt")
        checker = ProviderCapabilityChecker(
            config_loader=lambda: load_provider_configs(_provider_env()),
            client_factory=factory,
            timeout_seconds=1,
        )

        result = await checker.check()

        self.assertEqual(result["capabilities"]["tool_calling_auto"]["status"], "ready")
        tool_client = next(client for client in factory.clients if client.tool_attempts)
        self.assertEqual(tool_client.tool_attempts, 2)

    async def test_chat_requires_the_requested_token_not_merely_nonempty_text(self):
        factory = _RecordingFactory(mode="wrong_chat")
        checker = ProviderCapabilityChecker(
            config_loader=lambda: load_provider_configs(_provider_env()),
            client_factory=factory,
            timeout_seconds=1,
        )

        result = await checker.check()

        self.assertEqual(result["capabilities"]["chat"]["code"], "invalid_response")

    async def test_failures_are_sanitized_and_clients_are_closed(self):
        factory = _RecordingFactory(mode="error")
        checker = ProviderCapabilityChecker(
            config_loader=lambda: load_provider_configs(_provider_env()),
            client_factory=factory,
            timeout_seconds=1,
        )

        result = await checker.check()

        self.assertEqual(result["status"], "degraded")
        self.assertTrue(
            all(
                item["code"] == "request_failed"
                for item in result["capabilities"].values()
            )
        )
        self.assertNotIn("secret-", json.dumps(result))
        self.assertNotIn("internal.provider", json.dumps(result))
        self.assertTrue(all(client.closed for client in factory.clients))

    async def test_timeout_is_bounded_and_still_closes_the_client(self):
        factory = _RecordingFactory(mode="hang")
        checker = ProviderCapabilityChecker(
            config_loader=lambda: load_provider_configs(_provider_env()),
            client_factory=factory,
            timeout_seconds=0.01,
        )

        result = await asyncio.wait_for(checker.check(), timeout=0.2)

        self.assertEqual(result["capabilities"]["chat"]["code"], "request_timeout")
        self.assertTrue(all(client.closed for client in factory.clients))


if __name__ == "__main__":
    unittest.main(verbosity=2)
