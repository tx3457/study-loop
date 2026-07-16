"""Provider configuration, startup safety, and zero-generation health checks."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import routers.health as health_router
import routers.chat as chat_router
from main import app
from services.provider_config import (
    ProviderConfig,
    ProviderConfigurationError,
    build_async_openai,
    load_provider_configs,
)
from services.provider_health import ProviderHealthChecker


class _FakeClient:
    def __init__(
        self,
        *,
        model_ids=(),
        error: Exception | None = None,
        hang=False,
        delay_seconds=0.0,
    ):
        self._model_ids = model_ids
        self._error = error
        self._hang = hang
        self._delay_seconds = delay_seconds
        self.closed = False
        self.models = SimpleNamespace(list=self._list)

    async def _list(self):
        if self._hang:
            await asyncio.Event().wait()
        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(
            data=[SimpleNamespace(id=model_id) for model_id in self._model_ids]
        )

    async def close(self):
        self.closed = True


class _RecordingFactory:
    def __init__(
        self,
        *,
        model_ids=(),
        error: Exception | None = None,
        hang=False,
        delay_seconds=0.0,
    ):
        self.model_ids = model_ids
        self.error = error
        self.hang = hang
        self.delay_seconds = delay_seconds
        self.calls: list[tuple[ProviderConfig, dict]] = []
        self.clients: list[_FakeClient] = []

    def __call__(self, config: ProviderConfig, **kwargs):
        self.calls.append((config, kwargs))
        client = _FakeClient(
            model_ids=self.model_ids,
            error=self.error,
            hang=self.hang,
            delay_seconds=self.delay_seconds,
        )
        self.clients.append(client)
        return client


def _shared_provider_env() -> dict[str, str]:
    return {
        "LLM_API_KEY": "secret-chat-key",
        "LLM_BASE_URL": "https://provider.example/v1",
        "LLM_MODEL": "chat-model",
        "STRUCTURED_API_KEY": "",
        "STRUCTURED_BASE_URL": "",
        "STRUCTURED_MODEL": "structured-model",
        "EMBEDDING_API_KEY": "",
        "EMBEDDING_BASE_URL": "",
        "LLM_EMBEDDING_MODEL": "embedding-model",
    }


class TestProviderConfig(unittest.IsolatedAsyncioTestCase):
    def test_effective_fallbacks_are_explicit_and_repr_hides_secrets(self):
        configs = load_provider_configs(_shared_provider_env())

        self.assertEqual(configs["structured"].inherited_fields, ("api_key", "base_url"))
        self.assertEqual(configs["embedding"].inherited_fields, ("api_key", "base_url"))
        self.assertEqual(configs["structured"].model, "structured-model")
        self.assertEqual(configs["embedding"].model, "embedding-model")
        self.assertTrue(all(config.configured for config in configs.values()))
        self.assertNotIn("secret-chat-key", repr(configs["chat"]))
        self.assertNotIn("provider.example", repr(configs["chat"]))

    def test_placeholders_and_unsafe_base_urls_are_rejected(self):
        placeholder = load_provider_configs(
            {
                "LLM_API_KEY": "replace-with-your-provider-key",
                "LLM_BASE_URL": "https://api.example.com/v1",
                "LLM_MODEL": "replace-with-chat-model",
            }
        )["chat"]
        self.assertFalse(placeholder.configured)
        self.assertEqual(
            placeholder.issues,
            ("api_key_placeholder", "base_url_placeholder", "model_placeholder"),
        )

        unsafe = ProviderConfig(
            capability="chat",
            api_key="key",
            base_url="https://user:pass@provider.example/v1?tenant=secret",
            model="model",
        )
        self.assertIn("base_url_invalid", unsafe.issues)

        malformed = ProviderConfig(
            capability="chat",
            api_key="key",
            base_url="https://provider.example:not-a-port/v1",
            model="model",
        )
        self.assertEqual(malformed.connection_issues, ("base_url_invalid",))
        client = build_async_openai(malformed)
        with self.assertRaises(ProviderConfigurationError):
            _ = client.models

    def test_endpoint_identity_normalizes_slashes_without_merging_keys(self):
        first = ProviderConfig(
            capability="chat",
            api_key="key-one",
            base_url="https://provider.example/v1",
            model="model",
        )
        trailing_slash = ProviderConfig(
            capability="structured",
            api_key="key-one",
            base_url="https://provider.example/v1/",
            model="model",
        )
        different_key = ProviderConfig(
            capability="embedding",
            api_key="key-two",
            base_url="https://provider.example/v1",
            model="model",
        )

        self.assertEqual(first.credential_identity, trailing_slash.credential_identity)
        self.assertNotEqual(first.credential_identity, different_key.credential_identity)

    def test_openai_sdk_environment_is_not_an_implicit_fallback(self):
        configs = load_provider_configs(
            {
                "OPENAI_API_KEY": "must-not-be-used",
                "OPENAI_BASE_URL": "https://api.openai.com/v1",
            }
        )

        self.assertIn("api_key_missing", configs["chat"].issues)
        self.assertIsNone(configs["chat"].credential_identity)

    def test_partial_dedicated_credentials_never_cross_provider_boundaries(self):
        env = _shared_provider_env()
        env["STRUCTURED_BASE_URL"] = "https://structured.example/v1"
        env["EMBEDDING_API_KEY"] = "embedding-only-key"

        configs = load_provider_configs(env)

        self.assertEqual(configs["structured"].connection_issues, ("api_key_missing",))
        self.assertEqual(configs["embedding"].connection_issues, ("base_url_missing",))
        self.assertIsNone(configs["structured"].credential_identity)
        self.assertIsNone(configs["embedding"].credential_identity)

    async def test_unconfigured_client_fails_locally_without_sdk_default(self):
        config = ProviderConfig(
            capability="chat", api_key=None, base_url=None, model=None
        )
        client = build_async_openai(config)

        with self.assertRaises(ProviderConfigurationError) as raised:
            _ = client.chat

        self.assertEqual(raised.exception.capability, "chat")
        self.assertEqual(
            raised.exception.issues,
            ("api_key_missing", "base_url_missing", "model_missing"),
        )
        await client.close()


class TestProviderHealthChecker(unittest.IsolatedAsyncioTestCase):
    async def test_deduplicates_shared_credentials_and_caches_catalog_result(self):
        factory = _RecordingFactory(
            model_ids=("chat-model", "structured-model", "embedding-model")
        )
        checker = ProviderHealthChecker(
            config_loader=lambda: load_provider_configs(_shared_provider_env()),
            client_factory=factory,
            ttl_seconds=60,
            timeout_seconds=1,
        )

        first = await checker.check()
        second = await checker.check()

        self.assertEqual(len(factory.calls), 1)
        _, kwargs = factory.calls[0]
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertFalse(kwargs["require_model"])
        self.assertEqual(first["status"], "ready")
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(first["probe"], "models.list")
        self.assertFalse(first["generation_or_embedding_called"])
        self.assertTrue(all(client.closed for client in factory.clients))
        for result in first["providers"].values():
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["capability_support"], "unverified")

    async def test_missing_model_is_misconfigured_but_catalog_can_be_reached(self):
        env = _shared_provider_env()
        env["LLM_EMBEDDING_MODEL"] = ""
        factory = _RecordingFactory(model_ids=("chat-model", "structured-model"))
        checker = ProviderHealthChecker(
            config_loader=lambda: load_provider_configs(env),
            client_factory=factory,
            ttl_seconds=60,
            timeout_seconds=1,
        )

        result = await checker.check()

        embedding = result["providers"]["embedding"]
        self.assertEqual(result["status"], "degraded")
        self.assertEqual(embedding["status"], "misconfigured")
        self.assertEqual(embedding["issues"], ["model_missing"])
        self.assertTrue(embedding["catalog_reachable"])
        self.assertEqual(len(factory.calls), 1)

    async def test_partial_dedicated_credentials_are_not_probed(self):
        env = _shared_provider_env()
        env["STRUCTURED_BASE_URL"] = "https://structured.example/v1"
        env["EMBEDDING_API_KEY"] = "embedding-only-key"
        factory = _RecordingFactory(model_ids=("chat-model",))
        checker = ProviderHealthChecker(
            config_loader=lambda: load_provider_configs(env),
            client_factory=factory,
            ttl_seconds=60,
            timeout_seconds=1,
        )

        result = await checker.check()

        self.assertEqual(len(factory.calls), 1)
        self.assertEqual(result["providers"]["chat"]["status"], "ready")
        self.assertEqual(result["providers"]["structured"]["status"], "misconfigured")
        self.assertEqual(result["providers"]["embedding"]["status"], "misconfigured")

    async def test_provider_failure_is_sanitized(self):
        factory = _RecordingFactory(
            error=RuntimeError("secret-chat-key https://internal.provider/v1")
        )
        checker = ProviderHealthChecker(
            config_loader=lambda: load_provider_configs(_shared_provider_env()),
            client_factory=factory,
            ttl_seconds=60,
            timeout_seconds=1,
        )

        result = await checker.check()
        serialized = json.dumps(result)

        self.assertEqual(result["status"], "degraded")
        self.assertNotIn("secret-chat-key", serialized)
        self.assertNotIn("internal.provider", serialized)
        for provider in result["providers"].values():
            self.assertEqual(provider["code"], "catalog_unavailable")

    async def test_unlisted_model_is_catalog_degraded_not_capability_failure(self):
        factory = _RecordingFactory(model_ids=("different-model",))
        checker = ProviderHealthChecker(
            config_loader=lambda: load_provider_configs(_shared_provider_env()),
            client_factory=factory,
            ttl_seconds=60,
            timeout_seconds=1,
        )

        result = await checker.check()

        self.assertEqual(result["status"], "degraded")
        for provider in result["providers"].values():
            self.assertEqual(provider["code"], "model_not_listed")
            self.assertTrue(provider["catalog_reachable"])
            self.assertFalse(provider["model_visible"])
            self.assertEqual(provider["capability_support"], "unverified")

    async def test_wall_clock_timeout_bounds_a_hung_sdk_call(self):
        factory = _RecordingFactory(hang=True)
        checker = ProviderHealthChecker(
            config_loader=lambda: load_provider_configs(_shared_provider_env()),
            client_factory=factory,
            ttl_seconds=60,
            timeout_seconds=0.01,
        )

        result = await asyncio.wait_for(checker.check(), timeout=0.2)

        self.assertEqual(result["status"], "degraded")
        self.assertTrue(factory.clients[0].closed)

    async def test_concurrent_cold_checks_singleflight_and_ttl_expiry(self):
        factory = _RecordingFactory(
            model_ids=("chat-model", "structured-model", "embedding-model"),
            delay_seconds=0.01,
        )
        now = [100.0]
        checker = ProviderHealthChecker(
            config_loader=lambda: load_provider_configs(_shared_provider_env()),
            client_factory=factory,
            ttl_seconds=5,
            timeout_seconds=1,
            clock=lambda: now[0],
        )

        results = await asyncio.gather(checker.check(), checker.check(), checker.check())
        self.assertEqual(len(factory.calls), 1)
        self.assertEqual(sum(bool(result["cached"]) for result in results), 2)

        now[0] += 6
        refreshed = await checker.check()
        self.assertEqual(len(factory.calls), 2)
        self.assertFalse(refreshed["cached"])


class TestProviderHealthApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)

    def test_liveness_never_invokes_provider_checker(self):
        checker = SimpleNamespace(check=AsyncMock())
        with patch.object(health_router, "provider_health_checker", checker):
            response = self.client.get("/health/live")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"name": "StudyLoop", "status": "ok"})
        checker.check.assert_not_awaited()

    def test_provider_health_uses_503_only_for_degraded_result(self):
        ready = {"status": "ready", "providers": {}}
        degraded = {"status": "degraded", "providers": {}}
        checker = SimpleNamespace(check=AsyncMock(side_effect=[ready, degraded]))

        with patch.object(health_router, "provider_health_checker", checker):
            first = self.client.get("/health/providers")
            second = self.client.get("/health/providers")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 503)

    def test_stream_provider_failure_happens_before_response_starts(self):
        error = ProviderConfigurationError("chat", ("api_key_missing",))
        with patch.object(chat_router, "chat_stream", AsyncMock(side_effect=error)):
            response = self.client.post("/chat/stream", json={"message": "hello"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {"error": "服务暂时不可用", "detail": "模型服务尚未正确配置"},
        )

    def test_application_starts_and_reports_503_without_provider_credentials(self):
        repo_root = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory(dir="/tmp") as chroma_dir:
            env = os.environ.copy()
            for name in (
                "LLM_API_KEY",
                "LLM_BASE_URL",
                "LLM_MODEL",
                "STRUCTURED_API_KEY",
                "STRUCTURED_BASE_URL",
                "STRUCTURED_MODEL",
                "EMBEDDING_API_KEY",
                "EMBEDDING_BASE_URL",
                "LLM_EMBEDDING_MODEL",
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
            ):
                env[name] = ""
            env["CHROMA_DIR"] = chroma_dir
            env["MCP_LIVE_ENABLED"] = "false"
            script = """
import socket

from fastapi.testclient import TestClient


def reject_network(*args, **kwargs):
    raise AssertionError("unexpected network access")


socket.socket.connect = reject_network

from main import app

client = TestClient(app, raise_server_exceptions=False)
assert client.get("/health/live").status_code == 200
assert client.get("/health/providers").status_code == 503
response = client.post("/chat", json={"message": "hello"})
assert response.status_code == 503, response.text
assert response.json()["detail"] == "模型服务尚未正确配置"
stream_response = client.post("/chat/stream", json={"message": "hello"})
assert stream_response.status_code == 503, stream_response.text
assert stream_response.json()["detail"] == "模型服务尚未正确配置"
print("startup-safe")
"""
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=repo_root,
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("startup-safe", completed.stdout)
