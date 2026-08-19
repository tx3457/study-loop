"""FastAPI lifespan ordering and shared provider client lifecycle."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import main
from services.provider_config import (
    ProviderConfig,
    ProviderConfigurationError,
    build_managed_async_openai,
)


class _FakeSdkClient:
    def __init__(self) -> None:
        self.chat = object()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _RecordingClientFactory:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.clients: list[_FakeSdkClient] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        client = _FakeSdkClient()
        self.clients.append(client)
        return client


def _configured_chat() -> ProviderConfig:
    return ProviderConfig(
        capability="chat",
        api_key="test-key",
        base_url="http://127.0.0.1:9/v1",
        model="test-model",
    )


class TestManagedProviderClient(unittest.IsolatedAsyncioTestCase):
    async def test_client_is_lazy_closes_and_recreates_behind_stable_proxy(self):
        factory = _RecordingClientFactory()
        client = build_managed_async_openai(
            _configured_chat(), client_factory=factory
        )

        self.assertFalse(client.is_initialized)
        self.assertEqual(factory.calls, [])

        first_chat_resource = client.chat
        self.assertTrue(client.is_initialized)
        self.assertEqual(len(factory.clients), 1)
        self.assertEqual(factory.calls[0]["max_retries"], 0)

        await client.close()
        self.assertTrue(factory.clients[0].closed)
        self.assertFalse(client.is_initialized)

        second_chat_resource = client.chat
        self.assertIsNot(first_chat_resource, second_chat_resource)
        self.assertEqual(len(factory.clients), 2)
        await client.close()

    async def test_unconfigured_managed_client_closes_without_initializing(self):
        factory = _RecordingClientFactory()
        client = build_managed_async_openai(
            ProviderConfig(
                capability="chat", api_key=None, base_url=None, model=None
            ),
            client_factory=factory,
        )

        await client.close()
        self.assertEqual(factory.calls, [])
        with self.assertRaises(ProviderConfigurationError):
            _ = client.chat


class TestApplicationLifespan(unittest.TestCase):
    def test_lifespan_runs_startup_and_reverse_order_cleanup_each_time(self):
        events: list[str] = []

        async def load_memory():
            events.append("memory")

        async def connect_mcp():
            events.append("mcp_connect")

        async def cleanup_mcp():
            events.append("mcp_cleanup")

        async def save_memory():
            events.append("memory_save")

        async def close_providers():
            events.append("providers_close")

        with patch.object(main, "_load_memory_snapshot", new=load_memory), patch.object(
            main, "_connect_mcp_live_servers", new=connect_mcp
        ), patch.object(
            main, "_cleanup_mcp_live_servers", new=cleanup_mcp
        ), patch.object(
            main, "_save_memory_snapshot", new=save_memory
        ), patch.object(
            main, "close_managed_provider_clients", new=close_providers
        ):
            for _ in range(2):
                with TestClient(main.app) as client:
                    response = client.get("/health/live")
                    self.assertEqual(response.status_code, 200)

        self.assertEqual(
            events,
            [
                "memory",
                "mcp_connect",
                "mcp_cleanup",
                "memory_save",
                "providers_close",
                "memory",
                "mcp_connect",
                "mcp_cleanup",
                "memory_save",
                "providers_close",
            ],
        )
        self.assertEqual(main.app.router.on_startup, [])
        self.assertEqual(main.app.router.on_shutdown, [])

    def test_lifespan_closes_registered_client_and_next_cycle_recreates_it(self):
        factory = _RecordingClientFactory()
        managed = build_managed_async_openai(
            _configured_chat(), client_factory=factory
        )
        _ = managed.chat

        async def noop():
            return None

        with patch.object(main, "_load_memory_snapshot", new=noop), patch.object(
            main, "_connect_mcp_live_servers", new=noop
        ), patch.object(
            main, "_cleanup_mcp_live_servers", new=noop
        ), patch.object(main, "_save_memory_snapshot", new=noop):
            with TestClient(main.app):
                pass
            self.assertTrue(factory.clients[0].closed)
            self.assertFalse(managed.is_initialized)

            _ = managed.chat
            self.assertEqual(len(factory.clients), 2)
            with TestClient(main.app):
                pass

        self.assertTrue(factory.clients[1].closed)
        self.assertFalse(managed.is_initialized)

    def test_startup_helpers_fail_soft_and_shutdown_still_runs(self):
        from services import mcp_servers

        connect_mcp = AsyncMock(side_effect=RuntimeError("offline"))
        cleanup_mcp = AsyncMock()
        save_memory = AsyncMock()
        close_providers = AsyncMock()

        with patch.object(
            main, "load_snapshot", side_effect=RuntimeError("corrupt snapshot")
        ), patch.object(
            mcp_servers, "connect_and_register_all", new=connect_mcp
        ), patch.object(
            mcp_servers, "cleanup_all", new=cleanup_mcp
        ), patch.object(
            main, "close_managed_provider_clients", new=close_providers
        ), patch.object(
            main, "_save_memory_snapshot", new=save_memory
        ):
            with TestClient(main.app) as client:
                response = client.get("/health/live")

        self.assertEqual(response.status_code, 200)
        connect_mcp.assert_awaited_once_with()
        cleanup_mcp.assert_awaited_once_with()
        save_memory.assert_awaited_once_with()
        close_providers.assert_awaited_once_with()

    def test_startup_cancellation_releases_resources(self):
        events: list[str] = []

        async def load_memory():
            events.append("memory")

        async def cancel_connect():
            events.append("mcp_connect")
            raise asyncio.CancelledError

        async def cleanup_mcp():
            events.append("mcp_cleanup")

        async def close_providers():
            events.append("providers_close")

        async def save_memory():
            events.append("memory_save")

        async def run_lifespan():
            async with main.lifespan(main.app):
                self.fail("cancelled startup must not enter the application context")

        with patch.object(main, "_load_memory_snapshot", new=load_memory), patch.object(
            main, "_connect_mcp_live_servers", new=cancel_connect
        ), patch.object(
            main, "_cleanup_mcp_live_servers", new=cleanup_mcp
        ), patch.object(
            main, "_save_memory_snapshot", new=save_memory
        ), patch.object(
            main, "close_managed_provider_clients", new=close_providers
        ):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(run_lifespan())

        self.assertEqual(
            events,
            [
                "memory",
                "mcp_connect",
                "mcp_cleanup",
                "memory_save",
                "providers_close",
            ],
        )
