"""
Live MCP server 接入单测（mock，不连真实 server，CI 友好）

覆盖：
  1) mcp_live_enabled env 开关（默认 false / true）
  2) _live_server_configs 默认配 duckduckgo-mcp-server（uvx，search + fetch_content）
  3) disabled 时 connect_and_register_all 返回 []（不连）
  4) enabled 时 mock 连接 + 注册流程；单 server 连接失败 fail-soft 不抛、不阻断启动

真实 live server 连接已由一次性 smoke 验证过：MCPClient 握手成功，工具 = ['search','fetch_content']。

运行：python -m pytest tests/test_mcp_servers.py -q
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import services.mcp_servers as ms


class TestMcpLiveEnabled(unittest.TestCase):
    def test_default_false(self):
        env = {k: v for k, v in os.environ.items() if k != "MCP_LIVE_ENABLED"}
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(ms.mcp_live_enabled())

    def test_true(self):
        with patch.dict(os.environ, {"MCP_LIVE_ENABLED": "true"}):
            self.assertTrue(ms.mcp_live_enabled())


class TestServerConfigs(unittest.TestCase):
    def test_default_ddg(self):
        cfgs = ms._live_server_configs()
        self.assertEqual(len(cfgs), 1)
        self.assertEqual(cfgs[0].server_name, "ddg")
        self.assertIn("duckduckgo-mcp-server", cfgs[0].args)
        self.assertEqual(
            cfgs[0].read_only_tools,
            frozenset({"search", "fetch_content"}),
        )


class TestConnectRegister(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        await ms.cleanup_all()

    async def test_disabled_returns_empty(self):
        with patch.dict(os.environ, {"MCP_LIVE_ENABLED": "false"}):
            self.assertEqual(await ms.connect_and_register_all(), [])
        self.assertEqual(ms._clients, [])

    async def test_enabled_registers(self):
        ms._clients.clear()
        with patch.dict(os.environ, {"MCP_LIVE_ENABLED": "true"}), \
             patch.object(ms, "MCPClient") as MockClient, \
             patch.object(ms, "register_mcp_tools_to_registry",
                          new=AsyncMock(return_value=["mcp_ddg_search", "mcp_ddg_fetch_content"])):
            inst = MockClient.return_value
            inst.connect = AsyncMock()
            inst.cleanup = AsyncMock()
            names = await ms.connect_and_register_all()
        self.assertEqual(names, ["mcp_ddg_search", "mcp_ddg_fetch_content"])
        self.assertEqual(len(ms._clients), 1)

    async def test_connect_fail_soft(self):
        ms._clients.clear()
        with patch.dict(os.environ, {"MCP_LIVE_ENABLED": "true"}), \
             patch.object(ms, "MCPClient") as MockClient:
            inst = MockClient.return_value
            inst.connect = AsyncMock(side_effect=RuntimeError("boom"))
            inst.cleanup = AsyncMock()
            names = await ms.connect_and_register_all()   # 不抛
        self.assertEqual(names, [])
        self.assertEqual(ms._clients, [])

    async def test_repeated_connect_replaces_and_cleans_previous_client(self):
        first = MagicMock()
        first.connect = AsyncMock()
        first.cleanup = AsyncMock()
        second = MagicMock()
        second.connect = AsyncMock()
        second.cleanup = AsyncMock()

        with patch.dict(os.environ, {"MCP_LIVE_ENABLED": "true"}), patch.object(
            ms, "MCPClient", side_effect=[first, second]
        ), patch.object(
            ms,
            "register_mcp_tools_to_registry",
            new=AsyncMock(return_value=["mcp_ddg_search"]),
        ):
            await ms.connect_and_register_all()
            names = await ms.connect_and_register_all()

        first.cleanup.assert_awaited_once_with()
        self.assertEqual(names, ["mcp_ddg_search"])
        self.assertEqual(ms._clients, [second])

    async def test_failed_reconnect_cleans_previous_and_new_clients(self):
        previous = MagicMock()
        previous.cleanup = AsyncMock()
        ms._clients.append(previous)

        failed = MagicMock()
        failed.connect = AsyncMock(side_effect=RuntimeError("boom"))
        failed.cleanup = AsyncMock()

        with patch.dict(os.environ, {"MCP_LIVE_ENABLED": "true"}), patch.object(
            ms, "MCPClient", return_value=failed
        ):
            names = await ms.connect_and_register_all()

        previous.cleanup.assert_awaited_once_with()
        failed.cleanup.assert_awaited_once_with()
        self.assertEqual(names, [])
        self.assertEqual(ms._clients, [])


if __name__ == "__main__":
    unittest.main()
