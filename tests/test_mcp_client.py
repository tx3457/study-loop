"""
单测:MCP 客户端 + ToolRegistry 桥接(借鉴 letta/services/mcp/base_client.py)

不连真实 MCP server(无外部依赖),用 mock ClientSession 验证:
1. list_tools 返回 server.list_tools() 的 tools 字段
2. call_tool 拼接 TextContent 文本片段
3. 未 connect 前调用抛 RuntimeError
4. register_mcp_tools_to_registry 把 N 个 MCP 工具注册到 ToolRegistry
5. 闭包陷阱:多个 tool 注册后,各自 handler 调用各自对应的 MCP tool name

跑法:
  /path/to/python test/test_mcp_client.py -v
"""
import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_mock_session(tools_to_return: list, call_result_text: str = "ok"):
    """构造 mock ClientSession,模拟 list_tools / call_tool"""
    session = MagicMock()
    session.initialize = AsyncMock(return_value=None)
    session.list_tools = AsyncMock(
        return_value=SimpleNamespace(tools=tools_to_return)
    )
    session.call_tool = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(text=call_result_text)],
            isError=False,
        )
    )
    return session


def _make_mock_tool(name: str, description: str = "", schema: dict | None = None):
    """构造 mock MCP Tool 对象"""
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


class TestMCPClientCore(unittest.IsolatedAsyncioTestCase):
    """核心契约:list_tools / call_tool / 未连接保护"""

    async def test_not_connected_raises_runtime_error(self):
        from services.mcp_client import MCPClient, StdioMCPServerConfig
        cfg = StdioMCPServerConfig(server_name="test", command="echo", args=["hi"])
        client = MCPClient(cfg)
        with self.assertRaises(RuntimeError):
            await client.list_tools()
        with self.assertRaises(RuntimeError):
            await client.call_tool("foo", {})

    async def test_list_tools_returns_server_tools(self):
        from services.mcp_client import MCPClient, StdioMCPServerConfig
        cfg = StdioMCPServerConfig(server_name="test", command="echo", args=[])
        client = MCPClient(cfg)

        mock_tools = [_make_mock_tool("a"), _make_mock_tool("b")]
        client._session = _make_mock_session(mock_tools)
        client._initialized = True

        tools = await client.list_tools()
        self.assertEqual([t.name for t in tools], ["a", "b"])

    async def test_call_tool_joins_text_contents(self):
        from services.mcp_client import MCPClient, StdioMCPServerConfig
        cfg = StdioMCPServerConfig(server_name="test", command="echo", args=[])
        client = MCPClient(cfg)

        session = MagicMock()
        session.call_tool = AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(text="hello"), SimpleNamespace(text="world")],
            isError=False,
        ))
        client._session = session
        client._initialized = True

        result = await client.call_tool("greet", {"who": "x"})
        self.assertEqual(result, "hello world")
        session.call_tool.assert_awaited_once_with("greet", {"who": "x"})


class TestRegisterMCPToolsToRegistry(unittest.IsolatedAsyncioTestCase):
    """ToolRegistry 桥接:批量注册 + 闭包正确性"""

    async def asyncSetUp(self):
        from services.tool_registry import tool_registry

        self._original_tools = dict(tool_registry._tools)
        tool_registry._tools.clear()

    async def asyncTearDown(self):
        from services.tool_registry import tool_registry

        tool_registry._tools.clear()
        tool_registry._tools.update(self._original_tools)

    async def test_registers_all_tools_with_prefix(self):
        from services.mcp_client import (
            MCPClient, StdioMCPServerConfig, register_mcp_tools_to_registry,
        )
        from services.tool_registry import tool_registry

        cfg = StdioMCPServerConfig(server_name="fs", command="echo", args=[])
        client = MCPClient(cfg)
        mock_tools = [_make_mock_tool("read"), _make_mock_tool("write")]
        client._session = _make_mock_session(mock_tools)
        client._initialized = True

        names = await register_mcp_tools_to_registry(client)
        self.assertEqual(set(names), {"mcp_fs_read", "mcp_fs_write"})
        self.assertTrue(tool_registry.has("mcp_fs_read"))
        self.assertTrue(tool_registry.has("mcp_fs_write"))
        self.assertEqual(tool_registry.get("mcp_fs_read").metadata.max_retries, 0)
        self.assertEqual(tool_registry.get("mcp_fs_write").metadata.max_retries, 0)
        self.assertEqual(
            tool_registry.get("mcp_fs_read").metadata.effect_mode.value,
            "unknown",
        )
        self.assertEqual(
            tool_registry.get("mcp_fs_write").metadata.effect_mode.value,
            "unknown",
        )

    async def test_closure_routes_to_correct_mcp_tool_name(self):
        """关键:多 tool 注册后,各自 handler 调对应 MCP name 而不是最后一个"""
        from services.mcp_client import (
            MCPClient, StdioMCPServerConfig, register_mcp_tools_to_registry,
        )
        from services.tool_registry import tool_registry

        cfg = StdioMCPServerConfig(server_name="fs", command="echo", args=[])
        client = MCPClient(cfg)
        mock_tools = [_make_mock_tool("read"), _make_mock_tool("write"), _make_mock_tool("ls")]

        # 用真实 mock session 记录 call_tool 收到的 tool_name
        call_log: list[tuple[str, dict]] = []

        session = MagicMock()
        session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=mock_tools))

        async def _record_call(name, args):
            call_log.append((name, args))
            return SimpleNamespace(content=[SimpleNamespace(text="ok")], isError=False)

        session.call_tool = _record_call
        client._session = session
        client._initialized = True

        await register_mcp_tools_to_registry(client)

        # 分别触发 3 个 handler,验证 call_tool 收到的 name 各不相同
        read_tool = tool_registry.get("mcp_fs_read")
        write_tool = tool_registry.get("mcp_fs_write")
        ls_tool = tool_registry.get("mcp_fs_ls")
        await read_tool.handler(path="/x")
        await write_tool.handler(path="/y", content="z")
        await ls_tool.handler()

        # 关键 assertion:闭包没踩坑,各自 handler 调各自的 mcp tool name
        self.assertEqual([log[0] for log in call_log], ["read", "write", "ls"])
        self.assertEqual(call_log[0][1], {"path": "/x"})
        self.assertEqual(call_log[1][1], {"path": "/y", "content": "z"})

    async def test_cleanup_unregisters_tools_owned_by_client(self):
        from services.mcp_client import (
            MCPClient, StdioMCPServerConfig, register_mcp_tools_to_registry,
        )
        from services.tool_registry import tool_registry

        client = MCPClient(
            StdioMCPServerConfig(server_name="fs", command="echo", args=[])
        )
        client._session = _make_mock_session([_make_mock_tool("read")])
        client._initialized = True

        await register_mcp_tools_to_registry(client)
        self.assertTrue(tool_registry.has("mcp_fs_read"))

        await client.cleanup()

        self.assertFalse(tool_registry.has("mcp_fs_read"))
        self.assertEqual(client._registered_tools, {})
        self.assertIsNone(client._session)
        self.assertFalse(client._initialized)

    async def test_old_client_cleanup_preserves_new_tool_owner(self):
        from services.mcp_client import (
            MCPClient, StdioMCPServerConfig, register_mcp_tools_to_registry,
        )
        from services.tool_registry import tool_registry

        config = StdioMCPServerConfig(server_name="fs", command="echo", args=[])
        old_client = MCPClient(config)
        old_client._session = _make_mock_session([_make_mock_tool("read")])
        old_client._initialized = True
        await register_mcp_tools_to_registry(old_client)
        old_tool = tool_registry.get("mcp_fs_read")

        new_client = MCPClient(config)
        new_client._session = _make_mock_session([_make_mock_tool("read")])
        new_client._initialized = True
        await register_mcp_tools_to_registry(new_client)
        new_tool = tool_registry.get("mcp_fs_read")
        self.assertIsNot(old_tool, new_tool)

        await old_client.cleanup()
        self.assertIs(tool_registry.get("mcp_fs_read"), new_tool)

        await new_client.cleanup()
        self.assertFalse(tool_registry.has("mcp_fs_read"))

    async def test_partial_registration_can_be_rolled_back_by_cleanup(self):
        from services.mcp_client import (
            MCPClient, StdioMCPServerConfig, register_mcp_tools_to_registry,
        )
        from services.tool_registry import tool_registry

        client = MCPClient(
            StdioMCPServerConfig(server_name="fs", command="echo", args=[])
        )
        client._session = _make_mock_session(
            [_make_mock_tool("read"), _make_mock_tool("write")]
        )
        client._initialized = True
        original_register = tool_registry.register

        def fail_on_second_tool(tool):
            if tool.name == "mcp_fs_write":
                raise RuntimeError("registry unavailable")
            original_register(tool)

        with patch.object(tool_registry, "register", side_effect=fail_on_second_tool):
            with self.assertRaisesRegex(RuntimeError, "registry unavailable"):
                await register_mcp_tools_to_registry(client)

        self.assertTrue(tool_registry.has("mcp_fs_read"))
        await client.cleanup()
        self.assertFalse(tool_registry.has("mcp_fs_read"))
        self.assertFalse(tool_registry.has("mcp_fs_write"))


if __name__ == "__main__":
    unittest.main()
