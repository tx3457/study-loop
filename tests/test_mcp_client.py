"""
单测：MCP 客户端 + ToolRegistry 桥接

不连真实 MCP server(无外部依赖),用 mock ClientSession 验证:
1. list_tools 返回 server.list_tools() 的 tools 字段
2. call_tool 拼接 TextContent 文本片段
3. 未 connect 前调用抛 RuntimeError
4. register_mcp_tools_to_registry 把 N 个 MCP 工具注册到 ToolRegistry
5. 闭包陷阱:多个 tool 注册后,各自 handler 调用各自对应的 MCP tool name

跑法:
  python -m pytest tests/test_mcp_client.py -q
"""
import json
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

    async def test_error_result_fails_closed_without_exposing_remote_content(self):
        from services.mcp_client import (
            MCPClient,
            MCPToolExecutionError,
            StdioMCPServerConfig,
        )

        secret = "SENSITIVE_MCP_RESULT_SENTINEL_d2f9"
        client = MCPClient(
            StdioMCPServerConfig(server_name="test", command="echo", args=[])
        )
        session = MagicMock()
        session.call_tool = AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(text=f"remote traceback: {secret}")],
            isError=True,
        ))
        client._session = session
        client._initialized = True

        with self.assertLogs("services.mcp_client", level="WARNING") as captured:
            with self.assertRaises(MCPToolExecutionError) as raised:
                await client.call_tool("explode", {})

        self.assertEqual(str(raised.exception), MCPToolExecutionError.code)
        observable = str(raised.exception) + "\n" + "\n".join(captured.output)
        self.assertNotIn(secret, observable)

    async def test_transport_exception_is_wrapped_without_exposing_remote_message(self):
        from services.mcp_client import (
            MCPClient,
            MCPToolExecutionError,
            StdioMCPServerConfig,
        )

        secret = "SENSITIVE_MCP_TRANSPORT_SENTINEL_b73c"
        client = MCPClient(
            StdioMCPServerConfig(server_name="test", command="echo", args=[])
        )
        session = MagicMock()
        session.call_tool = AsyncMock(side_effect=RuntimeError(f"transport: {secret}"))
        client._session = session
        client._initialized = True

        with self.assertLogs("services.mcp_client", level="WARNING") as captured:
            with self.assertRaises(MCPToolExecutionError) as raised:
                await client.call_tool("explode", {})

        self.assertEqual(str(raised.exception), MCPToolExecutionError.code)
        observable = str(raised.exception) + "\n" + "\n".join(captured.output)
        self.assertNotIn(secret, observable)


class TestRegisterMCPToolsToRegistry(unittest.IsolatedAsyncioTestCase):
    """ToolRegistry 桥接:批量注册 + 闭包正确性"""

    async def asyncSetUp(self):
        from services.mcp_client import mcp_registry

        self._original_tools = dict(mcp_registry._tools)
        mcp_registry._tools.clear()

    async def asyncTearDown(self):
        from services.mcp_client import mcp_registry

        mcp_registry._tools.clear()
        mcp_registry._tools.update(self._original_tools)

    async def test_registers_all_tools_with_prefix(self):
        from services.mcp_client import (
            MCPClient, StdioMCPServerConfig, register_mcp_tools_to_registry,
        )
        from services.mcp_client import mcp_registry

        cfg = StdioMCPServerConfig(server_name="fs", command="echo", args=[])
        client = MCPClient(cfg)
        mock_tools = [_make_mock_tool("read"), _make_mock_tool("write")]
        client._session = _make_mock_session(mock_tools)
        client._initialized = True

        names = await register_mcp_tools_to_registry(client)
        self.assertEqual(set(names), {"mcp_fs_read", "mcp_fs_write"})
        self.assertTrue(mcp_registry.has("mcp_fs_read"))
        self.assertTrue(mcp_registry.has("mcp_fs_write"))
        self.assertEqual(mcp_registry.get("mcp_fs_read").metadata.max_retries, 0)
        self.assertEqual(mcp_registry.get("mcp_fs_write").metadata.max_retries, 0)
        self.assertEqual(
            mcp_registry.get("mcp_fs_read").metadata.effect_mode.value,
            "unknown",
        )
        self.assertEqual(
            mcp_registry.get("mcp_fs_write").metadata.effect_mode.value,
            "unknown",
        )

    async def test_only_locally_allowlisted_tool_is_read_only(self):
        from services.mcp_client import (
            MCPClient,
            MCPToolExecutionError,
            StdioMCPServerConfig,
            register_mcp_tools_to_registry,
        )
        from services.mcp_client import mcp_registry

        client = MCPClient(StdioMCPServerConfig(
            server_name="ddg",
            command="echo",
            read_only_tools=frozenset({"search"}),
        ))
        session = MagicMock()
        session.list_tools = AsyncMock(return_value=SimpleNamespace(
            tools=[_make_mock_tool("search"), _make_mock_tool("write")]
        ))
        session.call_tool = AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(text="remote failure")],
            isError=True,
        ))
        client._session = session
        client._initialized = True

        await register_mcp_tools_to_registry(client)

        search = mcp_registry.get("mcp_ddg_search")
        write = mcp_registry.get("mcp_ddg_write")
        self.assertEqual(search.metadata.effect_mode.value, "read_only")
        self.assertEqual(write.metadata.effect_mode.value, "unknown")
        with self.assertRaises(MCPToolExecutionError):
            await mcp_registry.invoke("mcp_ddg_search", {})

    async def test_invalid_remote_tool_name_is_rejected_before_registration(self):
        from services.mcp_client import (
            MCPClient,
            StdioMCPServerConfig,
            register_mcp_tools_to_registry,
        )
        from services.mcp_client import mcp_registry

        secret = "forged-log-entry"
        client = MCPClient(
            StdioMCPServerConfig(server_name="ddg", command="echo")
        )
        client._session = _make_mock_session([
            _make_mock_tool("search"),
            _make_mock_tool(f"bad\n{secret}"),
        ])
        client._initialized = True

        with self.assertRaisesRegex(ValueError, "invalid MCP tool definition"):
            await register_mcp_tools_to_registry(client)

        self.assertFalse(mcp_registry.has("mcp_ddg_search"))
        self.assertFalse(mcp_registry.has(f"mcp_ddg_bad\n{secret}"))

    async def test_closure_routes_to_correct_mcp_tool_name(self):
        """关键:多 tool 注册后,各自 handler 调对应 MCP name 而不是最后一个"""
        from services.mcp_client import (
            MCPClient, StdioMCPServerConfig, register_mcp_tools_to_registry,
        )
        from services.mcp_client import mcp_registry

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
        read_tool = mcp_registry.get("mcp_fs_read")
        write_tool = mcp_registry.get("mcp_fs_write")
        ls_tool = mcp_registry.get("mcp_fs_ls")
        await read_tool.handler(path="/x")
        await write_tool.handler(path="/y", content="z")
        await ls_tool.handler()

        # 关键 assertion:闭包没踩坑,各自 handler 调各自的 mcp tool name
        self.assertEqual([log[0] for log in call_log], ["read", "write", "ls"])
        self.assertEqual(call_log[0][1], {"path": "/x"})
        self.assertEqual(call_log[1][1], {"path": "/y", "content": "z"})

    async def test_remote_error_content_never_reaches_registry_audit_or_logs(self):
        from services.mcp_client import (
            MCPClient,
            StdioMCPServerConfig,
            register_mcp_tools_to_registry,
        )
        from services.mcp_client import mcp_registry
        from services.tool_registry import SideEffectAmbiguousError

        secret = "SENSITIVE_MCP_AUDIT_SENTINEL_035e"
        run_id = "mcp-error-redaction"
        client = MCPClient(
            StdioMCPServerConfig(server_name="fs", command="echo", args=[])
        )
        session = MagicMock()
        session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=[_make_mock_tool("read")])
        )
        session.call_tool = AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(text=f"remote traceback: {secret}")],
            isError=True,
        ))
        client._session = session
        client._initialized = True
        await register_mcp_tools_to_registry(client)

        with self.assertLogs(level="WARNING") as captured:
            with self.assertRaises(SideEffectAmbiguousError):
                await mcp_registry.invoke("mcp_fs_read", {}, run_id=run_id)

        audit = [record.to_dict() for record in mcp_registry.get_audit(run_id=run_id)]
        observable = json.dumps(audit, ensure_ascii=False) + "\n" + "\n".join(captured.output)
        self.assertNotIn(secret, observable)
        self.assertEqual(audit[0]["error_message"], "handler_error:MCPToolExecutionError")
        self.assertIsNone(audit[0]["output_preview"])

    async def test_cleanup_unregisters_tools_owned_by_client(self):
        from services.mcp_client import (
            MCPClient, StdioMCPServerConfig, register_mcp_tools_to_registry,
        )
        from services.mcp_client import mcp_registry

        client = MCPClient(
            StdioMCPServerConfig(server_name="fs", command="echo", args=[])
        )
        client._session = _make_mock_session([_make_mock_tool("read")])
        client._initialized = True

        await register_mcp_tools_to_registry(client)
        self.assertTrue(mcp_registry.has("mcp_fs_read"))

        await client.cleanup()

        self.assertFalse(mcp_registry.has("mcp_fs_read"))
        self.assertEqual(client._registered_tools, {})
        self.assertIsNone(client._session)
        self.assertFalse(client._initialized)

    async def test_old_client_cleanup_preserves_new_tool_owner(self):
        from services.mcp_client import (
            MCPClient, StdioMCPServerConfig, register_mcp_tools_to_registry,
        )
        from services.mcp_client import mcp_registry

        config = StdioMCPServerConfig(server_name="fs", command="echo", args=[])
        old_client = MCPClient(config)
        old_client._session = _make_mock_session([_make_mock_tool("read")])
        old_client._initialized = True
        await register_mcp_tools_to_registry(old_client)
        old_tool = mcp_registry.get("mcp_fs_read")

        new_client = MCPClient(config)
        new_client._session = _make_mock_session([_make_mock_tool("read")])
        new_client._initialized = True
        await register_mcp_tools_to_registry(new_client)
        new_tool = mcp_registry.get("mcp_fs_read")
        self.assertIsNot(old_tool, new_tool)

        await old_client.cleanup()
        self.assertIs(mcp_registry.get("mcp_fs_read"), new_tool)

        await new_client.cleanup()
        self.assertFalse(mcp_registry.has("mcp_fs_read"))

    async def test_partial_registration_can_be_rolled_back_by_cleanup(self):
        from services.mcp_client import (
            MCPClient, StdioMCPServerConfig, register_mcp_tools_to_registry,
        )
        from services.mcp_client import mcp_registry

        client = MCPClient(
            StdioMCPServerConfig(server_name="fs", command="echo", args=[])
        )
        client._session = _make_mock_session(
            [_make_mock_tool("read"), _make_mock_tool("write")]
        )
        client._initialized = True
        original_register = mcp_registry.register

        def fail_on_second_tool(tool):
            if tool.name == "mcp_fs_write":
                raise RuntimeError("registry unavailable")
            original_register(tool)

        with patch.object(mcp_registry, "register", side_effect=fail_on_second_tool):
            with self.assertRaisesRegex(RuntimeError, "registry unavailable"):
                await register_mcp_tools_to_registry(client)

        self.assertTrue(mcp_registry.has("mcp_fs_read"))
        await client.cleanup()
        self.assertFalse(mcp_registry.has("mcp_fs_read"))
        self.assertFalse(mcp_registry.has("mcp_fs_write"))


class TestMCPToolsStayOutOfTheModelFacingSurface(unittest.IsolatedAsyncioTestCase):
    """A remote tool must never appear in a list the model chooses from.

    knowledge_web owns the security contract for public web access (SSRF checks,
    per-hop redirect validation, pinned addresses, body limits). A raw MCP fetch
    tool enforces none of that, so publishing it to the global registry would let
    any agent path reach arbitrary URLs straight through the remote server.
    """

    async def test_registered_mcp_tools_are_absent_from_every_model_facing_list(self):
        from services.mcp_client import (
            MCPClient,
            StdioMCPServerConfig,
            mcp_registry,
            register_mcp_tools_to_registry,
        )
        from services.tools import (
            allowed_tool_names,
            get_read_only_tool_capabilities,
            get_tool_definitions,
            replay_safe_tool_names,
        )

        client = MCPClient(StdioMCPServerConfig(
            server_name="ddg",
            command="echo",
            read_only_tools=frozenset({"search", "fetch_content"}),
        ))
        client._session = _make_mock_session(
            [_make_mock_tool("search"), _make_mock_tool("fetch_content")]
        )
        client._initialized = True

        names = await register_mcp_tools_to_registry(client)
        try:
            self.assertEqual(set(names), {"mcp_ddg_search", "mcp_ddg_fetch_content"})
            # Registered and callable by the application adapter ...
            self.assertTrue(mcp_registry.has("mcp_ddg_search"))
            self.assertTrue(mcp_registry.has("mcp_ddg_fetch_content"))

            # ... but invisible to every surface a model picks tools from.
            exposed = {d["function"]["name"] for d in get_tool_definitions()}
            read_only_schemas, read_only_names = get_read_only_tool_capabilities()
            exposed_read_only = {d["function"]["name"] for d in read_only_schemas}
            allowed = allowed_tool_names()
            replay_safe = replay_safe_tool_names()

            # Control: an empty surface would make every assertion below pass
            # for the wrong reason.
            self.assertIn("search_document", exposed)
            self.assertIn("search_document", allowed)
            self.assertTrue(replay_safe)

            for tool_name in names:
                self.assertNotIn(tool_name, exposed)
                self.assertNotIn(tool_name, allowed)
                self.assertNotIn(tool_name, read_only_names)
                self.assertNotIn(tool_name, exposed_read_only)
                # replay_safe_tool_names is what the interrupt-capable
                # assistant path offers the model.
                self.assertNotIn(tool_name, replay_safe)
        finally:
            await client.cleanup()

        self.assertFalse(mcp_registry.has("mcp_ddg_search"))
        self.assertFalse(mcp_registry.has("mcp_ddg_fetch_content"))


if __name__ == "__main__":
    unittest.main()
