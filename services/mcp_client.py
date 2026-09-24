"""
MCP(Model Context Protocol)客户端 + ToolRegistry 桥接

精简范围:
  - 只实现 stdio transport(MCP 主流模式,本地子进程,无网络/OAuth)
  - 不实现 SSE / HTTP / OAuth

集成方式:
  1. MCPClient.connect() 建立 stdio 子进程连接
  2. register_mcp_tools_to_registry() 把 MCP server 暴露的工具批量
     注册到 StudyLoop 的 ToolRegistry,统一走 retry / timeout / audit
  3. MCPClient.cleanup() 释放子进程

为什么不直接调 mcp.ClientSession?
  把 MCP tool 接到 ToolRegistry 后,业务侧调用与原生工具完全一致
  (services/tools.py:dispatch_tool 一行),不需要每个调用点 if/else 分流。
"""
import logging
import re
from contextlib import AsyncExitStack
from typing import Optional

from pydantic import BaseModel, Field

from services.tool_registry import EffectMode, Tool, ToolMetadata, ToolRegistry

logger = logging.getLogger(__name__)
_FUNCTION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# MCP tools live in a registry of their own, never the global one. A remote
# server's tools are reachable only through an application adapter that owns the
# security contract for that capability (see services.knowledge_web.search_web).
# Registering them globally would publish them to every model-facing tool list
# (get_tool_definitions / allowed_tool_names), letting any agent path call a raw
# remote fetch tool that bypasses the SSRF, redirect and body limits enforced here.
mcp_registry = ToolRegistry.isolated()


class MCPToolExecutionError(RuntimeError):
    """Stable, public-safe failure raised for any remote MCP tool error."""

    code = "mcp_tool_execution_failed"

    def __init__(self):
        super().__init__(self.code)


class StdioMCPServerConfig(BaseModel):
    """MCP server 启动配置(stdio transport)"""
    server_name: str = Field(
        ...,
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="server 唯一名,用作 tool name 前缀",
    )
    command: str = Field(..., description="启动命令,如 'uvx' / 'npx' / 'python'")
    args: list[str] = Field(default_factory=list, description="启动参数")
    env: Optional[dict[str, str]] = Field(None, description="环境变量")
    read_only_tools: frozenset[str] = Field(
        default_factory=frozenset,
        description="由本地配置确认无副作用、可安全失败降级的工具名",
    )


class MCPClient:
    """MCP 客户端(stdio transport)。生命周期:connect → list_tools/call_tool → cleanup"""

    def __init__(self, config: StdioMCPServerConfig):
        self.config = config
        self._exit_stack = AsyncExitStack()
        self._session = None  # type: ignore[assignment]
        self._initialized = False
        self._registered_tools: dict[str, Tool] = {}

    async def connect(self) -> None:
        """建立 stdio 子进程连接 + MCP 协议握手"""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as e:
            raise RuntimeError(f"mcp package not installed: {e}. pip install mcp") from e

        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env,
        )
        stdio_transport = await self._exit_stack.enter_async_context(stdio_client(params))
        read, write = stdio_transport
        self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        self._initialized = True
        logger.info(f"[mcp_client] connected: {self.config.server_name}")

    def _check_initialized(self) -> None:
        if not self._initialized or self._session is None:
            raise RuntimeError(f"MCPClient '{self.config.server_name}' not connected, call connect() first")

    async def list_tools(self) -> list:
        """列出 MCP server 暴露的所有工具(返回 mcp.Tool 对象列表)"""
        self._check_initialized()
        response = await self._session.list_tools()
        return list(response.tools)

    async def call_tool(self, tool_name: str, args: dict) -> str:
        """调用一个 MCP 工具,返回 stringified 结果(适配 ToolRegistry 的 handler 协议)"""
        self._check_initialized()
        try:
            result = await self._session.call_tool(tool_name, args)
        except Exception as exc:
            logger.warning(
                "[mcp_client] call_tool failed: error_type=%s",
                type(exc).__name__,
            )
            raise MCPToolExecutionError() from None

        # MCP error results often carry a human-readable remote exception in
        # TextContent.  Treating that content as a normal tool result would feed
        # an error (and potentially secrets) back into the model and audit path.
        if getattr(result, "isError", False):
            logger.warning("[mcp_client] call_tool failed: error_type=remote_error_result")
            raise MCPToolExecutionError()

        # 拼接 TextContent 片段
        parts: list[str] = []
        for piece in result.content:
            text = getattr(piece, "text", None)
            parts.append(text if text is not None else str(piece))

        return " ".join(parts) if parts else ""

    async def cleanup(self) -> None:
        """释放子进程 + session 资源(幂等,best-effort)"""
        for name, tool in list(self._registered_tools.items()):
            mcp_registry.unregister(name, expected_tool=tool)
        self._registered_tools.clear()
        try:
            await self._exit_stack.aclose()
        except Exception as exc:
            logger.debug(
                "[mcp_client] cleanup error suppressed: error_type=%s",
                type(exc).__name__,
            )
        self._initialized = False
        self._session = None


async def register_mcp_tools_to_registry(
    client: MCPClient,
    *,
    timeout_sec: float = 30.0,
    max_retries: int = 0,
) -> list[str]:
    """把一个 MCP server 暴露的所有工具批量注册到全局 ToolRegistry。

    Tool name 规则:`mcp_{server_name}_{tool_name}`,避免与原生工具撞名。
    MCP annotations 只是非可信提示，无法证明远端副作用可安全重放；因此默认
    effect_mode=unknown 且不自动重试。只有部署者在本地配置中明确列出的工具
    才标为 read_only，远端 server 不能自行提升权限。

    Returns:
        注册成功的 tool name 列表
    """
    mcp_tools = await client.list_tools()
    server_name = client.config.server_name
    registered: list[str] = []

    # Validate every remote name before registering the first tool. Apart from
    # matching the downstream function-calling contract, this prevents a
    # remote server from injecting control characters into registry logs.
    validated_tools: list[tuple[object, str, str]] = []
    for mt in mcp_tools:
        remote_name = getattr(mt, "name", None)
        full_name = f"mcp_{server_name}_{remote_name}"
        if (
            not isinstance(remote_name, str)
            or not _FUNCTION_NAME_PATTERN.fullmatch(remote_name)
            or not _FUNCTION_NAME_PATTERN.fullmatch(full_name)
        ):
            raise ValueError("invalid MCP tool definition")
        validated_tools.append((mt, remote_name, full_name))

    for mt, remote_name, full_name in validated_tools:
        # 闭包陷阱:用 default arg 固定每轮的 mt
        async def _handler(_tool_name=remote_name, _client=client, **kwargs):
            return await _client.call_tool(_tool_name, kwargs)

        # 复用 MCP 工具的 inputSchema 作为 OpenAI Function Calling parameters
        params_schema = getattr(mt, "inputSchema", None) or {"type": "object", "properties": {}}

        effect_mode = (
            EffectMode.READ_ONLY
            if remote_name in client.config.read_only_tools
            else EffectMode.UNKNOWN
        )
        tool = Tool(
            name=full_name,
            description=getattr(mt, "description", "") or f"MCP tool {mt.name} from {server_name}",
            parameters_schema=params_schema,
            handler=_handler,
            metadata=ToolMetadata(
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                effect_mode=effect_mode,
            ),
        )
        client._registered_tools[full_name] = tool
        mcp_registry.register(tool)
        registered.append(full_name)

    logger.info(
        "[mcp_client] registered tools: server=%s count=%d",
        server_name,
        len(registered),
    )
    return registered
