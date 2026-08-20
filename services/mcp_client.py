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
from contextlib import AsyncExitStack
from typing import Optional

from pydantic import BaseModel, Field

from services.tool_registry import EffectMode, Tool, ToolMetadata, tool_registry

logger = logging.getLogger(__name__)


class StdioMCPServerConfig(BaseModel):
    """MCP server 启动配置(stdio transport)"""
    server_name: str = Field(..., description="server 唯一名,用作 tool name 前缀")
    command: str = Field(..., description="启动命令,如 'uvx' / 'npx' / 'python'")
    args: list[str] = Field(default_factory=list, description="启动参数")
    env: Optional[dict[str, str]] = Field(None, description="环境变量")


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
        except Exception as e:
            logger.warning(f"[mcp_client] call_tool '{tool_name}' failed: {type(e).__name__}: {e}")
            raise

        # 拼接 TextContent 片段
        parts: list[str] = []
        for piece in result.content:
            text = getattr(piece, "text", None)
            parts.append(text if text is not None else str(piece))

        return " ".join(parts) if parts else ""

    async def cleanup(self) -> None:
        """释放子进程 + session 资源(幂等,best-effort)"""
        for name, tool in list(self._registered_tools.items()):
            tool_registry.unregister(name, expected_tool=tool)
        self._registered_tools.clear()
        try:
            await self._exit_stack.aclose()
        except Exception as e:
            logger.debug(f"[mcp_client] cleanup error suppressed: {type(e).__name__}: {e}")
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
    effect_mode=unknown 且不自动重试。

    Returns:
        注册成功的 tool name 列表
    """
    mcp_tools = await client.list_tools()
    server_name = client.config.server_name
    registered: list[str] = []

    for mt in mcp_tools:
        # 闭包陷阱:用 default arg 固定每轮的 mt
        async def _handler(_tool_name=mt.name, _client=client, **kwargs):
            return await _client.call_tool(_tool_name, kwargs)

        # 复用 MCP 工具的 inputSchema 作为 OpenAI Function Calling parameters
        params_schema = getattr(mt, "inputSchema", None) or {"type": "object", "properties": {}}

        full_name = f"mcp_{server_name}_{mt.name}"
        tool = Tool(
            name=full_name,
            description=getattr(mt, "description", "") or f"MCP tool {mt.name} from {server_name}",
            parameters_schema=params_schema,
            handler=_handler,
            metadata=ToolMetadata(
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                effect_mode=EffectMode.UNKNOWN,
            ),
        )
        client._registered_tools[full_name] = tool
        tool_registry.register(tool)
        registered.append(full_name)

    logger.info(f"[mcp_client] registered {len(registered)} MCP tools from '{server_name}'")
    return registered
