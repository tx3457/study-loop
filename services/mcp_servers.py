"""
Live MCP server 接入（灰度）

接入 duckduckgo-mcp-server（uvx 启动，stdio transport，无需 API key）：
  - search        ：DuckDuckGo 联网搜索
  - fetch_content ：抓取指定 URL 的正文
经 services.mcp_client.register_mcp_tools_to_registry 注册进独立的 mcp_registry
（不是全局 ToolRegistry），工具名形如 mcp_ddg_search / mcp_ddg_fetch_content。
模型永远看不到这些工具：它们不进 get_tool_definitions / allowed_tool_names，
只由持有该能力安全契约的应用适配器调用——目前只有 services.knowledge_web.search_web
消费 mcp_ddg_search，并在自己这一层做 query 约束与结果边界。
原始 fetch 工具没有消费者：公网抓取一律走 knowledge_web.fetch_public_page，
它做 SSRF 校验、逐跳重定向校验、IP 钉定与正文大小限制，这些是远端 MCP server 不做的。

灰度：MCP_LIVE_ENABLED=true 才连（默认 false）；连不上 fail-soft 不阻断启动（降级回无联网）。
守底线：出题（quiz_agent）走 search_document 检索本地文档库，不碰这些联网工具；
       出题证据仍只来自已建库材料，保持 faithfulness 边界。

依赖：本机需 uv（uvx）。首次连接时 uvx 自动从 PyPI 拉 duckduckgo-mcp-server。
     UVX_PATH 可指定 uvx 绝对路径（systemd/docker 下 PATH 可能不含 ~/.local/bin）。
"""
import logging
import os

from services.mcp_client import (
    MCPClient,
    StdioMCPServerConfig,
    register_mcp_tools_to_registry,
)

logger = logging.getLogger(__name__)

# 已连接的 client（startup 连接、shutdown 释放子进程）
_clients: list[MCPClient] = []


def mcp_live_enabled() -> bool:
    """MCP_LIVE_ENABLED=true → 启动时连接真实 MCP live server（默认 false，灰度）。"""
    return os.getenv("MCP_LIVE_ENABLED", "false").lower() in ("1", "true", "yes")


def _live_server_configs() -> list[StdioMCPServerConfig]:
    """live MCP server 清单。默认接 duckduckgo-mcp-server（无 key，search + fetch_content）。"""
    uvx = os.getenv("UVX_PATH", "uvx")
    bundled = os.getenv("MCP_DDG_COMMAND", "").strip()
    arguments = ["--ref-url-threshold", "0"]
    if not bundled:
        arguments = ["--from", "duckduckgo-mcp-server==0.7.0", "duckduckgo-mcp-server", *arguments]
    return [
        StdioMCPServerConfig(
            server_name="ddg",
            command=bundled or uvx,
            args=arguments,
            read_only_tools=frozenset({"search", "fetch_content"}),
        ),
    ]


async def connect_and_register_all() -> list[str]:
    """连接所有 live MCP server 并把其工具注册进 ToolRegistry。返回注册成功的工具名。

    fail-soft：单个 server 连不上只 warn 并降级（清理该 client），不阻断启动 / 其他 server。
    """
    if _clients:
        logger.warning("[mcp_servers] replacing existing live MCP connections")
        await cleanup_all()

    if not mcp_live_enabled():
        logger.info("[mcp_servers] MCP_LIVE_ENABLED=false，跳过 live MCP 接入")
        return []

    registered: list[str] = []
    for cfg in _live_server_configs():
        client = MCPClient(cfg)
        try:
            await client.connect()
            names = await register_mcp_tools_to_registry(client)
            _clients.append(client)
            registered.extend(names)
            logger.info(
                "[mcp_servers] live server connected: server=%s tool_count=%d",
                cfg.server_name,
                len(names),
            )
        except Exception as e:
            logger.warning(
                "[mcp_servers] 连接失败（降级，无联网）: "
                "server=%s error_type=%s",
                cfg.server_name,
                type(e).__name__,
            )
            try:
                await client.cleanup()
            except Exception:
                pass
    return registered


async def cleanup_all() -> None:
    """释放所有 live MCP server 子进程（shutdown 调用，幂等）。"""
    for client in _clients:
        try:
            await client.cleanup()
        except Exception as e:
            logger.debug(
                "[mcp_servers] cleanup 忽略: error_type=%s",
                type(e).__name__,
            )
    _clients.clear()
