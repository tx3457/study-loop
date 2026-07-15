"""
Live MCP server 接入（灰度）——把「MCP 只做了 echo 测试、没接真实 server」补成
「接入真实 MCP live server，跑通端到端工具调用」。

接入 duckduckgo-mcp-server（uvx 启动，stdio transport，无需 API key）：
  - search        ：DuckDuckGo 联网搜索
  - fetch_content ：抓取指定 URL 的正文
经 services.mcp_client.register_mcp_tools_to_registry 批量注册进全局 ToolRegistry，
工具名形如 mcp_ddg_search / mcp_ddg_fetch_content。注册后 assistant 自由问答 worker
自动可见（get_tool_definitions() 与白名单都从 registry 派生），无需改 assistant 调用逻辑。

灰度：MCP_LIVE_ENABLED=true 才连（默认 false）；连不上 fail-soft 不阻断启动（降级回无联网）。
守底线：联网工具只供 assistant 自由问答用；出题（quiz_agent）走 search_document 检索本地
       文档库，不碰这些联网工具 → 出题证据仍只来自已建库材料，Faithfulness 卖点不受污染。

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
    return [
        StdioMCPServerConfig(
            server_name="ddg",
            command=uvx,
            args=["duckduckgo-mcp-server"],
        ),
    ]


async def connect_and_register_all() -> list[str]:
    """连接所有 live MCP server 并把其工具注册进 ToolRegistry。返回注册成功的工具名。

    fail-soft：单个 server 连不上只 warn 并降级（清理该 client），不阻断启动 / 其他 server。
    """
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
            logger.info(f"[mcp_servers] live server '{cfg.server_name}' 接入 → {names}")
        except Exception as e:
            logger.warning(f"[mcp_servers] 连接 '{cfg.server_name}' 失败（降级，无联网）: {e}")
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
            logger.debug(f"[mcp_servers] cleanup 忽略: {e}")
    _clients.clear()
