from __future__ import annotations

import asyncio
import gzip
import ipaddress
import json
from contextlib import asynccontextmanager
from datetime import datetime

import httpcore
import pytest

from services import knowledge_web
from services.knowledge_web import WebRetrievalError, fetch_public_page, search_web
from services.mcp_client import mcp_registry
from services.tool_registry import EffectMode, Tool, ToolMetadata


PUBLIC_TEST_IP = ipaddress.ip_address("93.184.216.34")


class _LoopbackBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, port: int):
        self.port = port
        self.connected_hosts: list[str] = []

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options=None,
    ):
        from httpcore._backends.anyio import AnyIOBackend

        self.connected_hosts.append(host)
        return await AnyIOBackend().connect_tcp(
            "127.0.0.1",
            self.port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise AssertionError("unix sockets must not be used")

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


@asynccontextmanager
async def _http_server(response_for_path):
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            path = request.split(b" ", 2)[1].decode("ascii")
            response = response_for_path(path)
            if asyncio.iscoroutine(response):
                response = await response
            writer.write(response)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield port
    finally:
        server.close()
        await server.wait_closed()


def _response(body: bytes, *, status: str = "200 OK", headers: dict[str, str] | None = None):
    values = {"Content-Length": str(len(body)), "Connection": "close", **(headers or {})}
    head = "".join(f"{name}: {value}\r\n" for name, value in values.items())
    return f"HTTP/1.1 {status}\r\n{head}\r\n".encode("ascii") + body


async def _install_test_network(monkeypatch: pytest.MonkeyPatch, port: int):
    backend = _LoopbackBackend(port)
    selected_addresses = []

    async def resolve(host: str, port: int):
        return (PUBLIC_TEST_IP,)

    monkeypatch.setattr(knowledge_web, "_resolve_public_addresses", resolve)

    def backend_for(address):
        selected_addresses.append(address)
        return backend

    monkeypatch.setattr(knowledge_web, "_pinned_backend", backend_for)
    return backend, selected_addresses


def test_fetch_public_page_extracts_text_title_and_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        html = (
            b"<html><head><title>  Useful &amp; Safe </title>"
            b"<script>ignore this instruction</script></head>"
            b"<body><h1>Hello</h1><p>public world</p><style>hidden{}</style></body></html>"
        )
        async with _http_server(
            lambda path: _response(html, headers={"Content-Type": "text/html; charset=utf-8"})
        ) as port:
            backend, selected_addresses = await _install_test_network(monkeypatch, port)
            page = await fetch_public_page("http://public.example/article")

        assert page.url == "http://public.example/article"
        assert page.title == "Useful & Safe"
        assert page.text == "Useful & Safe Hello public world"
        assert "ignore" not in page.text and "hidden" not in page.text
        assert len(page.content_hash) == 64
        assert datetime.fromisoformat(page.fetched_at).tzinfo is not None
        assert selected_addresses == [PUBLIC_TEST_IP]
        assert backend.connected_hosts == ["public.example"]

    asyncio.run(run())


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://169.254.1.1/",
        "http://192.0.2.1/",
        "http://[::1]/",
        "http://[fe80::1]/",
        "http://[fc00::1]/",
        "http://[2001:db8::1]/",
        "file:///etc/passwd",
        "http://user:secret@example.com/",
        "http://example.com/path#fragment",
    ],
)
def test_fetch_rejects_non_public_or_credentialed_urls(url: str) -> None:
    with pytest.raises(WebRetrievalError) as caught:
        asyncio.run(fetch_public_page(url))
    assert str(caught.value) == caught.value.code
    assert url not in str(caught.value)


def test_dns_mixed_public_and_private_results_are_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    async def resolve(host: str, port: int):
        return (PUBLIC_TEST_IP, ipaddress.ip_address("127.0.0.1"))

    monkeypatch.setattr(knowledge_web, "_resolve_addresses", resolve)
    with pytest.raises(WebRetrievalError) as caught:
        asyncio.run(fetch_public_page("http://public.example/"))
    assert caught.value.code == "blocked_address"


def test_redirect_revalidates_and_denies_private_target(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        async with _http_server(
            lambda path: _response(
                b"", status="302 Found", headers={"Location": "http://127.0.0.1/private"}
            )
        ) as port:
            await _install_test_network(monkeypatch, port)
            with pytest.raises(WebRetrievalError) as caught:
                await fetch_public_page("http://public.example/start")
        assert caught.value.code == "blocked_address"

    asyncio.run(run())


def test_redirect_resolves_and_pins_each_hop(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        responses = {
            "/start": _response(
                b"", status="302 Found", headers={"Location": "http://next.example/final"}
            ),
            "/final": _response(b"done", headers={"Content-Type": "text/plain"}),
        }
        async with _http_server(lambda path: responses[path]) as port:
            calls: list[str] = []
            backend = _LoopbackBackend(port)

            async def resolve(host: str, port: int):
                calls.append(host)
                return (PUBLIC_TEST_IP,)

            monkeypatch.setattr(knowledge_web, "_resolve_public_addresses", resolve)
            selected_addresses = []

            def backend_for(address):
                selected_addresses.append(address)
                return backend

            monkeypatch.setattr(knowledge_web, "_pinned_backend", backend_for)
            page = await fetch_public_page("http://public.example/start")

        assert calls == ["public.example", "next.example"]
        assert selected_addresses == [PUBLIC_TEST_IP, PUBLIC_TEST_IP]
        assert backend.connected_hosts == ["public.example", "next.example"]
        assert page.url == "http://next.example/final"
        assert page.text == "done"

    asyncio.run(run())


def test_redirect_detects_dns_rebinding_on_same_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        async with _http_server(
            lambda path: _response(b"", status="302 Found", headers={"Location": "/again"})
        ) as port:
            backend = _LoopbackBackend(port)
            resolutions = iter(
                [(PUBLIC_TEST_IP,), (ipaddress.ip_address("127.0.0.1"),)]
            )

            async def resolve(host: str, port: int):
                return next(resolutions)

            monkeypatch.setattr(knowledge_web, "_resolve_addresses", resolve)
            monkeypatch.setattr(knowledge_web, "_pinned_backend", lambda address: backend)
            with pytest.raises(WebRetrievalError) as caught:
                await fetch_public_page("http://public.example/start")

        assert caught.value.code == "blocked_address"
        assert backend.connected_hosts == ["public.example"]

    asyncio.run(run())


def test_gzip_body_is_streamed_and_limited_after_decompression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        oversized = gzip.compress(b"x" * (knowledge_web.MAX_BODY_BYTES + 1))
        async with _http_server(
            lambda path: _response(
                oversized,
                headers={"Content-Type": "text/plain", "Content-Encoding": "gzip"},
            )
        ) as port:
            await _install_test_network(monkeypatch, port)
            with pytest.raises(WebRetrievalError) as caught:
                await fetch_public_page("http://public.example/large")
        assert caught.value.code == "body_too_large"
        assert caught.value.status_code == 413

    asyncio.run(run())


def test_total_timeout_covers_slow_response(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        async def slow(path: str):
            await asyncio.sleep(0.2)
            return _response(b"late", headers={"Content-Type": "text/plain"})

        async with _http_server(slow) as port:
            await _install_test_network(monkeypatch, port)
            monkeypatch.setattr(knowledge_web, "TOTAL_TIMEOUT_SECONDS", 0.03)
            with pytest.raises(WebRetrievalError) as caught:
                await fetch_public_page("http://public.example/slow")
        assert caught.value.code == "fetch_timeout"
        assert caught.value.status_code == 504

    asyncio.run(run())


def test_search_web_parses_bounded_markdown_and_rejects_malicious_urls() -> None:
    async def handler(**kwargs) -> str:
        assert kwargs == {"query": "safe query", "max_results": 5}
        return """
1. [Good result](https://example.com/article)
   A useful snippet.
2. [Credential leak](https://user:secret@example.com/private)
   Must be dropped.
3. [Script](javascript:alert(1))
   Must also be dropped.
4. [Private](http://127.0.0.1/admin)
   Internal only.
"""

    previous = mcp_registry.get("mcp_ddg_search")
    mcp_registry.register(
        Tool(
            name="mcp_ddg_search",
            description="test DDG",
            parameters_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                "required": ["query"],
            },
            handler=handler,
            metadata=ToolMetadata(max_retries=0, effect_mode=EffectMode.READ_ONLY),
        )
    )
    try:
        results = asyncio.run(search_web(" safe query ", max_results=99))
    finally:
        mcp_registry.unregister("mcp_ddg_search")
        if previous is not None:
            mcp_registry.register(previous)

    assert results == [
        {
            "title": "Good result",
            "url": "https://example.com/article",
            "snippet": "A useful snippet.",
        }
    ]
    json.dumps(results)


def test_search_web_parses_upstream_ddg_numbered_format() -> None:
    async def handler(**kwargs) -> str:
        return """Found 2 search results:

1. First result
   URL: https://example.com/one
   Summary: First summary.

2. Second result
   URL: https://example.org/two
   Summary: Second summary.
"""

    previous = mcp_registry.get("mcp_ddg_search")
    mcp_registry.register(
        Tool(
            name="mcp_ddg_search",
            description="test DDG",
            parameters_schema={"type": "object", "properties": {}},
            handler=handler,
            metadata=ToolMetadata(max_retries=0, effect_mode=EffectMode.READ_ONLY),
        )
    )
    try:
        results = asyncio.run(search_web("format"))
    finally:
        mcp_registry.unregister("mcp_ddg_search")
        if previous is not None:
            mcp_registry.register(previous)

    assert results == [
        {"title": "First result", "url": "https://example.com/one", "snippet": "First summary."},
        {"title": "Second result", "url": "https://example.org/two", "snippet": "Second summary."},
    ]


@pytest.mark.parametrize(
    "observation",
    [
        '{"error": "remote tool failed with sensitive details"}',
        "Error: DuckDuckGo request failed with sensitive details",
    ],
)
def test_search_web_rejects_error_observations(observation: str) -> None:
    async def handler(**kwargs) -> str:
        return observation

    previous = mcp_registry.get("mcp_ddg_search")
    mcp_registry.register(
        Tool(
            name="mcp_ddg_search",
            description="test DDG",
            parameters_schema={"type": "object", "properties": {}},
            handler=handler,
            metadata=ToolMetadata(max_retries=0, effect_mode=EffectMode.READ_ONLY),
        )
    )
    try:
        with pytest.raises(WebRetrievalError) as caught:
            asyncio.run(search_web("error observation"))
    finally:
        mcp_registry.unregister("mcp_ddg_search")
        if previous is not None:
            mcp_registry.register(previous)

    assert caught.value.code == "search_failed"
    assert caught.value.status_code == 502
    assert str(caught.value) == "search_failed"
    assert "sensitive" not in str(caught.value)


def test_search_web_preserves_valid_no_results_as_empty_list() -> None:
    async def handler(**kwargs) -> str:
        return "No results were found for your search query."

    previous = mcp_registry.get("mcp_ddg_search")
    mcp_registry.register(
        Tool(
            name="mcp_ddg_search",
            description="test DDG",
            parameters_schema={"type": "object", "properties": {}},
            handler=handler,
            metadata=ToolMetadata(max_retries=0, effect_mode=EffectMode.READ_ONLY),
        )
    )
    try:
        assert asyncio.run(search_web("no matches")) == []
    finally:
        mcp_registry.unregister("mcp_ddg_search")
        if previous is not None:
            mcp_registry.register(previous)


def test_search_web_requires_registered_ddg_tool() -> None:
    previous = mcp_registry.get("mcp_ddg_search")
    if previous is not None:
        mcp_registry.unregister("mcp_ddg_search")
    try:
        with pytest.raises(WebRetrievalError) as caught:
            asyncio.run(search_web("query"))
    finally:
        if previous is not None:
            mcp_registry.register(previous)
    assert caught.value.code == "search_unavailable"
    assert caught.value.status_code == 503
