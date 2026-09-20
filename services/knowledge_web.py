"""Bounded, SSRF-resistant public web retrieval for knowledge-base evidence.

Search results and fetched pages are untrusted data.  This module never exposes
the raw MCP fetch tool and never follows a URL merely because it appeared in a
model or search result.
"""

from __future__ import annotations

import asyncio
import codecs
import hashlib
import ipaddress
import json
import re
import socket
import ssl
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import urljoin, urlsplit

import httpcore

from services.tool_registry import tool_registry


MAX_BODY_BYTES = 2 * 1024 * 1024
_MAX_TRANSFER_BYTES = MAX_BODY_BYTES + 64 * 1024
TOTAL_TIMEOUT_SECONDS = 10.0
MAX_REDIRECTS = 3
MAX_URL_LENGTH = 2048
MAX_QUERY_LENGTH = 500
MAX_SEARCH_RESPONSE_BYTES = 256 * 1024
MAX_TITLE_LENGTH = 300
MAX_SNIPPET_LENGTH = 1000
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_TEXT_CONTENT_TYPES = frozenset({"text/html", "text/plain", "application/xhtml+xml"})
_HTML_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_SKIPPED_HTML_ELEMENTS = frozenset({"script", "style", "noscript", "template"})
_MARKDOWN_RESULT = re.compile(
    r"^\s*(?:[-*]|\d+[.)])\s*\[([^\]]{1,300})\]\(([^)\s]+)\)\s*(.*)$"
)
_PLAIN_RESULT_TITLE = re.compile(r"^\s*\d+[.)]\s+(.{1,300})\s*$")
_LABELLED_FIELD = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?"
    r"(title|url|link|snippet|summary|description)(?:\*\*)?\s*:\s*(.*)$",
    re.IGNORECASE,
)
_CHARSET = re.compile(r"(?:^|;)\s*charset\s*=\s*[\"']?([^;\s\"']+)", re.IGNORECASE)


@dataclass(frozen=True)
class WebPage:
    url: str
    title: str
    text: str
    fetched_at: str
    content_hash: str


class WebRetrievalError(RuntimeError):
    """Public-safe web failure whose text never includes remote data or URLs."""

    def __init__(self, code: str, status_code: int):
        self.code = code
        self.status_code = status_code
        super().__init__(code)


@dataclass(frozen=True)
class _Target:
    url: str
    host: str
    port: int


class _PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    """Connect to one validated IP while httpcore retains the URL host for TLS SNI."""

    def __init__(self, address: ipaddress.IPv4Address | ipaddress.IPv6Address):
        # httpcore exposes the backend protocol but not a public concrete default.
        # Keeping this private import in one adapter makes that dependency explicit.
        from httpcore._backends.auto import AutoBackend

        self._address = str(address)
        self._backend = AutoBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options=None,
    ) -> httpcore.AsyncNetworkStream:
        return await self._backend.connect_tcp(
            self._address,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise OSError("unix_socket_disabled")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


class _PlainTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._parts: list[str] = []
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        normalized = tag.lower()
        if normalized in _SKIPPED_HTML_ELEMENTS:
            self._skip_depth += 1
        elif normalized == "title" and self._skip_depth == 0:
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in _SKIPPED_HTML_ELEMENTS and self._skip_depth:
            self._skip_depth -= 1
        elif normalized == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not data.strip():
            return
        self._parts.append(data)
        if self._in_title:
            self._title_parts.append(data)

    @property
    def title(self) -> str:
        return _collapse_text(" ".join(self._title_parts))[:MAX_TITLE_LENGTH]

    @property
    def text(self) -> str:
        return _collapse_text(" ".join(self._parts))


def _error(code: str, status_code: int) -> WebRetrievalError:
    return WebRetrievalError(code, status_code)


def _collapse_text(value: str) -> str:
    return " ".join(value.split())


def _is_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(address.is_global)


def _validate_url(url: str) -> _Target:
    if not isinstance(url, str) or not url or len(url) > MAX_URL_LENGTH:
        raise _error("invalid_url", 400)
    if any(ord(char) < 32 or ord(char) == 127 for char in url):
        raise _error("invalid_url", 400)
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        raise _error("invalid_url", 400) from None
    if parsed.scheme.lower() not in {"http", "https"}:
        raise _error("invalid_url", 400)
    if (
        not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise _error("invalid_url", 400)
    host = parsed.hostname.rstrip(".").lower()
    if not host or "%" in host or host == "localhost" or host.endswith(".localhost"):
        raise _error("blocked_address", 403)
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not _is_public_address(address):
        raise _error("blocked_address", 403)
    normalized_port = port or (443 if parsed.scheme.lower() == "https" else 80)
    return _Target(url=url, host=host, port=normalized_port)


async def _resolve_addresses(
    host: str,
    port: int,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        return (literal,)
    loop = asyncio.get_running_loop()
    records = await loop.getaddrinfo(
        host,
        port,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for _family, _type, _proto, _canonname, sockaddr in records:
        address = ipaddress.ip_address(sockaddr[0])
        if address not in addresses:
            addresses.append(address)
    return tuple(addresses)


async def _resolve_public_addresses(
    host: str,
    port: int,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    try:
        addresses = await _resolve_addresses(host, port)
    except (OSError, UnicodeError, ValueError):
        raise _error("dns_failed", 502) from None
    if not addresses:
        raise _error("dns_failed", 502)
    if any(not _is_public_address(address) for address in addresses):
        raise _error("blocked_address", 403)
    return addresses


def _pinned_backend(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> httpcore.AsyncNetworkBackend:
    return _PinnedNetworkBackend(address)


def _headers_map(headers: Iterable[tuple[bytes, bytes]]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for raw_name, raw_value in headers:
        name = raw_name.decode("ascii", "ignore").lower()
        value = raw_value.decode("latin-1").strip()
        if name in mapped:
            mapped[name] = f"{mapped[name]}, {value}"
        else:
            mapped[name] = value
    return mapped


def _decompressor(encoding: str):
    if encoding in {"", "identity"}:
        return None
    if encoding == "gzip":
        return zlib.decompressobj(16 + zlib.MAX_WBITS)
    if encoding == "deflate":
        return zlib.decompressobj()
    raise _error("unsupported_content_encoding", 502)


async def _read_bounded_body(response: httpcore.Response, headers: dict[str, str]) -> bytes:
    encoding = headers.get("content-encoding", "").lower().strip()
    decompressor = _decompressor(encoding)
    output = bytearray()
    raw_size = 0
    try:
        async for chunk in response.aiter_stream():
            raw_size += len(chunk)
            if raw_size > _MAX_TRANSFER_BYTES:
                raise _error("body_too_large", 413)
            remaining = MAX_BODY_BYTES - len(output)
            decoded = chunk if decompressor is None else decompressor.decompress(chunk, remaining + 1)
            output.extend(decoded)
            if len(output) > MAX_BODY_BYTES or (
                decompressor is not None and decompressor.unconsumed_tail
            ):
                raise _error("body_too_large", 413)
        if decompressor is not None:
            output.extend(decompressor.flush(MAX_BODY_BYTES - len(output) + 1))
            if not decompressor.eof:
                raise _error("invalid_content_encoding", 502)
    except zlib.error:
        raise _error("invalid_content_encoding", 502) from None
    if len(output) > MAX_BODY_BYTES:
        raise _error("body_too_large", 413)
    return bytes(output)


def _decode_body(body: bytes, content_type: str) -> str:
    charset_match = _CHARSET.search(content_type)
    charset = charset_match.group(1) if charset_match else "utf-8"
    try:
        codecs.lookup(charset)
    except LookupError:
        charset = "utf-8"
    return body.decode(charset, errors="replace")


async def _fetch_hop(target: _Target) -> tuple[int, dict[str, str], bytes]:
    addresses = await _resolve_public_addresses(target.host, target.port)
    backend = _pinned_backend(addresses[0])
    ssl_context = ssl.create_default_context()
    pool = httpcore.AsyncConnectionPool(
        ssl_context=ssl_context,
        network_backend=backend,
        max_connections=1,
        max_keepalive_connections=0,
        retries=0,
    )
    request_headers = [
        (b"Accept", b"text/html, application/xhtml+xml, text/plain;q=0.9"),
        (b"Accept-Encoding", b"gzip, deflate"),
        (b"User-Agent", b"StudyLoop-KnowledgeWeb/1.0"),
    ]
    try:
        async with pool.stream("GET", target.url, headers=request_headers) as response:
            headers = _headers_map(response.headers)
            if response.status in _REDIRECT_STATUSES:
                return response.status, headers, b""
            body = await _read_bounded_body(response, headers)
            return response.status, headers, body
    finally:
        await pool.aclose()


async def _fetch_public_page(url: str) -> WebPage:
    current_url = url
    for redirect_count in range(MAX_REDIRECTS + 1):
        target = _validate_url(current_url)
        status, headers, body = await _fetch_hop(target)
        if status in _REDIRECT_STATUSES:
            location = headers.get("location")
            if not location or redirect_count >= MAX_REDIRECTS:
                raise _error("redirect_rejected", 502)
            current_url = urljoin(current_url, location)
            continue
        if not 200 <= status < 300:
            raise _error("upstream_http_error", 502)
        media_type = headers.get("content-type", "text/plain").split(";", 1)[0].strip().lower()
        if media_type not in _TEXT_CONTENT_TYPES:
            raise _error("unsupported_content_type", 415)
        decoded = _decode_body(body, headers.get("content-type", ""))
        if media_type in _HTML_CONTENT_TYPES:
            parser = _PlainTextExtractor()
            parser.feed(decoded)
            parser.close()
            text = parser.text
            title = parser.title
        else:
            text = _collapse_text(decoded)
            title = ""
        if not text:
            raise _error("empty_content", 422)
        if not title:
            title = target.host[:MAX_TITLE_LENGTH]
        return WebPage(
            url=current_url,
            title=title,
            text=text,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
    raise _error("redirect_rejected", 502)


async def fetch_public_page(url: str) -> WebPage:
    """Fetch one public HTTP(S) page with a ten-second whole-operation deadline."""

    try:
        async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
            return await _fetch_public_page(url)
    except WebRetrievalError:
        raise
    except TimeoutError:
        raise _error("fetch_timeout", 504) from None
    except asyncio.CancelledError:
        raise
    except Exception:
        raise _error("fetch_failed", 502) from None


def _safe_result_url(value: object) -> str | None:
    if not isinstance(value, str) or not value or len(value) > MAX_URL_LENGTH:
        return None
    try:
        target = _validate_url(value.strip())
    except WebRetrievalError:
        return None
    return target.url


def _bounded_result(title: object, url: object, snippet: object) -> dict | None:
    safe_url = _safe_result_url(url)
    if safe_url is None:
        return None
    safe_title = _collapse_text(title if isinstance(title, str) else "")[:MAX_TITLE_LENGTH]
    safe_snippet = _collapse_text(snippet if isinstance(snippet, str) else "")[:MAX_SNIPPET_LENGTH]
    if not safe_title:
        safe_title = urlsplit(safe_url).hostname or "result"
    return {"title": safe_title, "url": safe_url, "snippet": safe_snippet}


def _parse_json_results(raw: str) -> list[dict] | None:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(payload, dict):
        rows = payload.get("results")
    else:
        rows = payload
    if not isinstance(rows, list):
        return None
    parsed: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        result = _bounded_result(
            row.get("title", row.get("name", "")),
            row.get("url", row.get("link", "")),
            row.get("snippet", row.get("description", row.get("body", ""))),
        )
        if result is not None:
            parsed.append(result)
    return parsed


def _parse_markdown_results(raw: str) -> list[dict]:
    results: list[dict] = []
    current: dict[str, str] | None = None
    labelled: dict[str, str] = {}

    def flush_labelled() -> None:
        nonlocal labelled
        if labelled:
            result = _bounded_result(
                labelled.get("title", ""),
                labelled.get("url", labelled.get("link", "")),
                labelled.get("snippet", labelled.get("summary", labelled.get("description", ""))),
            )
            if result is not None:
                results.append(result)
        labelled = {}

    for line in raw.splitlines():
        match = _MARKDOWN_RESULT.match(line)
        if match:
            flush_labelled()
            if current is not None:
                result = _bounded_result(current["title"], current["url"], current["snippet"])
                if result is not None:
                    results.append(result)
            current = {"title": match.group(1), "url": match.group(2), "snippet": match.group(3)}
            continue
        field = _LABELLED_FIELD.match(line)
        plain_title = _PLAIN_RESULT_TITLE.match(line)
        if plain_title:
            flush_labelled()
            if current is not None:
                result = _bounded_result(current["title"], current["url"], current["snippet"])
                if result is not None:
                    results.append(result)
            current = {"title": plain_title.group(1), "url": "", "snippet": ""}
            continue
        if field and current is not None:
            key = field.group(1).lower()
            value = field.group(2).strip()
            if key in {"url", "link"}:
                current["url"] = value
            elif key in {"snippet", "summary", "description"}:
                current["snippet"] = value
            elif key == "title":
                current["title"] = value
            continue
        if field and current is None:
            key = field.group(1).lower()
            if key == "title" and labelled.get("url"):
                flush_labelled()
            labelled[key] = field.group(2).strip()
            continue
        text = line.strip()
        if current is not None and text:
            current["snippet"] = f"{current['snippet']} {text}".strip()
        elif not text:
            flush_labelled()
    if current is not None:
        result = _bounded_result(current["title"], current["url"], current["snippet"])
        if result is not None:
            results.append(result)
    flush_labelled()
    return results


def _is_search_error_observation(raw: str) -> bool:
    stripped = raw.lstrip()
    if stripped.casefold().startswith("error:"):
        return True
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return False
    return (
        isinstance(payload, dict)
        and "error" in payload
        and payload["error"] not in (None, "", False)
    )


def _parse_search_results(raw: str, limit: int) -> list[dict]:
    if len(raw.encode("utf-8", "replace")) > MAX_SEARCH_RESPONSE_BYTES:
        raise _error("search_response_too_large", 502)
    if _is_search_error_observation(raw):
        raise _error("search_failed", 502)
    parsed = _parse_json_results(raw)
    if parsed is None:
        parsed = _parse_markdown_results(raw)
    unique: list[dict] = []
    seen: set[str] = set()
    for result in parsed:
        if result["url"] in seen:
            continue
        seen.add(result["url"])
        unique.append(result)
        if len(unique) >= limit:
            break
    return unique


async def search_web(query: str, *, max_results: int = 5) -> list[dict]:
    """Search through the registered DDG MCP adapter and return typed, bounded rows."""

    if not isinstance(query, str):
        raise _error("invalid_query", 400)
    normalized_query = _collapse_text(query)
    if not normalized_query or len(normalized_query) > MAX_QUERY_LENGTH:
        raise _error("invalid_query", 400)
    if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results <= 0:
        raise _error("invalid_result_limit", 400)
    limit = min(max_results, 5)
    if tool_registry.get("mcp_ddg_search") is None:
        raise _error("search_unavailable", 503)
    try:
        raw = await tool_registry.invoke(
            "mcp_ddg_search",
            {"query": normalized_query, "max_results": limit},
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        raise _error("search_failed", 502) from None
    if not isinstance(raw, str):
        raise _error("search_failed", 502)
    return _parse_search_results(raw, limit)


__all__ = ["WebPage", "WebRetrievalError", "fetch_public_page", "search_web"]
