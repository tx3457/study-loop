"""Request correlation and a fail-closed HTTP error boundary."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from contextvars import ContextVar, Token

from starlette.datastructures import Headers, MutableHeaders
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_PATTERN = re.compile(r"^req_[0-9a-f]{32}$")
_request_id: ContextVar[str | None] = ContextVar(
    "studyloop_request_id", default=None
)
# The request path, so work done deep inside a request (a model call made by a
# tool three layers down) can be attributed to the feature that caused it
# without threading a label through every call site.
_request_path: ContextVar[str | None] = ContextVar(
    "studyloop_request_path", default=None
)


class _ClientSendDisconnected(Exception):
    """Internal marker for a response transport that is already gone."""


def _new_request_id() -> str:
    return f"req_{uuid.uuid4().hex}"


def normalize_request_id(value: str | None) -> str:
    """Accept only the server's log-safe public request-id format."""
    if value is not None and _REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return _new_request_id()


def current_request_id() -> str:
    """Return the active request id, or a safe placeholder outside HTTP."""
    return _request_id.get() or "request_unavailable"


def current_request_path() -> str | None:
    """Return the active request path, or None for work outside any request."""
    return _request_path.get()


def public_error_payload(
    *,
    error: str,
    detail: str,
    code: str,
    include_request_id: bool = False,
) -> dict[str, str]:
    payload = {"error": error, "detail": detail, "code": code}
    if include_request_id:
        payload["request_id"] = current_request_id()
    return payload


def safe_sse_error(*, detail: str, code: str) -> str:
    """Serialize a public SSE error without exposing the underlying exception."""
    payload = {
        "type": "error",
        "detail": detail,
        "code": code,
        "request_id": current_request_id(),
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class SafeErrorMiddleware:
    """Convert unexpected application exceptions to a fixed public response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False
        response_complete = False

        async def track_response(message: Message) -> None:
            nonlocal response_started, response_complete
            if message["type"] == "http.response.start":
                response_started = True
            elif (
                message["type"] == "http.response.body"
                and not message.get("more_body", False)
            ):
                response_complete = True
            try:
                await send(message)
            except (ClientDisconnect, OSError) as exc:
                raise _ClientSendDisconnected from exc

        try:
            await self.app(scope, receive, track_response)
        except asyncio.CancelledError as cancelled:
            # Preserve cooperative cancellation without chaining an inner
            # handled exception that could contain provider/user data.
            raise cancelled from None
        except (ClientDisconnect, _ClientSendDisconnected):
            # A disconnected client is not an application failure. In
            # particular, do not attempt a second write to the dead transport.
            return
        except Exception as exc:
            logger.error(
                "unhandled request failure request_id=%s method=%s error_type=%s",
                current_request_id(),
                scope.get("method", "UNKNOWN"),
                type(exc).__name__,
            )
            if response_started:
                if not response_complete:
                    try:
                        await track_response(
                            {
                                "type": "http.response.body",
                                "body": b"",
                                "more_body": False,
                            }
                        )
                    except asyncio.CancelledError as cancelled:
                        raise cancelled from None
                    except _ClientSendDisconnected:
                        pass
                return

            response = JSONResponse(
                status_code=500,
                content=public_error_payload(
                    error="服务器内部错误",
                    detail="请求处理失败，请稍后重试",
                    code="internal_error",
                    include_request_id=True,
                ),
            )
            try:
                await response(scope, receive, track_response)
            except asyncio.CancelledError as cancelled:
                raise cancelled from None
            except _ClientSendDisconnected:
                return


class RequestContextMiddleware:
    """Bind one request id until the complete ASGI response has been sent.

    This middleware must wrap CORS and ``SafeErrorMiddleware``. That ordering
    ensures preflight responses also receive an ID while unexpected application
    errors still pass through CORS before reaching the browser.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming_id = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = normalize_request_id(incoming_id)
        scope.setdefault("state", {})["request_id"] = request_id
        token: Token[str | None] = _request_id.set(request_id)
        path_token: Token[str | None] = _request_path.set(scope.get("path"))
        response_status: int | None = None
        response_complete = False

        async def send_with_request_id(message: Message) -> None:
            nonlocal response_status, response_complete
            if message["type"] == "http.response.start":
                status = int(message["status"])
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
                await send(message)
                response_status = status
                return
            await send(message)
            if (
                message["type"] == "http.response.body"
                and not message.get("more_body", False)
            ):
                response_complete = True

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            if response_status is not None:
                route = scope.get("route")
                route_template = getattr(route, "path", "unmatched")
                if response_complete:
                    log_level = (
                        logging.WARNING
                        if response_status >= 400
                        else logging.INFO
                    )
                    logger.log(
                        log_level,
                        "request completed request_id=%s method=%s route=%s status=%s",
                        request_id,
                        scope.get("method", "UNKNOWN"),
                        route_template,
                        response_status,
                    )
                else:
                    logger.warning(
                        "request incomplete request_id=%s method=%s route=%s status=%s",
                        request_id,
                        scope.get("method", "UNKNOWN"),
                        route_template,
                        response_status,
                    )
            _request_path.reset(path_token)
            _request_id.reset(token)
