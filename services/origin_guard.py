"""Reject browser-initiated state changes from untrusted origins."""

from __future__ import annotations

import logging
from collections.abc import Collection

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from services.request_context import current_request_id, public_error_payload


logger = logging.getLogger(__name__)

ALLOWED_BROWSER_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4001",
    "http://127.0.0.1:4001",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
)

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class OriginGuardMiddleware:
    """Require an exact trusted ``Origin`` on browser state changes.

    Requests without an Origin header remain available to non-browser clients.
    Inspecting the raw ASGI headers lets the guard fail closed for repeated
    Origin fields before either the request body or an application route is
    reached.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        allowed_origins: Collection[str] = ALLOWED_BROWSER_ORIGINS,
    ) -> None:
        self.app = app
        self.allowed_origins = frozenset(allowed_origins)

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "UNKNOWN")).upper()
        if method in _SAFE_METHODS:
            await self.app(scope, receive, send)
            return

        origin_values = [
            value for name, value in scope.get("headers", ()) if name.lower() == b"origin"
        ]
        if not origin_values:
            await self.app(scope, receive, send)
            return

        if len(origin_values) == 1 and origin_values[0].decode("latin-1") in self.allowed_origins:
            await self.app(scope, receive, send)
            return

        logger.warning(
            "unsafe request origin rejected request_id=%s method=%s",
            current_request_id(),
            method,
        )
        response = JSONResponse(
            status_code=403,
            content=public_error_payload(
                error="请求被拒绝",
                detail="请求来源不受信任",
                code="origin_not_allowed",
                include_request_id=True,
            ),
        )
        await response(scope, receive, send)
