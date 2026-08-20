"""Public error redaction and request-correlation contracts."""

from __future__ import annotations

import asyncio
import json
import re
import traceback
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

import main
import agents.grader_agent as grader_agent_module
import routers.chat as chat_router
import routers.health as health_router
import routers.stream as stream_router
from services import request_context
from services.retry import RetryExhausted


_REQUEST_ID_RE = re.compile(r"^req_[0-9a-f]{32}$")


class _FailingOrchestrator:
    def __init__(self, secret: str) -> None:
        self._secret = secret

    async def astream(self, *_args, **_kwargs):
        if False:
            yield None
        raise RuntimeError(self._secret)


class _InterleavingFailingOrchestrator:
    async def astream(self, state, *_args, **_kwargs):
        if False:
            yield None
        await asyncio.sleep(0.02 if state["user_id"] == "slow" else 0)
        raise RuntimeError("interleaved-secret")


async def _failing_chat_stream(secret: str):
    yield "data: first\n\n"
    raise RuntimeError(secret)


class TestRequestErrorBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(main.app, raise_server_exceptions=False)

    def test_request_id_is_generated_reused_and_exposed_to_browser(self) -> None:
        generated = self.client.get(
            "/health/live",
            headers={"Origin": "http://127.0.0.1:5173"},
        )
        generated_id = generated.headers["x-request-id"]
        self.assertRegex(generated_id, _REQUEST_ID_RE)
        self.assertEqual(
            generated.headers.get("access-control-expose-headers"),
            "X-Request-ID",
        )

        supplied_id = "req_" + "a" * 32
        supplied = self.client.get(
            "/health/live", headers={"X-Request-ID": supplied_id}
        )
        self.assertEqual(supplied.headers["x-request-id"], supplied_id)

        invalid = self.client.get(
            "/health/live", headers={"X-Request-ID": "attacker-controlled value"}
        )
        self.assertRegex(invalid.headers["x-request-id"], _REQUEST_ID_RE)
        self.assertNotEqual(
            invalid.headers["x-request-id"], "attacker-controlled value"
        )

        not_found = self.client.get("/route-that-does-not-exist")
        self.assertEqual(not_found.status_code, 404)
        self.assertRegex(not_found.headers["x-request-id"], _REQUEST_ID_RE)

    def test_cors_preflight_always_has_request_id(self) -> None:
        accepted = self.client.options(
            "/health/live",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(accepted.status_code, 200)
        self.assertRegex(accepted.headers["x-request-id"], _REQUEST_ID_RE)

        rejected = self.client.options(
            "/health/live",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(rejected.status_code, 400)
        self.assertRegex(rejected.headers["x-request-id"], _REQUEST_ID_RE)

    def test_safe_access_log_correlates_success_and_http_error(self) -> None:
        success_id = "req_" + "1" * 32
        missing_id = "req_" + "2" * 32

        with self.assertLogs(request_context.logger, level="INFO") as captured:
            success = self.client.get(
                "/health/live", headers={"X-Request-ID": success_id}
            )
            missing = self.client.get(
                "/does-not-exist", headers={"X-Request-ID": missing_id}
            )

        self.assertEqual(success.status_code, 200)
        self.assertEqual(missing.status_code, 404)
        logs = "\n".join(captured.output)
        self.assertIn(
            f"request_id={success_id} method=GET route=/health/live status=200",
            logs,
        )
        self.assertIn(
            f"request_id={missing_id} method=GET route=unmatched status=404",
            logs,
        )

    def test_value_error_is_redacted_from_body_and_logs(self) -> None:
        secret = "provider-input-value-secret"
        checker = SimpleNamespace(check=AsyncMock(side_effect=ValueError(secret)))
        request_id = "req_" + "b" * 32

        with patch.object(
            health_router, "provider_health_checker", checker
        ), self.assertLogs(main.logger, level="WARNING") as captured:
            response = self.client.get(
                "/health/providers", headers={"X-Request-ID": request_id}
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {
                "error": "参数错误",
                "detail": "请求参数无效",
                "code": "invalid_request",
                "request_id": request_id,
            },
        )
        self.assertEqual(response.headers["x-request-id"], request_id)
        self.assertNotIn(secret, response.text)
        self.assertNotIn(secret, "\n".join(captured.output))

    def test_unexpected_error_is_fixed_500_with_cors_and_request_id(self) -> None:
        secret = "postgresql://user:secret@internal/database"
        checker = SimpleNamespace(check=AsyncMock(side_effect=RuntimeError(secret)))
        origin = "http://127.0.0.1:5173"

        with patch.object(
            health_router, "provider_health_checker", checker
        ), self.assertLogs(request_context.logger, level="ERROR") as captured:
            response = self.client.get(
                "/health/providers", headers={"Origin": origin}
            )

        request_id = response.headers["x-request-id"]
        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {
                "error": "服务器内部错误",
                "detail": "请求处理失败，请稍后重试",
                "code": "internal_error",
                "request_id": request_id,
            },
        )
        self.assertRegex(request_id, _REQUEST_ID_RE)
        self.assertEqual(response.headers["access-control-allow-origin"], origin)
        self.assertNotIn(secret, response.text)
        self.assertNotIn(secret, "\n".join(captured.output))

    def test_validation_error_keeps_request_id(self) -> None:
        response = self.client.post("/agent/adaptive/start", json={})

        self.assertEqual(response.status_code, 422)
        self.assertRegex(response.headers["x-request-id"], _REQUEST_ID_RE)

    def test_grader_retry_exhaustion_keeps_provider_failure_semantics(self) -> None:
        secret = "grader-session-log-injection\nforged-entry"
        with patch.object(
            grader_agent_module,
            "with_retry",
            AsyncMock(side_effect=RetryExhausted("provider unavailable")),
        ), self.assertLogs(grader_agent_module.logger, level="ERROR") as captured:
            with self.assertRaises(RetryExhausted):
                asyncio.run(grader_agent_module._grade({"session_id": secret}))

        self.assertNotIn(secret, "\n".join(captured.output))

    def test_retry_exhaustion_does_not_log_upstream_error_text(self) -> None:
        secret = "provider-body-with-private-prompt"
        checker = SimpleNamespace(
            check=AsyncMock(side_effect=RetryExhausted(secret))
        )

        with patch.object(
            health_router, "provider_health_checker", checker
        ), self.assertLogs(main.logger, level="WARNING") as captured:
            response = self.client.get("/health/providers")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "provider_unavailable")
        self.assertNotIn(secret, response.text)
        self.assertNotIn(secret, "\n".join(captured.output))

    def test_agent_stream_error_is_redacted_and_keeps_request_id(self) -> None:
        secret = "sk-abcdefghijklmnopqrstuvwxyz-secret"
        request_id = "req_" + "c" * 32

        with patch.object(
            stream_router, "orchestrator", _FailingOrchestrator(secret)
        ), self.assertLogs(stream_router.logger, level="ERROR") as captured:
            response = self.client.post(
                "/agent/stream",
                json={"action": "quiz"},
                headers={"X-Request-ID": request_id},
            )

        payload = json.loads(response.text.removeprefix("data: ").strip())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-request-id"], request_id)
        self.assertEqual(
            payload,
            {
                "type": "error",
                "detail": "Agent 执行失败，请稍后重试",
                "code": "agent_stream_failed",
                "request_id": request_id,
            },
        )
        self.assertNotIn(secret, response.text)
        self.assertNotIn(secret, "\n".join(captured.output))

    def test_chat_midstream_error_is_redacted(self) -> None:
        secret = "provider-response-body-secret"
        request_id = "req_" + "d" * 32

        with patch.object(
            chat_router,
            "chat_stream",
            AsyncMock(return_value=_failing_chat_stream(secret)),
        ), self.assertLogs(chat_router.logger, level="ERROR") as captured:
            response = self.client.post(
                "/chat/stream",
                json={"message": "hello"},
                headers={"X-Request-ID": request_id},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("data: first", response.text)
        self.assertIn('"code": "chat_stream_failed"', response.text)
        self.assertIn(f'"request_id": "{request_id}"', response.text)
        self.assertNotIn(secret, response.text)
        self.assertNotIn(secret, "\n".join(captured.output))

    def test_concurrent_streams_keep_request_ids_isolated(self) -> None:
        request_ids = ("req_" + "e" * 32, "req_" + "f" * 32)

        async def exercise():
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await asyncio.gather(
                    client.post(
                        "/agent/stream",
                        json={"action": "quiz", "user_id": "slow"},
                        headers={"X-Request-ID": request_ids[0]},
                    ),
                    client.post(
                        "/agent/stream",
                        json={"action": "quiz", "user_id": "fast"},
                        headers={"X-Request-ID": request_ids[1]},
                    ),
                )

        with patch.object(
            stream_router,
            "orchestrator",
            _InterleavingFailingOrchestrator(),
        ), self.assertLogs(stream_router.logger, level="ERROR"):
            responses = asyncio.run(exercise())

        for response, request_id in zip(responses, request_ids, strict=True):
            self.assertEqual(response.headers["x-request-id"], request_id)
            self.assertIn(f'"request_id": "{request_id}"', response.text)
            other = request_ids[1] if request_id == request_ids[0] else request_ids[0]
            self.assertNotIn(other, response.text)

    def test_send_disconnect_is_not_reported_as_application_failure(self) -> None:
        async def app(_scope, _receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [],
                }
            )

        async def disconnected_send(_message):
            raise OSError("client transport closed")

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/disconnect",
            "headers": [],
        }

        middleware = request_context.SafeErrorMiddleware(app)
        with self.assertNoLogs(request_context.logger, level="ERROR"):
            asyncio.run(
                middleware(scope, AsyncMock(), disconnected_send)
            )

    def test_started_response_application_failure_is_safely_terminated(self) -> None:
        secret = "response-started-application-secret"
        messages = []

        async def app(_scope, _receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [],
                }
            )
            await send({
                "type": "http.response.body",
                "body": b"partial",
                "more_body": True,
            })
            raise RuntimeError(secret)

        async def capture_send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/partial",
            "headers": [],
        }

        middleware = request_context.SafeErrorMiddleware(app)
        with self.assertLogs(request_context.logger, level="ERROR") as captured:
            asyncio.run(middleware(scope, AsyncMock(), capture_send))

        self.assertEqual(messages[-1]["type"], "http.response.body")
        self.assertFalse(messages[-1]["more_body"])
        self.assertNotIn(secret, "\n".join(captured.output))

    def test_error_response_disconnect_does_not_escape_or_log_secret(self) -> None:
        secret = "application-error-before-disconnected-response"

        async def app(_scope, _receive, _send):
            raise RuntimeError(secret)

        async def disconnected_send(_message):
            raise OSError("client transport closed")

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/disconnect-during-error",
            "headers": [],
        }

        middleware = request_context.SafeErrorMiddleware(app)
        with self.assertLogs(request_context.logger, level="ERROR") as captured:
            asyncio.run(middleware(scope, AsyncMock(), disconnected_send))

        self.assertNotIn(secret, "\n".join(captured.output))

    def test_error_response_cancellation_suppresses_original_exception(self) -> None:
        secret = "application-secret-before-cancelled-error-response"

        async def app(_scope, _receive, _send):
            raise RuntimeError(secret)

        async def cancelled_send(_message):
            raise asyncio.CancelledError

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/cancel-during-error",
            "headers": [],
        }

        middleware = request_context.SafeErrorMiddleware(app)
        with self.assertLogs(request_context.logger, level="ERROR") as captured:
            with self.assertRaises(asyncio.CancelledError) as raised:
                asyncio.run(middleware(scope, AsyncMock(), cancelled_send))

        formatted = "".join(traceback.format_exception(raised.exception))
        self.assertNotIn(secret, formatted)
        self.assertNotIn(secret, "\n".join(captured.output))

    def test_inner_error_handler_send_cancellation_suppresses_context(self) -> None:
        secret = "inner-handler-secret-before-cancelled-send"

        async def app(_scope, _receive, send):
            try:
                raise RuntimeError(secret)
            except RuntimeError:
                await send({
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [],
                })

        async def cancelled_send(_message):
            raise asyncio.CancelledError

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/inner-cancel",
            "headers": [],
        }

        middleware = request_context.SafeErrorMiddleware(app)
        with self.assertRaises(asyncio.CancelledError) as raised:
            asyncio.run(middleware(scope, AsyncMock(), cancelled_send))

        formatted = "".join(traceback.format_exception(raised.exception))
        self.assertNotIn(secret, formatted)

    def test_body_disconnect_is_logged_as_incomplete_not_completed(self) -> None:
        sent = []

        async def app(_scope, _receive, send):
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            })
            await send({
                "type": "http.response.body",
                "body": b"payload",
                "more_body": False,
            })

        async def disconnect_on_body(message):
            if message["type"] == "http.response.body":
                raise OSError("client transport closed")
            sent.append(message)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/body-disconnect",
            "headers": [],
        }
        middleware = request_context.RequestContextMiddleware(
            request_context.SafeErrorMiddleware(app)
        )

        with self.assertLogs(request_context.logger, level="WARNING") as captured:
            asyncio.run(middleware(scope, AsyncMock(), disconnect_on_body))

        logs = "\n".join(captured.output)
        self.assertEqual(len(sent), 1)
        self.assertIn("request incomplete", logs)
        self.assertNotIn("request completed", logs)


if __name__ == "__main__":
    unittest.main()
