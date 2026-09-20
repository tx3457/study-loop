"""HTTP boundaries for provider failures, local CORS, and first-user state."""

import asyncio
import copy
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from chromadb.errors import InternalError, NotFoundError
from openai import APIConnectionError, APITimeoutError, RateLimitError

import routers.autonomous as autonomous_router
import routers.chat as chat_router
import routers.documents as documents_router
import routers.learning_path as learning_path_router
import routers.user as user_router
import services.learning_path as learning_path_service
import services.tools as tool_module
from main import app
from models.learning_path import CompressedReport, LearningPathWire, PathBrief
from services.autonomous_sessions import AutonomousSessionStore
from services.provider_config import ProviderDeadlineExceeded
from services.retry import RetryExhausted
from services.tool_registry import SideEffectAmbiguousError, tool_registry
from services.vectorstore import DocumentAlreadyExistsError


def _paused_autonomous_session(conversation_id: str):
    return autonomous_router.AutonomousSession(
        conversation_id=conversation_id,
        messages=[
            {"role": "system", "content": "test"},
            {"role": "user", "content": "learn"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "ask-1",
                    "type": "function",
                    "function": {
                        "name": "ask_user",
                        "arguments": '{"question":"continue?"}',
                    },
                }],
            },
        ],
        plan=[],
        steps=[autonomous_router.StepRecord(
            round_index=0,
            tool_name="ask_user",
            tool_args={"question": "continue?"},
        )],
        tools_called=[],
        rounds_used=1,
        user_id="default_user",
        document_id=None,
        evidence_registry={},
        grounding_required=False,
        pending_ask_call_id="ask-1",
    )


class TestApiErrorBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_session_store = autonomous_router.autonomous_sessions
        self.session_db_path = str(Path(self.temp_dir.name) / "sessions.sqlite3")
        self.session_store = AutonomousSessionStore(
            sqlite_path=self.session_db_path
        )
        autonomous_router.autonomous_sessions = self.session_store

    def tearDown(self):
        autonomous_router.autonomous_sessions = self.original_session_store
        self.temp_dir.cleanup()

    def _seed_session(self, conversation_id: str):
        session = _paused_autonomous_session(conversation_id)
        asyncio.run(self.session_store.save(
            conversation_id,
            autonomous_router._session_to_payload(session),
        ))
        return session

    def _inspect_session(self, conversation_id: str):
        return asyncio.run(self.session_store.inspect(conversation_id))

    def test_dev_cors_allows_localhost_and_loopback_but_not_unknown_origins(self):
        for origin in ("http://localhost:5173", "http://127.0.0.1:5173"):
            with self.subTest(origin=origin):
                response = self.client.options(
                    "/documents",
                    headers={
                        "Origin": origin,
                        "Access-Control-Request-Method": "GET",
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers.get("access-control-allow-origin"), origin)

        response = self.client.options(
            "/documents",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIsNone(response.headers.get("access-control-allow-origin"))

    def test_missing_profile_is_a_normal_empty_state(self):
        origin = "http://127.0.0.1:5173"
        with patch.object(user_router, "get_user_profile", AsyncMock(return_value=None)):
            response = self.client.get(
                "/user/default_user/profile", headers={"Origin": origin}
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json())
        self.assertEqual(response.headers.get("access-control-allow-origin"), origin)

    def test_learning_path_provider_error_is_stable_503_with_cors(self):
        origin = "http://127.0.0.1:5173"
        provider_error = APIConnectionError(
            request=httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
        )
        with patch.object(
            learning_path_router,
            "generate_learning_path",
            AsyncMock(side_effect=provider_error),
        ):
            response = self.client.post(
                "/learning-path/notes.md", headers={"Origin": origin}
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {
                "error": "服务暂时不可用",
                "detail": "模型服务请求失败",
                "code": "provider_unavailable",
            },
        )
        self.assertEqual(response.headers.get("access-control-allow-origin"), origin)

    def test_autonomous_provider_error_is_not_reported_as_max_rounds(self):
        origin = "http://127.0.0.1:5173"
        finish = AsyncMock()
        with patch.object(
            autonomous_router, "check_injection", AsyncMock(return_value=(False, ""))
        ), patch.object(
            autonomous_router,
            "run_tool_round",
            AsyncMock(side_effect=RetryExhausted("provider down")),
        ), patch.object(autonomous_router, "llm_chat", finish):
            response = self.client.post(
                "/agent/autonomous",
                headers={"Origin": origin},
                json={"query": "学RAG", "user_id": "new-user"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {
                "error": "服务暂时不可用",
                "detail": "模型服务请求失败",
                "code": "provider_unavailable",
            },
        )
        self.assertFalse(response.json().get("truncated", False))
        self.assertNotEqual(response.json().get("finalize_reason"), "max_rounds_truncated")
        self.assertEqual(response.headers.get("access-control-allow-origin"), origin)
        finish.assert_not_awaited()

    def test_autonomous_request_size_limits_fail_before_agent_execution(self):
        injection_check = AsyncMock()
        with patch.object(
            autonomous_router, "check_injection", injection_check
        ):
            empty = self.client.post(
                "/agent/autonomous",
                json={"query": "", "user_id": "user"},
            )
            oversized = self.client.post(
                "/agent/autonomous/continue",
                json={
                    "conversation_id": "conv",
                    "user_reply": "x" * 8001,
                },
            )

        self.assertEqual(empty.status_code, 422)
        self.assertEqual(oversized.status_code, 422)
        injection_check.assert_not_awaited()

    def test_tool_chat_request_boundaries_fail_before_agent_execution(self):
        invalid_payloads = [
            {"message": ""},
            {"message": " "},
            {"message": "x" * 8001},
            {"message": "x", "user_id": " "},
            {"message": "x", "user_id": "x" * 129},
            {"message": "x", "user_id": "user\nadmin"},
            {"message": "x", "user_id": "user\x7f"},
            {"message": "x", "document_id": " "},
            {"message": "x", "document_id": "x" * 513},
            {"message": "x", "document_id": "doc\tother"},
            {"message": "x", "unexpected": True},
        ]
        injection_check = AsyncMock()

        with patch.object(chat_router, "check_injection", injection_check):
            for payload in invalid_payloads:
                with self.subTest(payload=payload):
                    response = self.client.post("/chat/tools", json=payload)
                    self.assertEqual(response.status_code, 422, response.text)

        injection_check.assert_not_awaited()

    def test_invalid_persisted_session_version_is_consumed_fail_closed(self):
        conversation_id = "invalid-session-version"
        secret = "sk-123456789012345678901234"
        self._seed_session(conversation_id)
        self._corrupt_session_payload(conversation_id, {
                "schema_version": 2,
                "messages": [{"role": "system", "content": secret}],
            })
        run_loop = AsyncMock()

        with self.assertLogs(
            autonomous_router.logger, level="ERROR"
        ) as captured, patch.object(
            autonomous_router,
            "check_injection",
            AsyncMock(return_value=(False, "")),
        ), patch.object(autonomous_router, "_run_react_loop", run_loop):
            response = self.client.post(
                "/agent/autonomous/continue",
                json={"conversation_id": conversation_id, "user_reply": "继续"},
            )

        self.assertEqual(response.status_code, 410)
        self.assertIn("暂停会话数据无效", response.json()["detail"])
        self.assertIsNone(self._inspect_session(conversation_id))
        run_loop.assert_not_awaited()
        self.assertNotIn(secret, "\n".join(captured.output))

    def test_semantically_invalid_pause_snapshots_are_consumed_fail_closed(self):
        valid = autonomous_router._session_to_payload(
            _paused_autonomous_session("template")
        )
        cases = {
            "too-many-rounds": {"rounds_used": 999},
            "empty-user": {"user_id": ""},
            "empty-call-id": {"pending_ask_call_id": ""},
            "missing-call": {"pending_ask_call_id": "unknown-call"},
        }

        for label, changes in cases.items():
            with self.subTest(label=label):
                conversation_id = f"invalid-{label}"
                payload = copy.deepcopy(valid)
                payload.update(changes)
                self._seed_session(conversation_id)
                self._corrupt_session_payload(conversation_id, payload)
                run_loop = AsyncMock()

                with patch.object(
                    autonomous_router,
                    "check_injection",
                    AsyncMock(return_value=(False, "")),
                ), patch.object(autonomous_router, "_run_react_loop", run_loop):
                    response = self.client.post(
                        "/agent/autonomous/continue",
                        json={
                            "conversation_id": conversation_id,
                            "user_reply": "继续",
                        },
                    )

                self.assertEqual(response.status_code, 410)
                self.assertIsNone(self._inspect_session(conversation_id))
                run_loop.assert_not_awaited()

    def _corrupt_session_payload(self, conversation_id, payload):
        # Corrupt an already-owned snapshot without erasing its durable owner.
        with closing(sqlite3.connect(self.session_db_path)) as connection:
            connection.execute(
                "UPDATE studyloop_autonomous_sessions SET payload_json = ? "
                "WHERE conversation_id = ?",
                (json.dumps(payload), conversation_id),
            )
            connection.commit()

    def test_legacy_snapshot_without_owner_is_not_claimed_or_consumed(self):
        conversation_id = "unowned-legacy-snapshot"
        asyncio.run(self.session_store.save(conversation_id, {"schema_version": 2}))
        run_loop = AsyncMock()
        with patch.object(autonomous_router, "_run_react_loop", run_loop):
            response = self.client.post(
                "/agent/autonomous/continue",
                json={"conversation_id": conversation_id, "user_reply": "继续"},
            )
        self.assertEqual(response.status_code, 404)
        self.assertTrue(asyncio.run(self.session_store.exists_without_reaping(conversation_id)))
        run_loop.assert_not_awaited()

    def test_corrupt_pause_json_is_consumed_before_provider_execution(self):
        conversation_id = "corrupt-session-json"
        self._seed_session(conversation_id)
        with closing(sqlite3.connect(self.session_db_path)) as connection:
            connection.execute(
                """
                UPDATE studyloop_autonomous_sessions
                SET payload_json = '{'
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            )
            connection.commit()
        run_loop = AsyncMock()

        with patch.object(
            autonomous_router,
            "check_injection",
            AsyncMock(return_value=(False, "")),
        ), patch.object(autonomous_router, "_run_react_loop", run_loop):
            response = self.client.post(
                "/agent/autonomous/continue",
                json={"conversation_id": conversation_id, "user_reply": "继续"},
            )

        self.assertEqual(response.status_code, 410)
        self.assertIsNone(self._inspect_session(conversation_id))
        run_loop.assert_not_awaited()

    def test_autonomous_continue_provider_failure_before_progress_stays_retryable(self):
        origin = "http://127.0.0.1:5173"
        conversation_id = "retryable-provider-failure"
        session = self._seed_session(conversation_id)

        with patch.object(
            autonomous_router,
            "check_injection",
            AsyncMock(return_value=(False, "")),
        ), patch.object(
            autonomous_router,
            "_run_react_loop",
            AsyncMock(side_effect=RetryExhausted("provider down")),
        ):
            response = self.client.post(
                "/agent/autonomous/continue",
                headers={"Origin": origin},
                json={"conversation_id": conversation_id, "user_reply": "第三章"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {
                "error": "服务暂时不可用",
                "detail": "模型服务请求失败",
                "code": "provider_unavailable",
            },
        )
        restored = self._inspect_session(conversation_id)
        self.assertEqual(
            restored.payload,
            autonomous_router._session_to_payload(session),
        )
        self.assertEqual(response.headers.get("access-control-allow-origin"), origin)

    def test_autonomous_continue_failure_after_progress_returns_gone(self):
        origin = "http://127.0.0.1:5173"
        conversation_id = "consumed-provider-failure"
        self._seed_session(conversation_id)

        async def fail_after_progress(**kwargs):
            await kwargs["on_before_tool_calls"]()
            await kwargs["on_before_tool_dispatch"]()
            kwargs["tools_called"].append("update_learning_profile")
            kwargs["steps"].append(
                autonomous_router.StepRecord(
                    round_index=1,
                    tool_name="update_learning_profile",
                )
            )
            raise RetryExhausted("provider down after write")

        with patch.object(
            autonomous_router,
            "check_injection",
            AsyncMock(return_value=(False, "")),
        ), patch.object(
            autonomous_router,
            "_run_react_loop",
            AsyncMock(side_effect=fail_after_progress),
        ):
            response = self.client.post(
                "/agent/autonomous/continue",
                headers={"Origin": origin},
                json={"conversation_id": conversation_id, "user_reply": "继续"},
            )

        self.assertEqual(response.status_code, 410)
        self.assertEqual(
            response.json(),
            {"detail": "续跑已执行部分操作，无法安全重试；请重新开始"},
        )
        self.assertIsNone(self._inspect_session(conversation_id))
        self.assertEqual(response.headers.get("access-control-allow-origin"), origin)

    def test_autonomous_continue_dispatched_tool_without_trajectory_returns_gone(self):
        origin = "http://127.0.0.1:5173"
        conversation_id = "audited-provider-failure"
        self._seed_session(conversation_id)
        tool = tool_registry.get("update_learning_profile")
        self.assertIsNotNone(tool)
        original_audit = list(tool_registry._audit_log)
        writes = []
        captured_run_id = []

        async def fake_write(**kwargs):
            writes.append(kwargs)
            return '{"status":"updated"}'

        async def fail_after_committed_write(**kwargs):
            captured_run_id.append(kwargs["run_id"])
            await tool_module.dispatch_tool(
                "update_learning_profile",
                {
                    "user_id": "default_user",
                    "document_id": "notes.md",
                    "grade_result": {"score": 1.0},
                },
                run_id=kwargs["run_id"],
                user_id="default_user",
            )
            raise RetryExhausted("provider down after committed write")

        try:
            with patch.object(tool, "handler", new=fake_write), patch.object(
                autonomous_router,
                "check_injection",
                AsyncMock(return_value=(False, "")),
            ), patch.object(
                autonomous_router,
                "_run_react_loop",
                AsyncMock(side_effect=fail_after_committed_write),
            ):
                response = self.client.post(
                    "/agent/autonomous/continue",
                    headers={"Origin": origin},
                    json={"conversation_id": conversation_id, "user_reply": "继续"},
                )

            self.assertEqual(response.status_code, 410)
            self.assertEqual(
                response.json(),
                {"detail": "续跑已执行部分操作，无法安全重试；请重新开始"},
            )
            self.assertEqual(len(writes), 1)
            records = tool_registry.get_audit(run_id=captured_run_id[0])
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].tool_name, "update_learning_profile")
            self.assertEqual(records[0].status, "ok")
            self.assertIsNone(self._inspect_session(conversation_id))

            second = self.client.post(
                "/agent/autonomous/continue",
                headers={"Origin": origin},
                json={"conversation_id": conversation_id, "user_reply": "继续"},
            )
            self.assertEqual(second.status_code, 404)
            self.assertEqual(len(writes), 1)
            self.assertEqual(response.headers.get("access-control-allow-origin"), origin)
        finally:
            tool_registry._audit_log[:] = original_audit

    def test_autonomous_continue_ambiguous_side_effect_returns_gone(self):
        origin = "http://127.0.0.1:5173"
        conversation_id = "ambiguous-side-effect"
        self._seed_session(conversation_id)
        tool = tool_registry.get("update_learning_profile")
        self.assertIsNotNone(tool)
        original_audit = list(tool_registry._audit_log)

        tool_call = SimpleNamespace(
            id="update-1",
            function=SimpleNamespace(
                name="update_learning_profile",
                arguments=(
                    '{"user_id":"default_user","document_id":"notes.md",'
                    '"grade_result":{"score":1.0}}'
                ),
            ),
        )
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[tool_call],
            ))]
        ))
        handler_calls = []

        async def ambiguous_handler(**kwargs):
            inspection = await self.session_store.inspect(conversation_id)
            handler_calls.append((inspection.state, inspection.progress_started))
            raise asyncio.TimeoutError

        try:
            with patch.object(tool, "handler", new=ambiguous_handler), patch.object(
                autonomous_router, "_client", client
            ), patch.object(
                autonomous_router,
                "check_injection",
                AsyncMock(return_value=(False, "")),
            ):
                response = self.client.post(
                    "/agent/autonomous/continue",
                    headers={"Origin": origin},
                    json={"conversation_id": conversation_id, "user_reply": "继续"},
                )

            self.assertEqual(response.status_code, 410)
            self.assertEqual(
                response.json(),
                {"detail": "续跑已执行部分操作，无法安全重试；请重新开始"},
            )
            self.assertEqual(handler_calls, [("in_flight", True)])
            records = tool_registry._audit_log[len(original_audit):]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].tool_name, "update_learning_profile")
            self.assertEqual(records[0].status, "ambiguous")
            self.assertIsNone(self._inspect_session(conversation_id))
            self.assertEqual(response.headers.get("access-control-allow-origin"), origin)
        finally:
            tool_registry._audit_log[:] = original_audit

    def test_initial_ambiguous_side_effect_is_non_retryable_conflict(self):
        origin = "http://127.0.0.1:5173"
        with patch.object(
            autonomous_router,
            "check_injection",
            AsyncMock(return_value=(False, "")),
        ), patch.object(
            autonomous_router,
            "_run_react_loop",
            AsyncMock(side_effect=SideEffectAmbiguousError(
                "update_learning_profile"
            )),
        ):
            response = self.client.post(
                "/agent/autonomous",
                headers={"Origin": origin},
                json={"query": "学RAG", "user_id": "default_user"},
            )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json(),
            {
                "detail": "工具执行结果不确定，请勿自动重试；请刷新学习状态后重新开始",
                "code": "side_effect_ambiguous",
                "reason": "ambiguous",
            },
        )
        self.assertNotIn("update_learning_profile", response.text)
        self.assertEqual(response.headers.get("access-control-allow-origin"), origin)

    def test_initial_provider_failure_after_committed_write_is_conflict(self):
        origin = "http://127.0.0.1:5173"
        tool = tool_registry.get("update_learning_profile")
        self.assertIsNotNone(tool)
        original_audit = list(tool_registry._audit_log)
        writes = []

        async def fake_write(**kwargs):
            writes.append(kwargs)
            return '{"status":"updated"}'

        async def fail_after_write(**kwargs):
            await tool_module.dispatch_tool(
                "update_learning_profile",
                {
                    "user_id": "default_user",
                    "document_id": "notes.md",
                    "grade_result": {"score": 1.0},
                },
                run_id=kwargs["run_id"],
                user_id="default_user",
            )
            raise RetryExhausted("provider down after committed write")

        try:
            with patch.object(tool, "handler", new=fake_write), patch.object(
                autonomous_router,
                "check_injection",
                AsyncMock(return_value=(False, "")),
            ), patch.object(
                autonomous_router,
                "_run_react_loop",
                AsyncMock(side_effect=fail_after_write),
            ):
                response = self.client.post(
                    "/agent/autonomous",
                    headers={"Origin": origin},
                    json={"query": "学RAG", "user_id": "default_user"},
                )

            self.assertEqual(response.status_code, 409)
            self.assertEqual(len(writes), 1)
            records = tool_registry._audit_log[len(original_audit):]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].status, "ok")
            self.assertNotIn("update_learning_profile", response.text)
            self.assertEqual(response.headers.get("access-control-allow-origin"), origin)
        finally:
            tool_registry._audit_log[:] = original_audit

    def test_tool_chat_ambiguous_side_effect_is_non_retryable_conflict(self):
        origin = "http://127.0.0.1:5173"
        with patch.object(
            chat_router,
            "check_injection",
            AsyncMock(return_value=(False, "")),
        ), patch.object(
            chat_router,
            "run_tool_round",
            AsyncMock(side_effect=SideEffectAmbiguousError(
                "update_learning_profile"
            )),
        ):
            response = self.client.post(
                "/chat/tools",
                headers={"Origin": origin},
                json={"message": "更新画像", "user_id": "default_user"},
            )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json(),
            {
                "detail": "工具执行结果不确定，请勿自动重试；请刷新学习状态后重新开始",
                "code": "side_effect_ambiguous",
                "reason": "ambiguous",
            },
        )
        self.assertNotIn("update_learning_profile", response.text)
        self.assertEqual(response.headers.get("access-control-allow-origin"), origin)

    def test_tool_chat_provider_failure_after_committed_write_is_conflict(self):
        origin = "http://127.0.0.1:5173"
        tool = tool_registry.get("update_learning_profile")
        self.assertIsNotNone(tool)
        original_audit = list(tool_registry._audit_log)
        writes = []
        rounds = 0

        async def fake_write(**kwargs):
            writes.append(kwargs)
            return '{"status":"updated"}'

        async def write_then_fail(*args, **kwargs):
            nonlocal rounds
            rounds += 1
            if rounds == 1:
                await tool_module.dispatch_tool(
                    "update_learning_profile",
                    {
                        "user_id": "default_user",
                        "document_id": "notes.md",
                        "grade_result": {"score": 1.0},
                    },
                    run_id=kwargs["run_id"],
                    user_id="default_user",
                )
                return SimpleNamespace(
                    has_tool_calls=True,
                    outcomes=[SimpleNamespace(
                        kind="dispatched",
                        name="update_learning_profile",
                    )],
                )
            raise RetryExhausted("provider down after committed write")

        try:
            with patch.object(tool, "handler", new=fake_write), patch.object(
                chat_router,
                "check_injection",
                AsyncMock(return_value=(False, "")),
            ), patch.object(
                chat_router,
                "run_tool_round",
                AsyncMock(side_effect=write_then_fail),
            ):
                response = self.client.post(
                    "/chat/tools",
                    headers={"Origin": origin},
                    json={"message": "更新画像", "user_id": "default_user"},
                )

            self.assertEqual(response.status_code, 409)
            self.assertEqual(len(writes), 1)
            records = tool_registry._audit_log[len(original_audit):]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].status, "ok")
            self.assertNotIn("update_learning_profile", response.text)
            self.assertEqual(response.headers.get("access-control-allow-origin"), origin)
        finally:
            tool_registry._audit_log[:] = original_audit

    def test_upload_provider_failure_and_duplicate_have_explicit_statuses(self):
        origin = "http://127.0.0.1:5173"
        files = {"file": ("notes.md", b"# Notes\n\nRAG content", "text/markdown")}

        with patch.object(
            documents_router,
            "deal_document",
            AsyncMock(side_effect=RetryExhausted("embedding unavailable")),
        ):
            response = self.client.post(
                "/documents/upload", headers={"Origin": origin}, files=files
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {
                "error": "服务暂时不可用",
                "detail": "模型服务请求失败",
                "code": "provider_unavailable",
            },
        )
        self.assertEqual(response.headers.get("access-control-allow-origin"), origin)

        with patch.object(
            documents_router,
            "deal_document",
            AsyncMock(side_effect=DocumentAlreadyExistsError("文档已存在")),
        ):
            response = self.client.post(
                "/documents/upload", headers={"Origin": origin}, files=files
            )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json(), {"detail": "文档已存在"})

    def test_delete_reports_owner_mismatch_and_storage_failure(self):
        with patch.object(
            documents_router,
            "delete_document",
            AsyncMock(side_effect=NotFoundError("missing")),
        ):
            response = self.client.delete("/documents/missing.md")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "文档不存在"})

        with patch.object(
            documents_router,
            "delete_document",
            AsyncMock(return_value="material_deleted"),
        ):
            response = self.client.delete("/documents/notes.md")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "material_deleted",
                "document_id": "notes.md",
                "scope": "material_only",
                "learning_data_retained": True,
                "document_id_reusable": False,
            },
        )

        with patch.object(
            documents_router,
            "delete_document",
            AsyncMock(side_effect=InternalError("storage unavailable")),
        ):
            response = self.client.delete("/documents/notes.md")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "文档存储暂时不可用"})

    def test_document_list_storage_failure_is_json_503(self):
        with patch.object(
            documents_router,
            "get_all_document",
            AsyncMock(side_effect=InternalError("storage unavailable")),
        ):
            response = self.client.get("/documents")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "文档存储暂时不可用"})

    def test_learning_path_retry_exhaustion_hides_provider_details(self):
        origin = "http://127.0.0.1:5173"
        with patch.object(
            learning_path_router,
            "generate_learning_path",
            AsyncMock(side_effect=RetryExhausted("internal-model-42 upstream body")),
        ):
            response = self.client.post(
                "/learning-path/notes.md", headers={"Origin": origin}
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {
                "error": "服务暂时不可用",
                "detail": "模型服务请求失败",
                "code": "provider_unavailable",
            },
        )
        self.assertNotIn("internal-model-42", response.text)
        self.assertEqual(response.headers.get("access-control-allow-origin"), origin)

    def test_provider_deadline_is_stable_504(self):
        with patch.object(
            learning_path_router,
            "generate_learning_path",
            AsyncMock(side_effect=ProviderDeadlineExceeded(30.0)),
        ):
            response = self.client.post("/learning-path/notes.md")

        self.assertEqual(response.status_code, 504)
        self.assertEqual(
            response.json(),
            {
                "error": "模型服务请求超时",
                "detail": "模型服务未在时间预算内响应",
                "code": "provider_timeout",
            },
        )

    def test_provider_sdk_timeout_is_stable_504(self):
        timeout = APITimeoutError(
            request=httpx.Request(
                "POST", "https://provider.invalid/v1/chat/completions"
            )
        )
        with patch.object(
            learning_path_router,
            "generate_learning_path",
            AsyncMock(side_effect=timeout),
        ):
            response = self.client.post("/learning-path/notes.md")

        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["code"], "provider_timeout")

    def test_retry_exhausted_rate_limit_is_stable_429(self):
        request = httpx.Request(
            "POST", "https://provider.invalid/v1/chat/completions"
        )
        upstream = RateLimitError(
            "rate limited",
            response=httpx.Response(429, request=request),
            body=None,
        )
        exhausted = RetryExhausted("provider retries exhausted")
        exhausted.__cause__ = upstream

        with patch.object(
            learning_path_router,
            "generate_learning_path",
            AsyncMock(side_effect=exhausted),
        ):
            response = self.client.post("/learning-path/notes.md")

        self.assertEqual(response.status_code, 429)
        self.assertEqual(
            response.json(),
            {
                "error": "模型服务请求过于频繁",
                "detail": "模型服务当前限流，请稍后重试",
                "code": "provider_rate_limited",
            },
        )


class TestLearningPathProviderBoundary(unittest.IsolatedAsyncioTestCase):
    async def test_synthesize_overrides_model_controlled_document_id(self):
        brief = PathBrief(
            title="RAG 学习路径",
            scope="RAG 基础",
            level="beginner",
            target_count=1,
            keywords=["retrieval"],
        )
        compressed = CompressedReport(
            summary="RAG combines retrieval and generation.",
            key_concepts=["retrieval"],
            suggested_stage_count=1,
        )
        parsed = {
            "document_id": "model-chosen.md",
            "title": "RAG 基础",
            "total_stages": 1,
            "stages": [
                {
                    "stage": 1,
                    "title": "检索基础",
                    "topics": ["retrieval"],
                    "description": "理解检索增强生成中的召回步骤。",
                    "estimated_minutes": 20,
                }
            ],
        }
        parse = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
            )
        )

        with patch.object(learning_path_service, "llm_parse", parse):
            path = await learning_path_service.synthesize(
                "notes.md", brief, compressed
            )

        self.assertEqual(path.document_id, "notes.md")
        self.assertEqual(path.total_stages, 1)
        self.assertIs(parse.await_args.args[1], LearningPathWire)

    async def test_synthesize_uses_retrying_llm_entrypoint(self):
        brief = PathBrief(
            title="RAG 学习路径",
            scope="RAG 基础",
            level="beginner",
            target_count=3,
            keywords=["retrieval", "generation"],
        )
        compressed = CompressedReport(
            summary="RAG combines retrieval and generation.",
            key_concepts=["retrieval", "generation"],
            suggested_stage_count=3,
        )
        parse = AsyncMock(side_effect=RetryExhausted("provider down"))

        with patch.object(learning_path_service, "llm_parse", parse):
            with self.assertRaises(RetryExhausted):
                await learning_path_service.synthesize("notes.md", brief, compressed)

        parse.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
