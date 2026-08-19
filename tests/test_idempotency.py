"""Persistent request idempotency and durable side-effect guards."""

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import routers.autonomous as autonomous_router
import services.tool_registry as registry_module
from main import app
from services.autonomous_sessions import AutonomousSessionStore
from services.idempotency import (
    IdempotencyConflictError,
    IdempotencyStore,
)
from services.tool_registry import tool_registry


class TestIdempotencyStore(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "receipts.sqlite3")
        self.store = IdempotencyStore(sqlite_path=self.path)

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_completed_response_replays_after_store_recreation(self):
        decision = await self.store.begin(
            "request-1", "agent.autonomous", {"query": "learn RAG"}
        )
        self.assertFalse(decision.replayed)

        response = {"final_answer": "done", "tools_called": []}
        await self.store.complete("request-1", response)

        reopened = IdempotencyStore(sqlite_path=self.path)
        replay = await reopened.begin(
            "request-1", "agent.autonomous", {"query": "learn RAG"}
        )
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.response, response)

    async def test_concurrent_claim_allows_only_one_owner(self):
        peer = IdempotencyStore(sqlite_path=self.path)

        async def claim(store):
            try:
                return await store.begin(
                    "request-race", "agent.autonomous", {"query": "q"}
                )
            except IdempotencyConflictError as exc:
                return exc.reason

        results = await asyncio.gather(claim(self.store), claim(peer))
        owners = [result for result in results if not isinstance(result, str)]
        conflicts = [result for result in results if isinstance(result, str)]
        self.assertEqual(len(owners), 1)
        self.assertEqual(conflicts, ["in_progress"])

    async def test_same_key_with_different_payload_is_conflict(self):
        await self.store.begin(
            "request-conflict", "agent.autonomous", {"query": "first"}
        )

        with self.assertRaises(IdempotencyConflictError) as raised:
            await self.store.begin(
                "request-conflict", "agent.autonomous", {"query": "second"}
            )

        self.assertEqual(raised.exception.reason, "payload_mismatch")

    async def test_clean_failure_releases_claim_for_retry(self):
        await self.store.begin(
            "request-retry", "agent.autonomous", {"query": "q"}
        )
        effect_started = await self.store.abort("request-retry")
        self.assertFalse(effect_started)

        retry = await self.store.begin(
            "request-retry", "agent.autonomous", {"query": "q"}
        )
        self.assertFalse(retry.replayed)

    async def test_cancelled_begin_releases_a_claim_committed_by_worker_thread(self):
        committed = threading.Event()
        release_worker = threading.Event()
        original_begin = self.store._begin_sync

        def pause_after_commit(*args):
            decision = original_begin(*args)
            committed.set()
            release_worker.wait(timeout=5)
            return decision

        with patch.object(self.store, "_begin_sync", side_effect=pause_after_commit):
            claim = asyncio.create_task(
                self.store.begin(
                    "request-cancelled-claim",
                    "agent.autonomous",
                    {"query": "q"},
                )
            )
            self.assertTrue(await asyncio.to_thread(committed.wait, 5))
            claim.cancel()
            release_worker.set()
            with self.assertRaises(asyncio.CancelledError):
                await claim

        retry = await self.store.begin(
            "request-cancelled-claim", "agent.autonomous", {"query": "q"}
        )
        self.assertFalse(retry.replayed)

    async def test_repeated_cancellation_still_releases_committed_claim(self):
        committed = threading.Event()
        release_worker = threading.Event()
        original_begin = self.store._begin_sync

        def pause_after_commit(*args):
            decision = original_begin(*args)
            committed.set()
            release_worker.wait(timeout=5)
            return decision

        with patch.object(self.store, "_begin_sync", side_effect=pause_after_commit):
            claim = asyncio.create_task(
                self.store.begin(
                    "request-double-cancel",
                    "agent.autonomous",
                    {"query": "q"},
                )
            )
            self.assertTrue(await asyncio.to_thread(committed.wait, 5))
            claim.cancel()
            await asyncio.sleep(0)
            claim.cancel()
            release_worker.set()
            with self.assertRaises(asyncio.CancelledError):
                await claim

        retry = await self.store.begin(
            "request-double-cancel", "agent.autonomous", {"query": "q"}
        )
        self.assertFalse(retry.replayed)

    async def test_effect_started_failure_stays_ambiguous_after_recreation(self):
        await self.store.begin(
            "request-effect", "agent.autonomous", {"query": "q"}
        )
        await self.store.mark_effect_started(
            "request-effect", "update_learning_profile"
        )
        effect_started = await self.store.abort("request-effect")
        self.assertTrue(effect_started)

        reopened = IdempotencyStore(sqlite_path=self.path)
        with self.assertRaises(IdempotencyConflictError) as raised:
            await reopened.begin(
                "request-effect", "agent.autonomous", {"query": "q"}
            )
        self.assertEqual(raised.exception.reason, "ambiguous")

    async def test_non_replayable_tool_marks_receipt_before_handler_runs(self):
        key = "request-tool-effect"
        await self.store.begin(key, "agent.autonomous", {"query": "q"})
        tool = tool_registry.get("update_learning_profile")
        self.assertIsNotNone(tool)
        original_audit = list(tool_registry._audit_log)
        marker_seen = []

        async def handler(**kwargs):
            marker_seen.append(await self.store.has_effect_started(key))
            return '{"status":"updated"}'

        try:
            with patch.object(
                registry_module, "request_idempotency", self.store
            ), patch.object(tool, "handler", new=handler):
                result = await tool_registry.invoke(
                    "update_learning_profile",
                    {
                        "user_id": "u",
                        "document_id": "d",
                        "grade_result": {"score": 1.0},
                    },
                    idempotency_key=key,
                )

            self.assertEqual(result, '{"status":"updated"}')
            self.assertEqual(marker_seen, [True])
        finally:
            tool_registry._audit_log[:] = original_audit


class TestAutonomousIdempotencyBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.database_path = str(Path(self.tmp.name) / "api-state.sqlite3")
        self.store = IdempotencyStore(
            sqlite_path=self.database_path
        )
        self.session_store = AutonomousSessionStore(
            sqlite_path=self.database_path
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _seed_session(self, conversation_id: str):
        session = autonomous_router.AutonomousSession(
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
            user_id="u",
            document_id=None,
            evidence_registry={},
            grounding_required=False,
            pending_ask_call_id="ask-1",
        )
        asyncio.run(self.session_store.save(
            conversation_id,
            autonomous_router._session_to_payload(session),
        ))
        return session

    def test_completed_request_is_replayed_without_rerunning_agent(self):
        result = autonomous_router.AutonomousResponse(
            final_answer="cached answer",
            rounds_used=1,
            finalize_reason="explicit_finalize",
        )
        run_loop = AsyncMock(return_value=result)

        with patch.object(
            autonomous_router, "request_idempotency", self.store
        ), patch.object(
            autonomous_router,
            "check_injection",
            AsyncMock(return_value=(False, "")),
        ), patch.object(autonomous_router, "_run_react_loop", run_loop):
            first = self.client.post(
                "/agent/autonomous",
                headers={"Idempotency-Key": "browser-request-1"},
                json={"query": "学RAG", "user_id": "u"},
            )
            second = self.client.post(
                "/agent/autonomous",
                headers={"Idempotency-Key": "browser-request-1"},
                json={"query": "学RAG", "user_id": "u"},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json(), first.json())
        self.assertEqual(run_loop.await_count, 1)

    def test_reused_key_with_changed_request_is_conflict(self):
        run_loop = AsyncMock(return_value=autonomous_router.AutonomousResponse(
            final_answer="done"
        ))
        with patch.object(
            autonomous_router, "request_idempotency", self.store
        ), patch.object(
            autonomous_router,
            "check_injection",
            AsyncMock(return_value=(False, "")),
        ), patch.object(autonomous_router, "_run_react_loop", run_loop):
            first = self.client.post(
                "/agent/autonomous",
                headers={"Idempotency-Key": "browser-request-2"},
                json={"query": "first", "user_id": "u"},
            )
            second = self.client.post(
                "/agent/autonomous",
                headers={"Idempotency-Key": "browser-request-2"},
                json={"query": "changed", "user_id": "u"},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 409)
        self.assertEqual(second.json()["code"], "idempotency_conflict")
        self.assertEqual(second.json()["reason"], "payload_mismatch")
        self.assertEqual(run_loop.await_count, 1)

    def test_invalid_key_is_rejected_before_agent_execution(self):
        run_loop = AsyncMock()
        with patch.object(autonomous_router, "_run_react_loop", run_loop):
            response = self.client.post(
                "/agent/autonomous",
                headers={"Idempotency-Key": "short"},
                json={"query": "学RAG", "user_id": "u"},
            )

        self.assertEqual(response.status_code, 400)
        run_loop.assert_not_awaited()

    def test_continue_injection_check_failure_releases_receipt_for_retry(self):
        conversation_id = "injection-check-retry"
        self._seed_session(conversation_id)
        payload = {
            "conversation_id": conversation_id,
            "user_reply": "第三章",
        }
        headers = {"Idempotency-Key": "browser-continue-retry-1"}
        run_loop = AsyncMock(return_value=autonomous_router.AutonomousResponse(
            final_answer="continued",
            rounds_used=2,
            finalize_reason="explicit_finalize",
        ))

        with patch.object(
            autonomous_router, "request_idempotency", self.store
        ), patch.object(
            autonomous_router, "autonomous_sessions", self.session_store
        ), patch.object(
            autonomous_router,
            "check_injection",
            AsyncMock(
                side_effect=[
                    RuntimeError("classifier unavailable"),
                    (True, "blocked"),
                    (False, ""),
                ]
            ),
        ), patch.object(autonomous_router, "_run_react_loop", run_loop):
            first = self.client.post(
                "/agent/autonomous/continue",
                headers=headers,
                json=payload,
            )
            retry = self.client.post(
                "/agent/autonomous/continue",
                headers=headers,
                json=payload,
            )
            safe_retry = self.client.post(
                "/agent/autonomous/continue",
                headers=headers,
                json={**payload, "user_reply": "安全回答"},
            )

        self.assertEqual(first.status_code, 500)
        self.assertEqual(retry.status_code, 422)
        self.assertIn("安全检查未通过", retry.json()["detail"])
        self.assertEqual(safe_retry.status_code, 200)
        self.assertEqual(safe_retry.json()["final_answer"], "continued")
        self.assertEqual(run_loop.await_count, 1)

    def test_completed_continue_replays_after_session_is_consumed(self):
        conversation_id = "continue-replay-session"
        self._seed_session(conversation_id)
        result = autonomous_router.AutonomousResponse(
            final_answer="continued",
            rounds_used=2,
            finalize_reason="explicit_finalize",
        )
        run_loop = AsyncMock(return_value=result)
        payload = {"conversation_id": conversation_id, "user_reply": "继续"}
        headers = {"Idempotency-Key": "browser-continue-replay-1"}

        with patch.object(
            autonomous_router, "request_idempotency", self.store
        ), patch.object(
            autonomous_router, "autonomous_sessions", self.session_store
        ), patch.object(
            autonomous_router, "check_injection", AsyncMock(return_value=(False, ""))
        ), patch.object(autonomous_router, "_run_react_loop", run_loop):
            first = self.client.post(
                "/agent/autonomous/continue", headers=headers, json=payload
            )

        reopened_receipts = IdempotencyStore(sqlite_path=self.database_path)
        reopened_sessions = AutonomousSessionStore(
            sqlite_path=self.database_path
        )
        with patch.object(
            autonomous_router, "request_idempotency", reopened_receipts
        ), patch.object(
            autonomous_router, "autonomous_sessions", reopened_sessions
        ), patch.object(autonomous_router, "_run_react_loop", run_loop):
            second = self.client.post(
                "/agent/autonomous/continue", headers=headers, json=payload
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json(), first.json())
        self.assertEqual(run_loop.await_count, 1)
        self.assertIsNone(asyncio.run(reopened_sessions.inspect(conversation_id)))

    def test_replayed_pause_response_fails_when_snapshot_is_gone(self):
        request = autonomous_router.AutonomousRequest(query="学RAG", user_id="u")
        key = "browser-expired-pause-1"
        asyncio.run(self.store.begin(
            key,
            "agent.autonomous",
            request.model_dump(mode="json"),
        ))
        asyncio.run(self.store.complete(key, autonomous_router.AutonomousResponse(
            awaiting_user_input=True,
            user_question="继续吗？",
            conversation_id="missing-pause-session",
        ).model_dump(mode="json")))
        run_loop = AsyncMock()

        with patch.object(
            autonomous_router, "request_idempotency", self.store
        ), patch.object(
            autonomous_router, "autonomous_sessions", self.session_store
        ), patch.object(autonomous_router, "_run_react_loop", run_loop):
            response = self.client.post(
                "/agent/autonomous",
                headers={"Idempotency-Key": key},
                json={"query": "学RAG", "user_id": "u"},
            )

        self.assertEqual(response.status_code, 410)
        self.assertIn("暂停会话已过期", response.json()["detail"])
        run_loop.assert_not_awaited()


if __name__ == "__main__":
    unittest.main(verbosity=2)
