"""Persistent request idempotency and durable side-effect guards."""

import asyncio
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
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
    _fingerprint,
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
        self.assertIsNotNone(decision.lease)

        response = {"final_answer": "done", "tools_called": []}
        await self.store.complete(decision.lease, response)

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
        decision = await self.store.begin(
            "request-retry", "agent.autonomous", {"query": "q"}
        )
        effect_started = await self.store.abort(decision.lease)
        self.assertFalse(effect_started)

        retry = await self.store.begin(
            "request-retry", "agent.autonomous", {"query": "changed"}
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
        decision = await self.store.begin(
            "request-effect", "agent.autonomous", {"query": "q"}
        )
        await self.store.mark_effect_started(
            decision.lease, "update_learning_profile"
        )
        effect_started = await self.store.abort(decision.lease)
        self.assertTrue(effect_started)

        reopened = IdempotencyStore(sqlite_path=self.path)
        with self.assertRaises(IdempotencyConflictError) as raised:
            await reopened.begin(
                "request-effect", "agent.autonomous", {"query": "q"}
            )
        self.assertEqual(raised.exception.reason, "ambiguous")

    async def test_canonical_outcome_repairs_an_ambiguous_effect_receipt(self):
        operation = "agent.autonomous.continue"
        payload = {"conversation_id": "conv-1", "user_reply": "继续"}
        response = {"final_answer": "canonical", "rounds_used": 2}
        decision = await self.store.begin(
            "request-effect-reconcile", operation, payload
        )
        await self.store.mark_effect_started(
            decision.lease, "update_learning_profile"
        )
        self.assertTrue(await self.store.abort(decision.lease))

        await self.store.reconcile_completed(
            "request-effect-reconcile",
            operation,
            payload,
            response,
            allow_effect_started=True,
        )
        replay = await self.store.begin(
            "request-effect-reconcile", operation, payload
        )
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.response, response)

        with self.assertRaises(IdempotencyConflictError) as mismatch:
            await self.store.reconcile_completed(
                "request-effect-reconcile",
                operation,
                {**payload, "user_reply": "changed"},
                response,
                allow_effect_started=True,
            )
        self.assertEqual(mismatch.exception.reason, "payload_mismatch")

    async def test_expired_clean_claim_is_taken_over_and_old_owner_is_fenced(self):
        now = [100.0]
        first_store = IdempotencyStore(
            sqlite_path=self.path,
            lease_seconds=10,
            clock=lambda: now[0],
        )
        second_store = IdempotencyStore(
            sqlite_path=self.path,
            lease_seconds=10,
            clock=lambda: now[0],
        )
        first = await first_store.begin(
            "request-takeover", "agent.autonomous", {"query": "q"}
        )
        now[0] = 111.0
        second = await second_store.begin(
            "request-takeover", "agent.autonomous", {"query": "q"}
        )

        self.assertNotEqual(first.lease.owner_token, second.lease.owner_token)
        self.assertIsNone(await first_store.renew(first.lease))
        with self.assertRaises(IdempotencyConflictError) as stale_complete:
            await first_store.complete(first.lease, {"final_answer": "stale"})
        self.assertEqual(
            stale_complete.exception.reason,
            "receipt_not_completable",
        )
        with self.assertRaises(IdempotencyConflictError) as stale_effect:
            await first_store.mark_effect_started(
                first.lease,
                "update_learning_profile",
            )
        self.assertEqual(stale_effect.exception.reason, "receipt_not_pending")
        self.assertFalse(
            await first_store.has_effect_started("request-takeover")
        )
        self.assertFalse(await first_store.abort(first.lease))

        with self.assertRaises(IdempotencyConflictError) as still_owned:
            await first_store.begin(
                "request-takeover", "agent.autonomous", {"query": "q"}
            )
        self.assertEqual(still_owned.exception.reason, "in_progress")

        response = {"final_answer": "fresh"}
        await second_store.complete(second.lease, response)
        replay = await first_store.begin(
            "request-takeover", "agent.autonomous", {"query": "q"}
        )
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.response, response)

    async def test_expired_claim_remains_bound_to_original_request(self):
        now = [10.0]
        store = IdempotencyStore(
            sqlite_path=self.path,
            lease_seconds=5,
            clock=lambda: now[0],
        )
        await store.begin(
            "request-expired-binding",
            "agent.autonomous",
            {"query": "first"},
        )
        now[0] = 20.0

        with self.assertRaises(IdempotencyConflictError) as raised:
            await store.begin(
                "request-expired-binding",
                "agent.autonomous",
                {"query": "changed"},
            )
        self.assertEqual(raised.exception.reason, "payload_mismatch")

    async def test_renew_extends_only_a_live_clean_claim(self):
        now = [50.0]
        store = IdempotencyStore(
            sqlite_path=self.path,
            lease_seconds=10,
            clock=lambda: now[0],
        )
        peer = IdempotencyStore(
            sqlite_path=self.path,
            lease_seconds=10,
            clock=lambda: now[0],
        )
        decision = await store.begin(
            "request-renew", "agent.autonomous", {"query": "q"}
        )
        now[0] = 55.0
        renewed = await store.renew(decision.lease)
        self.assertIsNotNone(renewed)
        self.assertEqual(renewed.expires_at, 65.0)
        now[0] = 54.0
        renewed_after_clock_regression = await store.renew(renewed)
        self.assertEqual(renewed_after_clock_regression.expires_at, 65.0)

        now[0] = 61.0
        with self.assertRaises(IdempotencyConflictError) as blocked:
            await peer.begin(
                "request-renew", "agent.autonomous", {"query": "q"}
            )
        self.assertEqual(blocked.exception.reason, "in_progress")

        now[0] = 66.0
        self.assertIsNone(await store.renew(renewed))
        takeover = await peer.begin(
            "request-renew", "agent.autonomous", {"query": "q"}
        )
        self.assertIsNotNone(takeover.lease)

    async def test_effect_started_claim_never_expires_and_can_complete(self):
        now = [1.0]
        store = IdempotencyStore(
            sqlite_path=self.path,
            lease_seconds=5,
            clock=lambda: now[0],
        )
        decision = await store.begin(
            "request-effect-expiry", "agent.autonomous", {"query": "q"}
        )
        await store.mark_effect_started(
            decision.lease,
            "update_learning_profile",
        )
        now[0] = 50.0
        renewed_effect_owner = await store.renew(decision.lease)
        self.assertIsNotNone(renewed_effect_owner)
        self.assertEqual(
            renewed_effect_owner.owner_token,
            decision.lease.owner_token,
        )
        now[0] = 100.0

        with self.assertRaises(IdempotencyConflictError) as raised:
            await store.begin(
                "request-effect-expiry", "agent.autonomous", {"query": "q"}
            )
        self.assertEqual(raised.exception.reason, "ambiguous")

        response = {"final_answer": "effect committed"}
        await store.complete(decision.lease, response)
        replay = await store.begin(
            "request-effect-expiry", "agent.autonomous", {"query": "q"}
        )
        self.assertEqual(replay.response, response)

    async def test_cancelled_old_begin_cannot_delete_a_takeover_claim(self):
        now = [0.0]
        first_store = IdempotencyStore(
            sqlite_path=self.path,
            lease_seconds=10,
            clock=lambda: now[0],
        )
        second_store = IdempotencyStore(
            sqlite_path=self.path,
            lease_seconds=10,
            clock=lambda: now[0],
        )
        committed = threading.Event()
        release_worker = threading.Event()
        original_begin = first_store._begin_sync

        def pause_after_commit(*args):
            decision = original_begin(*args)
            committed.set()
            release_worker.wait(timeout=5)
            return decision

        with patch.object(
            first_store,
            "_begin_sync",
            side_effect=pause_after_commit,
        ):
            old_task = asyncio.create_task(
                first_store.begin(
                    "request-cancelled-old-owner",
                    "agent.autonomous",
                    {"query": "q"},
                )
            )
            self.assertTrue(await asyncio.to_thread(committed.wait, 5))
            now[0] = 11.0
            takeover = await second_store.begin(
                "request-cancelled-old-owner",
                "agent.autonomous",
                {"query": "q"},
            )
            old_task.cancel()
            release_worker.set()
            with self.assertRaises(asyncio.CancelledError):
                await old_task

        with self.assertRaises(IdempotencyConflictError) as raised:
            await first_store.begin(
                "request-cancelled-old-owner",
                "agent.autonomous",
                {"query": "q"},
            )
        self.assertEqual(raised.exception.reason, "in_progress")
        await second_store.complete(takeover.lease, {"final_answer": "done"})

    async def test_cancelled_complete_drains_committed_database_thread(self):
        decision = await self.store.begin(
            "request-cancelled-complete",
            "agent.autonomous",
            {"query": "q"},
        )
        committed = threading.Event()
        release_worker = threading.Event()
        original_complete = self.store._complete_sync

        def pause_after_commit(*args):
            result = original_complete(*args)
            committed.set()
            release_worker.wait(timeout=5)
            return result

        with patch.object(
            self.store,
            "_complete_sync",
            side_effect=pause_after_commit,
        ):
            completion = asyncio.create_task(
                self.store.complete(decision.lease, {"final_answer": "done"})
            )
            self.assertTrue(await asyncio.to_thread(committed.wait, 5))
            completion.cancel()
            await asyncio.sleep(0)
            completion.cancel()
            release_worker.set()
            with self.assertRaises(asyncio.CancelledError):
                await completion

        replay = await self.store.begin(
            "request-cancelled-complete",
            "agent.autonomous",
            {"query": "q"},
        )
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.response, {"final_answer": "done"})

    async def test_legacy_schema_is_migrated_without_reclaiming_pending_rows(self):
        legacy_path = str(Path(self.tmp.name) / "legacy-receipts.sqlite3")
        operation = "agent.autonomous"
        payload = {"query": "legacy"}
        fingerprint = _fingerprint(operation, payload)
        with closing(sqlite3.connect(legacy_path)) as connection:
            connection.execute(
                """
                CREATE TABLE studyloop_idempotency_receipts (
                    idempotency_key TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    state TEXT NOT NULL,
                    response_json TEXT,
                    effect_tool TEXT,
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO studyloop_idempotency_receipts
                    (idempotency_key, operation, request_fingerprint, state,
                     created_at, updated_at)
                VALUES (?, ?, ?, 'pending', 1, 1)
                """,
                ("request-legacy-pending", operation, fingerprint),
            )
            connection.execute(
                """
                INSERT INTO studyloop_idempotency_receipts
                    (idempotency_key, operation, request_fingerprint, state,
                     response_json, created_at, updated_at)
                VALUES (?, ?, ?, 'completed', ?, 1, 1)
                """,
                (
                    "request-legacy-completed",
                    operation,
                    fingerprint,
                    '{"final_answer":"legacy"}',
                ),
            )
            connection.commit()

        migrated = IdempotencyStore(sqlite_path=legacy_path)
        with self.assertRaises(IdempotencyConflictError) as raised:
            await migrated.begin("request-legacy-pending", operation, payload)
        self.assertEqual(raised.exception.reason, "in_progress")
        replay = await migrated.begin(
            "request-legacy-completed",
            operation,
            payload,
        )
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.response, {"final_answer": "legacy"})

        with closing(sqlite3.connect(legacy_path)) as connection:
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(studyloop_idempotency_receipts)"
                ).fetchall()
            }
            state = connection.execute(
                """
                SELECT state FROM studyloop_idempotency_receipts
                WHERE idempotency_key = ?
                """,
                ("request-legacy-pending",),
            ).fetchone()[0]
        self.assertIn("owner_token", columns)
        self.assertIn("recovery_token", columns)
        self.assertIn("lease_expires_at", columns)
        self.assertEqual(state, "pending")

    async def test_non_replayable_tool_marks_receipt_before_handler_runs(self):
        key = "request-tool-effect"
        decision = await self.store.begin(
            key, "agent.autonomous", {"query": "q"}
        )
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
                    idempotency_lease=decision.lease,
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
        inspection = asyncio.run(reopened_sessions.inspect(conversation_id))
        self.assertEqual(inspection.state, "completed")
        self.assertEqual(inspection.outcome, first.json())

    def test_start_does_not_reconcile_effect_receipt_from_unfenced_pause(self):
        key = "browser-start-outcome-repair"
        request = autonomous_router.AutonomousRequest(query="学RAG", user_id="u")
        decision = asyncio.run(self.store.begin(
            key,
            "agent.autonomous",
            request.model_dump(mode="json"),
        ))
        asyncio.run(self.store.mark_effect_started(
            decision.lease,
            "update_learning_profile",
        ))
        conversation_id = autonomous_router._initial_conversation_id(
            decision.lease
        )
        self._seed_session(conversation_id)
        self.assertTrue(asyncio.run(self.store.abort(decision.lease)))
        run_loop = AsyncMock()

        with patch.object(
            autonomous_router, "request_idempotency", self.store
        ), patch.object(
            autonomous_router, "autonomous_sessions", self.session_store
        ), patch.object(autonomous_router, "_run_react_loop", run_loop):
            response = self.client.post(
                "/agent/autonomous",
                headers={"Idempotency-Key": key},
                json=request.model_dump(mode="json"),
            )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["reason"], "ambiguous")
        run_loop.assert_not_awaited()
        with self.assertRaises(IdempotencyConflictError) as replay:
            asyncio.run(self.store.begin(
                key,
                "agent.autonomous",
                request.model_dump(mode="json"),
            ))
        self.assertEqual(replay.exception.reason, "ambiguous")

    def test_start_recovers_pause_when_session_save_commit_ack_is_lost(self):
        key = "browser-start-pause-save-ack-loss"
        request = autonomous_router.AutonomousRequest(query="学RAG", user_id="u")
        original_save = self.session_store.save

        async def commit_then_lose_ack(*args, **kwargs):
            await original_save(*args, **kwargs)
            raise RuntimeError("pause save ACK lost")

        async def pause_once(**kwargs):
            conversation_id = kwargs["initial_conversation_id"]
            messages = list(kwargs["messages"])
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "ask-ack-loss",
                    "type": "function",
                    "function": {
                        "name": "ask_user",
                        "arguments": '{"question":"继续吗？"}',
                    },
                }],
            })
            session = autonomous_router.AutonomousSession(
                conversation_id=conversation_id,
                messages=messages,
                plan=kwargs["plan"],
                steps=[autonomous_router.StepRecord(
                    round_index=0,
                    tool_name="ask_user",
                    tool_args={"question": "继续吗？"},
                )],
                tools_called=[],
                rounds_used=1,
                user_id=kwargs["user_id"],
                document_id=kwargs["document_id"],
                evidence_registry={},
                grounding_required=kwargs["grounding_required"],
                pending_ask_call_id="ask-ack-loss",
            )
            response = autonomous_router.AutonomousResponse(
                awaiting_user_input=True,
                user_question="继续吗？",
                conversation_id=conversation_id,
                rounds_used=1,
            )
            return await kwargs["pause_session_saver"](session, response)

        run_loop = AsyncMock(side_effect=pause_once)
        with patch.object(
            autonomous_router, "request_idempotency", self.store
        ), patch.object(
            autonomous_router, "autonomous_sessions", self.session_store
        ), patch.object(
            autonomous_router, "check_injection", AsyncMock(return_value=(False, ""))
        ), patch.object(
            autonomous_router, "_run_react_loop", run_loop
        ), patch.object(
            self.session_store, "save", side_effect=commit_then_lose_ack
        ):
            response = self.client.post(
                "/agent/autonomous",
                headers={"Idempotency-Key": key},
                json=request.model_dump(mode="json"),
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["awaiting_user_input"])
        self.assertEqual(run_loop.await_count, 1)
        replay = asyncio.run(self.store.begin(
            key, "agent.autonomous", request.model_dump(mode="json")
        ))
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.response, response.json())

    def test_cancel_after_pause_commit_preserves_canonical_replay(self):
        key = "browser-start-pause-save-cancel"
        request = autonomous_router.AutonomousRequest(query="学RAG", user_id="u")
        original_save = self.session_store.save

        async def commit_then_cancel(*args, **kwargs):
            await original_save(*args, **kwargs)
            raise asyncio.CancelledError

        async def pause_then_cancel(**kwargs):
            conversation_id = kwargs["initial_conversation_id"]
            messages = [*kwargs["messages"], {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "ask-cancel",
                    "type": "function",
                    "function": {
                        "name": "ask_user",
                        "arguments": '{"question":"继续吗？"}',
                    },
                }],
            }]
            session = autonomous_router.AutonomousSession(
                conversation_id=conversation_id,
                messages=messages,
                plan=kwargs["plan"],
                steps=[autonomous_router.StepRecord(
                    round_index=0,
                    tool_name="ask_user",
                    tool_args={"question": "继续吗？"},
                )],
                tools_called=[],
                rounds_used=1,
                user_id=kwargs["user_id"],
                document_id=kwargs["document_id"],
                evidence_registry={},
                grounding_required=kwargs["grounding_required"],
                pending_ask_call_id="ask-cancel",
            )
            response = autonomous_router.AutonomousResponse(
                awaiting_user_input=True,
                user_question="继续吗？",
                conversation_id=conversation_id,
                rounds_used=1,
            )
            return await kwargs["pause_session_saver"](session, response)

        async def scenario():
            run_loop = AsyncMock(side_effect=pause_then_cancel)
            with patch.object(
                autonomous_router, "request_idempotency", self.store
            ), patch.object(
                autonomous_router, "autonomous_sessions", self.session_store
            ), patch.object(
                autonomous_router,
                "check_injection",
                AsyncMock(return_value=(False, "")),
            ), patch.object(
                autonomous_router, "_run_react_loop", run_loop
            ), patch.object(
                self.session_store, "save", side_effect=commit_then_cancel
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await autonomous_router.autonomous_agent(request, key)
            replay_loop = AsyncMock()
            with patch.object(
                autonomous_router, "request_idempotency", self.store
            ), patch.object(
                autonomous_router, "autonomous_sessions", self.session_store
            ), patch.object(
                autonomous_router, "_run_react_loop", replay_loop
            ):
                replayed = await autonomous_router.autonomous_agent(request, key)
            return run_loop, replay_loop, replayed

        run_loop, replay_loop, replayed = asyncio.run(scenario())
        self.assertEqual(run_loop.await_count, 1)
        replay_loop.assert_not_awaited()
        self.assertTrue(replayed.awaiting_user_input)

    def test_expired_clean_start_recovers_pause_without_rerunning_agent(self):
        now = [100.0]
        self.store = IdempotencyStore(
            sqlite_path=self.database_path,
            lease_seconds=5,
            clock=lambda: now[0],
        )
        key = "weak-key-but-valid"
        request = autonomous_router.AutonomousRequest(query="学RAG", user_id="u")
        decision = asyncio.run(self.store.begin(
            key,
            "agent.autonomous",
            request.model_dump(mode="json"),
        ))
        conversation_id = autonomous_router._initial_conversation_id(
            decision.lease
        )
        old_key_digest = __import__("hashlib").sha256(
            f"agent.autonomous:{key}".encode("utf-8")
        ).hexdigest()
        self.assertNotEqual(conversation_id, f"conv_{old_key_digest}")
        self._seed_session(conversation_id)
        now[0] = 106.0
        run_loop = AsyncMock()

        with patch.object(
            autonomous_router, "request_idempotency", self.store
        ), patch.object(
            autonomous_router, "autonomous_sessions", self.session_store
        ), patch.object(autonomous_router, "_run_react_loop", run_loop):
            response = self.client.post(
                "/agent/autonomous",
                headers={"Idempotency-Key": key},
                json=request.model_dump(mode="json"),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["conversation_id"], conversation_id)
        run_loop.assert_not_awaited()
        replay = asyncio.run(self.store.begin(
            key, "agent.autonomous", request.model_dump(mode="json")
        ))
        self.assertTrue(replay.replayed)

    def test_corrupt_initial_pause_recovery_returns_generic_gone(self):
        key = "browser-corrupt-initial-pause"
        request = autonomous_router.AutonomousRequest(query="学RAG", user_id="u")
        decision = asyncio.run(self.store.begin(
            key,
            "agent.autonomous",
            request.model_dump(mode="json"),
        ))
        conversation_id = autonomous_router._initial_conversation_id(
            decision.lease
        )
        self._seed_session(conversation_id)
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                "UPDATE studyloop_autonomous_sessions SET payload_json = ? "
                "WHERE conversation_id = ?",
                ('{"schema_version":2}', conversation_id),
            )
            connection.commit()

        with patch.object(
            autonomous_router, "request_idempotency", self.store
        ), patch.object(
            autonomous_router, "autonomous_sessions", self.session_store
        ):
            response = self.client.post(
                "/agent/autonomous",
                headers={"Idempotency-Key": key},
                json=request.model_dump(mode="json"),
            )

        self.assertEqual(response.status_code, 410)
        self.assertEqual(response.json(), {
            "detail": "暂停会话数据无效；请重新开始"
        })

    def test_continue_ownership_loss_keeps_exact_request_retryable(self):
        now = [100.0]
        self.store = IdempotencyStore(
            sqlite_path=self.database_path,
            lease_seconds=5,
            clock=lambda: now[0],
        )
        self.session_store = AutonomousSessionStore(
            sqlite_path=self.database_path,
            ttl_seconds=60,
            operation_lease_seconds=5,
            clock=lambda: now[0],
        )
        conversation_id = "continue-clean-lease-loss"
        self._seed_session(conversation_id)
        payload = {"conversation_id": conversation_id, "user_reply": "继续"}
        headers = {"Idempotency-Key": "continue-clean-lease-loss-key"}
        result = autonomous_router.AutonomousResponse(
            final_answer="安全重试完成",
            rounds_used=2,
            finalize_reason="explicit_finalize",
        )

        async def expire_before_round(**kwargs):
            now[0] = 106.0
            await kwargs["on_before_round"]()

        attempts = 0

        async def dispatch_attempt(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return await expire_before_round(**kwargs)
            return result

        run_loop = AsyncMock(side_effect=dispatch_attempt)
        with patch.object(
            autonomous_router, "request_idempotency", self.store
        ), patch.object(
            autonomous_router, "autonomous_sessions", self.session_store
        ), patch.object(
            autonomous_router, "check_injection", AsyncMock(return_value=(False, ""))
        ), patch.object(autonomous_router, "_run_react_loop", run_loop):
            lost = self.client.post(
                "/agent/autonomous/continue", headers=headers, json=payload
            )
            retried = self.client.post(
                "/agent/autonomous/continue", headers=headers, json=payload
            )

        self.assertEqual(lost.status_code, 409)
        self.assertEqual(lost.json()["reason"], "in_progress")
        self.assertEqual(retried.status_code, 200)
        self.assertEqual(retried.json()["final_answer"], "安全重试完成")
        self.assertEqual(run_loop.await_count, 2)

    def test_cancel_endpoint_cancels_only_a_paused_session(self):
        paused_id = "cancel-http-paused"
        self._seed_session(paused_id)
        running_id = "cancel-http-running"
        self._seed_session(running_id)
        running = asyncio.run(self.session_store.claim(
            running_id, autonomous_router._request_fingerprint("continue", {"x": 1})
        ))

        with patch.object(
            autonomous_router, "autonomous_sessions", self.session_store
        ):
            canceled = self.client.delete(f"/agent/autonomous/{paused_id}")
            repeated = self.client.delete(f"/agent/autonomous/{paused_id}")
            busy = self.client.delete(f"/agent/autonomous/{running_id}")

        self.assertEqual(canceled.status_code, 200)
        self.assertEqual(canceled.json(), {"status": "canceled"})
        self.assertEqual(repeated.status_code, 200)
        self.assertEqual(repeated.json(), {"status": "missing"})
        self.assertEqual(busy.status_code, 409)
        self.assertEqual(
            asyncio.run(self.session_store.inspect(running_id)).state,
            "in_flight",
        )
        self.assertTrue(asyncio.run(self.session_store.consume(
            running_id, running.claim_token
        )))

    def test_continue_repairs_effect_receipt_from_durable_terminal_outcome(self):
        conversation_id = "continue-effect-outcome-repair"
        self._seed_session(conversation_id)
        payload = {"conversation_id": conversation_id, "user_reply": "继续"}
        key = "browser-continue-outcome-repair"
        decision = asyncio.run(self.store.begin(
            key,
            "agent.autonomous.continue",
            payload,
        ))
        asyncio.run(self.store.mark_effect_started(
            decision.lease,
            "update_learning_profile",
        ))
        fingerprint = autonomous_router._request_fingerprint(
            "agent.autonomous.continue", payload
        )
        claim = asyncio.run(self.session_store.claim(
            conversation_id, fingerprint
        ))
        outcome = autonomous_router.AutonomousResponse(
            final_answer="已完成",
            rounds_used=2,
            finalize_reason="explicit_finalize",
        ).model_dump(mode="json")
        self.assertTrue(asyncio.run(self.session_store.finish(
            conversation_id,
            claim.claim_token,
            outcome,
        )))
        self.assertTrue(asyncio.run(self.store.abort(decision.lease)))
        run_loop = AsyncMock()

        with patch.object(
            autonomous_router, "request_idempotency", self.store
        ), patch.object(
            autonomous_router, "autonomous_sessions", self.session_store
        ), patch.object(autonomous_router, "_run_react_loop", run_loop):
            response = self.client.post(
                "/agent/autonomous/continue",
                headers={"Idempotency-Key": key},
                json=payload,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), outcome)
        run_loop.assert_not_awaited()
        replay = asyncio.run(self.store.begin(
            key,
            "agent.autonomous.continue",
            payload,
        ))
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.response, outcome)

    def test_continue_returns_canonical_finish_after_commit_ack_is_lost(self):
        conversation_id = "continue-finish-ack-loss"
        self._seed_session(conversation_id)
        payload = {"conversation_id": conversation_id, "user_reply": "继续"}
        key = "browser-continue-finish-ack-loss"
        result = autonomous_router.AutonomousResponse(
            final_answer="规范终态",
            rounds_used=2,
            finalize_reason="explicit_finalize",
        )
        run_loop = AsyncMock(return_value=result)
        original_finish = self.session_store.finish

        async def commit_then_lose_ack(*args, **kwargs):
            self.assertTrue(await original_finish(*args, **kwargs))
            raise RuntimeError("finish ACK lost")

        with patch.object(
            autonomous_router, "request_idempotency", self.store
        ), patch.object(
            autonomous_router, "autonomous_sessions", self.session_store
        ), patch.object(
            autonomous_router, "check_injection", AsyncMock(return_value=(False, ""))
        ), patch.object(
            autonomous_router, "_run_react_loop", run_loop
        ), patch.object(
            self.session_store, "finish", side_effect=commit_then_lose_ack
        ):
            response = self.client.post(
                "/agent/autonomous/continue",
                headers={"Idempotency-Key": key},
                json=payload,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), result.model_dump(mode="json"))
        self.assertEqual(run_loop.await_count, 1)
        replay = asyncio.run(self.store.begin(
            key, "agent.autonomous.continue", payload
        ))
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.response, response.json())

    def test_continue_returns_canonical_handoff_after_commit_ack_is_lost(self):
        conversation_id = "continue-handoff-ack-loss"
        self._seed_session(conversation_id)
        payload = {"conversation_id": conversation_id, "user_reply": "继续"}
        key = "browser-continue-handoff-ack-loss"
        next_id = "continue-handoff-ack-loss-next"
        original_handoff = self.session_store.handoff

        async def commit_then_lose_ack(*args, **kwargs):
            self.assertTrue(await original_handoff(*args, **kwargs))
            raise RuntimeError("handoff ACK lost")

        async def pause_again(**kwargs):
            next_messages = list(kwargs["messages"])
            next_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "ask-2",
                    "type": "function",
                    "function": {
                        "name": "ask_user",
                        "arguments": '{"question":"下一章？"}',
                    },
                }],
            })
            next_session = autonomous_router.AutonomousSession(
                conversation_id=next_id,
                messages=next_messages,
                plan=kwargs["plan"],
                steps=[*kwargs["steps"], autonomous_router.StepRecord(
                    round_index=1,
                    tool_name="ask_user",
                    tool_args={"question": "下一章？"},
                )],
                tools_called=kwargs["tools_called"],
                rounds_used=2,
                user_id=kwargs["user_id"],
                document_id=kwargs["document_id"],
                evidence_registry=kwargs["evidence_registry"],
                grounding_required=kwargs["grounding_required"],
                pending_ask_call_id="ask-2",
            )
            response = autonomous_router.AutonomousResponse(
                awaiting_user_input=True,
                user_question="下一章？",
                conversation_id=next_id,
                rounds_used=2,
            )
            await kwargs["pause_session_saver"](next_session, response)
            return response

        run_loop = AsyncMock(side_effect=pause_again)
        with patch.object(
            autonomous_router, "request_idempotency", self.store
        ), patch.object(
            autonomous_router, "autonomous_sessions", self.session_store
        ), patch.object(
            autonomous_router, "check_injection", AsyncMock(return_value=(False, ""))
        ), patch.object(
            autonomous_router, "_run_react_loop", run_loop
        ), patch.object(
            self.session_store, "handoff", side_effect=commit_then_lose_ack
        ):
            response = self.client.post(
                "/agent/autonomous/continue",
                headers={"Idempotency-Key": key},
                json=payload,
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["awaiting_user_input"])
        self.assertEqual(response.json()["conversation_id"], next_id)
        self.assertEqual(run_loop.await_count, 1)
        next_inspection = asyncio.run(self.session_store.inspect(next_id))
        self.assertEqual(next_inspection.state, "paused")
        replay = asyncio.run(self.store.begin(
            key, "agent.autonomous.continue", payload
        ))
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.response, response.json())

    def test_replayed_pause_response_fails_when_snapshot_is_gone(self):
        request = autonomous_router.AutonomousRequest(query="学RAG", user_id="u")
        key = "browser-expired-pause-1"
        decision = asyncio.run(self.store.begin(
            key,
            "agent.autonomous",
            request.model_dump(mode="json"),
        ))
        asyncio.run(self.store.complete(decision.lease, autonomous_router.AutonomousResponse(
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
