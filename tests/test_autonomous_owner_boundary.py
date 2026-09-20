"""HTTP owners must be checked before leases, cancellation, or receipt replay."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

import routers.autonomous as au
from services.autonomous_sessions import AutonomousSessionStore


class AutonomousOwnerBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = AutonomousSessionStore(sqlite_path=str(Path(self.directory.name) / "sessions.db"))
        self.session = au.AutonomousSession(
            conversation_id="conv_alice", user_id="alice", document_id=None,
            messages=[
                {"role": "system", "content": "Test assistant"},
                {"role": "user", "content": "Explain"},
                {"role": "assistant", "tool_calls": [{"id": "ask1", "type": "function",
                    "function": {"name": "ask_user", "arguments": '{"question":"Which topic?"}'}}]},
            ], plan=[], steps=[au.StepRecord(round_index=0, tool_name="ask_user")],
            tools_called=[], rounds_used=1, evidence_registry={}, grounding_required=False,
            pending_ask_call_id="ask1", registry_sha256=au._current_registry_sha256(),
        )
        await self.store.save("conv_alice", au._session_to_payload(self.session))
        self.original = await self.store.inspect("conv_alice")

    async def test_other_owner_cannot_continue_or_even_claim_a_receipt(self):
        with (patch.object(au, "autonomous_sessions", self.store),
              patch.object(au.request_idempotency, "begin", new=AsyncMock()) as begin):
            with self.assertRaises(HTTPException) as raised:
                await au.continue_autonomous(au.ContinueRequest(
                    conversation_id="conv_alice", user_reply="private",
                ), idempotency_key="owner-boundary-key", subject="bob")
            self.assertEqual(raised.exception.status_code, 404)
            self.assertEqual(begin.await_count, 0)
        self.assertEqual(await self.store.inspect("conv_alice"), self.original)

    async def test_other_owner_cancel_is_an_opaque_noop_and_owner_can_cancel(self):
        with patch.object(au, "autonomous_sessions", self.store):
            result = await au.cancel_autonomous_session("conv_alice", subject="bob")
            self.assertEqual(result, {"status": "missing"})
            self.assertEqual(await self.store.inspect("conv_alice"), self.original)
            own_result = await au.cancel_autonomous_session("conv_alice", subject="alice")
            self.assertEqual(own_result, {"status": "canceled"})

    async def test_completed_recovery_cannot_disclose_other_owner_outcome(self):
        req = au.ContinueRequest(conversation_id="conv_alice", user_reply="private")
        fingerprint = au.request_fingerprint("agent.autonomous.continue", req.model_dump(mode="json"))
        claim = await self.store.claim("conv_alice", fingerprint)
        await self.store.finish("conv_alice", claim.claim_token, au.AutonomousResponse(
            final_answer="Alice-only answer", rounds_used=2,
        ).model_dump(mode="json"))
        before = await self.store.inspect("conv_alice")
        with patch.object(au, "autonomous_sessions", self.store):
            with self.assertRaises(HTTPException) as raised:
                await au.continue_autonomous(req, idempotency_key=None, subject="bob")
            self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(await self.store.inspect("conv_alice"), before)
