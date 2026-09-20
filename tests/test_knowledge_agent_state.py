"""Knowledge scope and evidence remain bound across an Autonomous pause."""

import unittest

from routers.autonomous import AutonomousRequest
from services.autonomous_snapshot import (
    AutonomousSession, StepRecord, session_from_payload, session_to_payload,
)


KB_ID = "141a5a02-1748-4b17-8440-464867a47b86"


class KnowledgeScopeTests(unittest.TestCase):
    def test_knowledge_request_forces_grounding_and_rejects_mixed_scope(self):
        request = AutonomousRequest(query="Explain", knowledge_base_id=KB_ID)
        self.assertEqual(request.knowledge_base_id, KB_ID)
        self.assertTrue(request.grounding_required)
        with self.assertRaises(ValueError):
            AutonomousRequest(query="Explain", knowledge_base_id=KB_ID, document_id="old.md")
        with self.assertRaises(ValueError):
            AutonomousRequest(query="Explain", web_enabled=True)

    def test_legacy_request_serialization_keeps_existing_idempotency_body(self):
        request = AutonomousRequest(query="Explain", document_id="old.md", user_id="learner")
        self.assertEqual(request.model_dump(mode="json"), {
            "query": "Explain", "document_id": "old.md", "user_id": "learner",
            "grounding_required": False,
        })

    def test_knowledge_pause_preserves_scope_without_becoming_a_legacy_document(self):
        from models.knowledge_evidence import KnowledgeRunState

        state = KnowledgeRunState(
            knowledge_base_id=KB_ID, revision=2, epoch=4, owner_id="learner",
            session_id="kbs_123", web_enabled=False,
        )
        session = AutonomousSession(
            conversation_id="conv_123", user_id="learner", document_id=None,
            messages=[
                {"role": "system", "content": "Grounded knowledge assistant"},
                {"role": "user", "content": "Explain"},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "ask_1", "type": "function", "function": {
                        "name": "ask_user", "arguments": '{"question":"Which topic?"}',
                    },
                }]},
            ],
            plan=[], steps=[StepRecord(round_index=0, tool_name="ask_user")],
            tools_called=[], rounds_used=1, evidence_registry={}, grounding_required=True,
            pending_ask_call_id="ask_1", registry_sha256="a" * 64,
            knowledge_state=state,
        )
        payload = session_to_payload(session)
        restored = session_from_payload("conv_123", payload)
        self.assertEqual(restored.knowledge_state, state)
        self.assertIsNone(restored.document_id)
        self.assertEqual(restored.registry_sha256, "a" * 64)
        payload["schema_version"] = 3
        with self.assertRaises(ValueError):
            session_from_payload("conv_123", payload)

    def test_evidence_cannot_cross_knowledge_or_enable_unrequested_web(self):
        from models.knowledge_evidence import KnowledgeRunState

        state = dict(knowledge_base_id=KB_ID, revision=2, epoch=4,
                     owner_id="learner", session_id="kbs_123", web_enabled=False)
        invalid = {
            "kind": "web_snapshot", "evidence_id": "web_1", "snapshot_id": "snapshot1",
            "title": "Web page", "snippet": "fact", "text": "fact",
            "url": "https://example.org", "content_hash": "a" * 64,
            "fetched_at": "2026-09-20T00:00:00+00:00",
        }
        with self.assertRaises(ValueError):
            KnowledgeRunState(**state, evidence={"web_1": invalid})
