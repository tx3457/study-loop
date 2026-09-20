import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

import routers.autonomous as au
from models.knowledge_evidence import KnowledgeRunState
from services.knowledge_agent import KnowledgeAgentContext
from services.autonomous_sessions import AutonomousSessionStore
from test_knowledge_agent import EvidenceClient, KB_ID


def completion(name, arguments, call_id):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=None, tool_calls=[SimpleNamespace(
            id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
        )],
    ))])


class KnowledgeAutonomousTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def context(self):
        return KnowledgeAgentContext(KnowledgeRunState(
            knowledge_base_id=KB_ID, revision=2, epoch=3, owner_id="learner",
            session_id="kbs_test", web_enabled=False,
        ), client=EvidenceClient())

    async def run_context(self, context, *, forged=False, mutate=False):
        rounds = 0

        async def provider(*args, **kwargs):
            nonlocal rounds
            rounds += 1
            if rounds == 1:
                return completion("search_knowledge_base", {"query": "concepts"}, "search1")
            if mutate:
                context.client.epoch += 1
            evidence_id = "forged" if forged else next(iter(context.state.evidence))
            return completion("finalize", {
                "final_answer": "Graphs connect concepts.", "citation_ids": [evidence_id],
            }, "final1")

        with patch("services.tool_loop.llm_chat", side_effect=provider):
            return await au._run_react_loop(
                messages=[{"role": "system", "content": "Use evidence"},
                          {"role": "user", "content": "Explain concepts"}],
                plan=[], steps=[], tools_called=[], user_id="learner", document_id=None,
                starting_round=0, run_id="kb_test", evidence_registry={},
                grounding_required=True, knowledge_context=context,
            )

    async def test_knowledge_answer_uses_observed_sources_without_legacy_citations(self):
        result = await self.run_context(self.context())
        self.assertFalse(result.abstained)
        self.assertEqual(result.knowledge_base_id, KB_ID)
        self.assertEqual(result.citations, [])
        self.assertEqual(result.source_citations[0].source_version_id, "version-1")
        self.assertEqual(result.tools_called, ["search_knowledge_base"])
        self.assertEqual(result.steps[0].tool_name, "search_knowledge_base")

    async def test_forged_source_abstains(self):
        result = await self.run_context(self.context(), forged=True)
        self.assertTrue(result.abstained)
        self.assertEqual(result.source_citations, [])

    async def test_mid_generation_mutation_cannot_publish_an_old_scope(self):
        with self.assertRaises(HTTPException) as raised:
            await self.run_context(self.context(), mutate=True)
        self.assertEqual(raised.exception.status_code, 409)

    async def test_public_knowledge_pause_and_continue_preserve_observed_evidence(self):
        client = EvidenceClient()
        calls = 0
        evidence_id = None

        async def provider(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return completion("search_knowledge_base", {"query": "concepts"}, "search1")
            if calls == 2:
                return completion("ask_user", {"question": "需要例子吗？"}, "ask1")
            return completion("finalize", {
                "final_answer": "Graphs connect concepts.", "citation_ids": [evidence_id],
            }, "final1")

        with tempfile.TemporaryDirectory() as directory:
            store = AutonomousSessionStore(sqlite_path=str(Path(directory) / "sessions.db"))
            with (
                patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}),
                patch("services.knowledge_client.knowledge_client", client),
                patch.object(au, "autonomous_sessions", store),
                patch("services.tool_loop.llm_chat", side_effect=provider),
            ):
                pending = await au.autonomous_agent(
                    au.AutonomousRequest(query="Explain", knowledge_base_id=KB_ID),
                    idempotency_key=None, subject="learner",
                )
                self.assertTrue(pending.awaiting_user_input)
                inspection = await store.inspect(pending.conversation_id)
                evidence_id = next(iter(inspection.payload["knowledge_state"]["evidence"]))
                self.assertEqual(inspection.payload["schema_version"], 4)
                replay = await au._validate_replayed_response(pending.model_dump(mode="json"))
                self.assertEqual(replay.knowledge_base_id, KB_ID)
                result = await au.continue_autonomous(au.ContinueRequest(
                    conversation_id=pending.conversation_id, user_reply="需要",
                ), idempotency_key=None)
                self.assertFalse(result.abstained)
                self.assertEqual(result.source_citations[0].evidence_id, evidence_id)
                terminal = await au._validate_replayed_response(result.model_dump(mode="json"))
                self.assertEqual(terminal.final_answer, result.final_answer)
