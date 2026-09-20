import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from models.knowledge_evidence import KnowledgeRunState


KB_ID = "141a5a02-1748-4b17-8440-464867a47b86"


class EvidenceClient:
    def __init__(self):
        self.epoch = 3
        self.queries = []

    async def validate_scope(self, kb_id, owner_id, revision, epoch):
        if owner_id != "learner" or kb_id != KB_ID or epoch != self.epoch:
            raise HTTPException(409, "scope changed")
        return SimpleNamespace(knowledge_base_id=KB_ID, revision=2, epoch=3, status="ready")

    async def get_scope(self, kb_id, owner_id):
        return await self.validate_scope(kb_id, owner_id, 2, 3)

    async def query(self, kb_id, owner_id, query, revision, epoch):
        await self.validate_scope(kb_id, owner_id, revision, epoch)
        self.queries.append((kb_id, owner_id, query))
        return {
            "scope": {"knowledge_base_id": KB_ID, "revision": 2, "epoch": 3},
            "evidence": [{
                "kind": "kb_chunk", "knowledge_base_id": KB_ID,
                "document_id": "doc-1", "source_version_id": "version-1",
                "chunk_id": "version-1-chunk-0", "title": "Learning",
                "text": "A graph connects concepts across documents.",
                "snippet": "A graph connects concepts across documents.",
            }],
        }


class KnowledgeAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def context(self, client=None):
        from services.knowledge_agent import KnowledgeAgentContext

        return KnowledgeAgentContext(KnowledgeRunState(
            knowledge_base_id=KB_ID, owner_id="learner", revision=2, epoch=3,
            session_id="kbs_test", web_enabled=False,
        ), client=client or EvidenceClient())

    async def test_search_binds_authority_and_only_observed_evidence_can_be_cited(self):
        client = EvidenceClient()
        context = self.context(client)
        payload = json.loads(await context.registry.invoke(
            "search_knowledge_base", {"query": "concepts"}, run_id="r", user_id="learner",
        ))
        self.assertEqual(client.queries, [(KB_ID, "learner", "concepts")])
        evidence_id = payload["evidence"][0]["evidence_id"]
        resolved = context.resolve([evidence_id, "forged-source"])
        self.assertEqual(resolved.invalid_ids, ["forged-source"])
        self.assertEqual(resolved.citations[0].source_version_id, "version-1")
        self.assertEqual(context.registry.list_tools(), ["search_knowledge_base"])

    async def test_model_cannot_replace_the_bound_identity(self):
        client = EvidenceClient()
        context = self.context(client)
        payload = json.loads(await context.registry.invoke(
            "search_knowledge_base", {"query": "x", "owner_id": "other"},
        ))
        self.assertEqual(payload["reason"], "invalid_tool_arguments")
        self.assertEqual(client.queries, [])

    async def test_failed_mutation_epoch_prevents_publication_even_without_new_revision(self):
        client = EvidenceClient()
        context = self.context(client)
        await context.ensure_current()
        client.epoch = 4
        with self.assertRaises(HTTPException) as raised:
            await context.ensure_current()
        self.assertEqual(raised.exception.status_code, 409)

    async def test_web_search_uses_only_user_query_and_rejects_model_supplied_private_text(self):
        from services.knowledge_agent import KnowledgeAgentContext

        context = KnowledgeAgentContext(KnowledgeRunState(
            knowledge_base_id=KB_ID, owner_id="learner", revision=2, epoch=3,
            session_id="kbs_test", web_enabled=True, web_query="公开学习资料",
        ), client=EvidenceClient())
        outbound = AsyncMock(return_value=[{
            "title": "Public", "url": "https://example.org/paper", "snippet": "Summary",
        }])
        with patch("services.knowledge_web.search_web", outbound):
            blocked = json.loads(await context.registry.invoke(
                "search_web", {"query": "PRIVATE_CHUNK secret diagnosis"},
            ))
            self.assertEqual(blocked["reason"], "invalid_tool_arguments")
            self.assertEqual(outbound.await_count, 0)
            await context.registry.invoke("search_web", {})
            outbound.assert_awaited_once_with("公开学习资料", max_results=5)
        with patch("services.knowledge_web.fetch_public_page", new=AsyncMock()) as fetch:
            with self.assertRaises(HTTPException):
                await context.web_fetch("https://example.org/paper?leak=PRIVATE_CHUNK")
            self.assertEqual(fetch.await_count, 0)

    async def test_flagged_evidence_blocks_outbound_tools(self):
        from services.knowledge_agent import KnowledgeAgentContext

        context = KnowledgeAgentContext(KnowledgeRunState(
            knowledge_base_id=KB_ID, owner_id="learner", revision=2, epoch=3,
            session_id="kbs_test", web_enabled=True, web_query="公开资料", outbound_blocked=True,
        ), client=EvidenceClient())
        with patch("services.knowledge_web.search_web", new=AsyncMock()) as search:
            with self.assertRaises(HTTPException):
                await context.web_search()
            self.assertEqual(search.await_count, 0)
