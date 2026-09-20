"""Per-request tools must not mutate or borrow the legacy global authority."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from services.autonomous_snapshot import current_registry_sha256
from services.tool_loop import run_tool_round
from services.tool_registry import EffectMode, Tool, ToolMetadata, ToolRegistry, tool_registry


def response(name, arguments):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(
            id="scoped_call",
            function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
        )],
    ))])


def probe_tool(name="scoped_probe"):
    async def handler(query: str):
        return json.dumps({"answer": f"scoped:{query}", "injection_flagged": True})
    return Tool(
        name=name,
        description="Scope test",
        parameters_schema={
            "type": "object", "properties": {"query": {"type": "string"}},
            "required": ["query"], "additionalProperties": False,
        },
        handler=handler,
        metadata=ToolMetadata(effect_mode=EffectMode.READ_ONLY, max_retries=0),
    )


class ScopedRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_scoped_registration_preserves_legacy_contract_and_other_scopes(self):
        before = current_registry_sha256()
        registry = ToolRegistry.isolated()
        other = ToolRegistry.isolated()
        registry.register(probe_tool())
        self.assertFalse(other.has("scoped_probe"))
        self.assertFalse(tool_registry.has("scoped_probe"))
        self.assertEqual(current_registry_sha256(), before)
        self.assertNotEqual(current_registry_sha256(registry), before)

    async def test_scoped_round_invokes_real_scoped_handler_and_audits_locally(self):
        registry = ToolRegistry.isolated()
        registry.register(probe_tool())
        messages = [{"role": "user", "content": "test"}]
        with patch("services.tool_loop.llm_chat", new=AsyncMock(
            return_value=response("scoped_probe", {"query": "abc"}),
        )):
            result = await run_tool_round(
                messages, tools=registry.get_openai_schemas(),
                registry=registry, run_id="scoped_run", user_id="owner",
            )
        self.assertEqual(result.outcomes[0].kind, "dispatched")
        self.assertEqual(json.loads(messages[-1]["content"])["answer"], "scoped:abc")
        self.assertTrue(registry.get_audit(run_id="scoped_run"))
        self.assertEqual(tool_registry.get_audit(run_id="scoped_run"), [])
        self.assertTrue(registry.has_untrusted_content("scoped_run"))
        self.assertFalse(tool_registry.has_untrusted_content("scoped_run"))

    async def test_scoped_round_cannot_borrow_a_global_tool(self):
        registry = ToolRegistry.isolated()
        messages = [{"role": "user", "content": "test"}]
        with patch("services.tool_loop.llm_chat", new=AsyncMock(
            return_value=response("search_document", {"query": "x"}),
        )):
            result = await run_tool_round(messages, tools=[], registry=registry)
        self.assertEqual(result.outcomes[0].blocked_reason, "not_in_whitelist")
