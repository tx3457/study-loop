"""Model usage accounting: provider-reported tokens, attributed to product features."""

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

import services.llm as llm
import services.vectorstore as vectorstore
from services import request_context
from services.request_context import RequestContextMiddleware
from services.usage import UsageLedger, operation_for, usage_ledger


def _chat_response(prompt, completion, content="ok"):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, refusal=None, tool_calls=None),
            finish_reason="stop",
        )],
    )


def _within_request(path, coroutine_factory):
    async def run():
        token = request_context._request_path.set(path)
        try:
            return await coroutine_factory()
        finally:
            request_context._request_path.reset(token)
    return asyncio.run(run())


class TestOperationMapping(unittest.TestCase):
    def test_paths_map_to_product_features(self):
        self.assertEqual(operation_for("/session/start")[0], "quiz")
        self.assertEqual(operation_for("/session/abc/grade")[0], "quiz")
        self.assertEqual(operation_for("/learning-paths/current")[0], "learning_path")
        self.assertEqual(operation_for("/learning-path/notes.md")[0], "learning_path")
        self.assertEqual(operation_for("/agent/autonomous/continue")[0], "autonomous")
        self.assertEqual(operation_for("/documents/upload")[0], "documents")

    def test_a_prefix_only_matches_whole_segments(self):
        # "/session" must not swallow an unrelated path that merely starts with it.
        self.assertEqual(operation_for("/sessions-export")[0], "other")

    def test_work_outside_any_request_is_background(self):
        self.assertEqual(operation_for(None)[0], "background")


class TestUsageLedger(unittest.TestCase):
    def setUp(self):
        self.ledger = UsageLedger()

    def _row(self, key):
        return next(row for row in self.ledger.snapshot()["operations"] if row["key"] == key)

    def test_chat_tokens_are_attributed_to_the_request_feature(self):
        async def record():
            self.ledger.record("chat", _chat_response(120, 30))

        _within_request("/session/start", record)
        row = self._row("quiz")
        self.assertEqual((row["calls"], row["prompt_tokens"], row["completion_tokens"]),
                         (1, 120, 30))
        self.assertEqual(row["chat_tokens"], 150)
        self.assertEqual(row["label"], "答题练习（含批改）")

    def test_missing_usage_is_unknown_not_zero(self):
        self.ledger.record("chat", SimpleNamespace(usage=None))
        row = self._row("background")
        self.assertEqual(row["calls"], 1)
        self.assertEqual(row["calls_without_usage"], 1)
        self.assertEqual(row["chat_tokens"], 0)

    def test_malformed_counts_are_never_added(self):
        for bogus in (-5, True, "12", 3.5):
            self.ledger.record("chat", SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=bogus, completion_tokens=bogus)))
        row = self._row("background")
        self.assertEqual(row["chat_tokens"], 0)
        self.assertEqual(row["calls_without_usage"], 4)

    def test_embedding_tokens_stay_separate_from_chat_tokens(self):
        self.ledger.record("chat", _chat_response(10, 5))
        self.ledger.record("embedding", SimpleNamespace(usage=SimpleNamespace(total_tokens=900)))
        snapshot = self.ledger.snapshot()
        self.assertEqual(snapshot["totals"]["chat_tokens"], 15)
        self.assertEqual(snapshot["totals"]["embedding_tokens"], 900)

    def test_accounting_never_breaks_the_model_call(self):
        class Exploding:
            @property
            def usage(self):
                raise RuntimeError("provider object misbehaved")

        self.ledger.record("chat", Exploding())  # must not raise

    def test_snapshot_states_its_boundaries(self):
        snapshot = self.ledger.snapshot()
        self.assertFalse(snapshot["currency_estimated"])
        self.assertEqual(snapshot["unit"], "tokens")
        self.assertIn("knowledge_service", snapshot["excludes"])
        self.assertTrue(snapshot["since"].endswith("Z"))


class TestFunnelPointsRecord(unittest.TestCase):
    def setUp(self):
        usage_ledger.reset()
        self.addCleanup(usage_ledger.reset)

    def _row(self, key):
        return next(row for row in usage_ledger.snapshot()["operations"] if row["key"] == key)

    def test_llm_chat_records_the_provider_response(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_chat_response(40, 8))
        _within_request("/agent/autonomous", lambda: llm.llm_chat(
            [{"role": "user", "content": "hi"}], client=client, max_retries=0))
        self.assertEqual(self._row("autonomous")["chat_tokens"], 48)

    def test_a_discarded_empty_output_retry_is_still_billed(self):
        # The first response is empty and thrown away, but the provider billed it.
        empty = _chat_response(30, 0, content="")
        full = _chat_response(30, 12, content="answer")
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=[empty, full])
        _within_request("/session/start", lambda: llm.llm_chat(
            [{"role": "user", "content": "q"}], client=client, max_retries=0,
            require_nonempty_response=True))
        row = self._row("quiz")
        self.assertEqual(row["calls"], 2)
        self.assertEqual(row["chat_tokens"], 72)

    def test_llm_parse_records_the_provider_response(self):
        client = MagicMock()
        client.beta.chat.completions.parse = AsyncMock(return_value=_chat_response(55, 20))
        _within_request("/learning-paths", lambda: llm.llm_parse(
            [{"role": "user", "content": "plan"}], dict, client=client, max_retries=0))
        self.assertEqual(self._row("learning_path")["chat_tokens"], 75)

    def test_embeddings_record_under_the_uploading_feature(self):
        response = SimpleNamespace(usage=SimpleNamespace(total_tokens=333), data=[])
        with patch.object(vectorstore.client.embeddings, "create",
                          AsyncMock(return_value=response)):
            _within_request("/documents/upload", lambda: vectorstore._embed(["text"]))
        row = self._row("documents")
        self.assertEqual(row["embedding_tokens"], 333)
        self.assertEqual(row["chat_tokens"], 0)


class TestMiddlewareAttribution(unittest.TestCase):
    """A call made several layers down inside a request lands on that request's feature."""

    def setUp(self):
        usage_ledger.reset()
        self.addCleanup(usage_ledger.reset)

    def test_request_path_is_bound_for_the_whole_request_and_released_after(self):
        app = FastAPI()

        async def deep_helper():
            usage_ledger.record("chat", _chat_response(7, 3))

        @app.post("/session/start")
        async def start():
            await deep_helper()
            return {"ok": True}

        app.add_middleware(RequestContextMiddleware)
        with TestClient(app) as client:
            self.assertEqual(client.post("/session/start").status_code, 200)

        rows = {row["key"]: row for row in usage_ledger.snapshot()["operations"]}
        self.assertEqual(rows["quiz"]["chat_tokens"], 10)
        # The binding does not leak past the request.
        self.assertIsNone(request_context.current_request_path())


class TestUsageEndpoint(unittest.TestCase):
    def test_endpoint_serves_the_snapshot(self):
        from main import app

        usage_ledger.reset()
        self.addCleanup(usage_ledger.reset)
        usage_ledger.record("chat", _chat_response(1, 1))
        with TestClient(app) as client:
            response = client.get("/usage")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["totals"]["chat_tokens"], 2)
        self.assertFalse(payload["currency_estimated"])


if __name__ == "__main__":
    unittest.main()
