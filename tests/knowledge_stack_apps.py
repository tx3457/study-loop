"""Real HTTP test factories. Only external model providers are deterministic fakes.

Run from an isolated temporary cwd with explicit disposable database/provider env.
These factories are never imported by production entry points.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace


def graph_app():
    import graph_service.api as api
    from graph_service.config import Settings
    from graph_service.engine import LightRAGEngine
    from lightrag_contract.support import build_embedding_func, deterministic_llm

    def engine(settings):
        return LightRAGEngine(settings, llm_func=deterministic_llm,
                             embedding_func=build_embedding_func())

    api.LightRAGEngine = engine
    return api.create_app(Settings(
        database_url=os.environ["KNOWLEDGE_DATABASE_URL"],
        internal_token=os.environ["KNOWLEDGE_SERVICE_TOKEN"],
        materials_dir=Path(os.environ["KNOWLEDGE_MATERIALS_DIR"]),
        working_dir=Path(os.environ["KNOWLEDGE_WORKING_DIR"]),
        provider="test", llm_model="deterministic", embedding_model="deterministic",
        embedding_dim=64,
    ))


def backend_app():
    import main
    import services.tool_loop as loop

    async def completion(messages, **kwargs):
        evidence = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            try:
                payload = json.loads(message.get("content", ""))
            except (ValueError, TypeError):
                continue
            evidence.extend(payload.get("evidence", []))
        if evidence:
            name = "finalize"
            arguments = {
                "final_answer": "Shared connects to Anchor in the uploaded material.",
                "citation_ids": [item["evidence_id"] for item in evidence[:2]],
            }
        else:
            name, arguments = "search_knowledge_base", {"query": "Shared Anchor"}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(id=f"test_{name}", function=SimpleNamespace(
                name=name, arguments=json.dumps(arguments),
            ))],
        ))])

    loop.llm_chat = completion
    return main.app
