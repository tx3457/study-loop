from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("lightrag")

from graph_service.config import Settings
from graph_service.engine import LightRAGEngine
from graph_service.errors import TooLarge


def provider_settings(tmp_path) -> Settings:
    return Settings(
        database_url="postgresql://graph@127.0.0.1:5432/graph",
        internal_token="test-token",
        materials_dir=tmp_path / "materials",
        working_dir=tmp_path / "workspaces",
        provider="openai",
        llm_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        embedding_model="BAAI/bge-m3",
        embedding_dim=1024,
        llm_api_key="test-key",
        llm_base_url="https://api.siliconflow.cn/v1",
        embedding_api_key="test-key",
        embedding_base_url="https://api.siliconflow.cn/v1",
        llm_enable_thinking=False,
        llm_temperature=0.7,
        llm_top_p=0.8,
        llm_top_k=20,
        llm_min_p=0.0,
    )


@pytest.mark.asyncio
async def test_embedding_guard_rejects_oversize_normalized_utf8_before_provider(
    tmp_path,
) -> None:
    engine = LightRAGEngine(replace(provider_settings(tmp_path), embedding_max_utf8_bytes=8))
    _, embedding = engine._provider_callbacks()

    with pytest.raises(TooLarge, match="embedding input"):
        await embedding(["汉字汉"])


@pytest.mark.asyncio
async def test_legacy_bge_large_profile_rejects_oversize_input_before_provider(
    tmp_path, monkeypatch
) -> None:
    import lightrag.llm.openai as sdk_openai

    called = False

    async def fake_embed(*args, **kwargs):
        nonlocal called
        called = True
        return np.zeros((1, 1024), dtype=np.float32)

    monkeypatch.setattr(sdk_openai.openai_embed, "func", fake_embed)
    engine = LightRAGEngine(
        replace(
            provider_settings(tmp_path),
            embedding_model="BAAI/bge-large-zh-v1.5",
            embedding_max_input_tokens=512,
            embedding_token_budget=512,
            embedding_max_utf8_bytes=480,
        )
    )
    _, embedding = engine._provider_callbacks()

    with pytest.raises(TooLarge, match="480-byte"):
        await embedding(["x" * 481])
    assert called is False


@pytest.mark.asyncio
async def test_embedding_callback_disables_provider_truncation_and_transport_retries(
    tmp_path, monkeypatch
) -> None:
    import lightrag.llm.openai as sdk_openai

    calls: list[dict] = []

    async def fake_embed(texts, **kwargs):
        calls.append({"texts": texts, **kwargs})
        return np.zeros((len(texts), 1024), dtype=np.float32)

    monkeypatch.setattr(sdk_openai.openai_embed, "func", fake_embed)
    engine = LightRAGEngine(replace(provider_settings(tmp_path), embedding_max_utf8_bytes=3))
    _, embedding = engine._provider_callbacks()

    vectors = await embedding(["ＡＡＡ"], context="query")

    assert vectors.shape == (1, 1024)
    assert calls == [
        {
            "texts": ["AAA"],
            "model": "BAAI/bge-m3",
            "api_key": "test-key",
            "base_url": "https://api.siliconflow.cn/v1",
            "max_token_size": 0,
            "client_configs": {"timeout": 60, "max_retries": 0},
            "context": "query",
        }
    ]


@pytest.mark.asyncio
async def test_llm_callback_enforces_generation_policy_and_forwards_protocol_kwargs(
    tmp_path, monkeypatch
) -> None:
    import lightrag.llm.openai as sdk_openai

    calls: list[dict] = []

    async def fake_complete(model, prompt, **kwargs):
        calls.append({"model": model, "prompt": prompt, **kwargs})
        return "complete"

    monkeypatch.setattr(sdk_openai, "openai_complete_if_cache", fake_complete)
    engine = LightRAGEngine(provider_settings(tmp_path))
    llm, _ = engine._provider_callbacks()

    result = await llm(
        "extract",
        system_prompt="system",
        response_format={"type": "json_object"},
        hashing_kv={"cache": "protocol"},
    )

    assert result == "complete"
    assert calls == [
        {
            "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "prompt": "extract",
            "system_prompt": "system",
            "response_format": {"type": "json_object"},
            "hashing_kv": {"cache": "protocol"},
            "api_key": "test-key",
            "base_url": "https://api.siliconflow.cn/v1",
            "timeout": 180,
            "max_tokens": 8192,
            "temperature": 0.7,
            "top_p": 0.8,
            "extra_body": {"enable_thinking": False, "top_k": 20, "min_p": 0.0},
            "openai_client_configs": {"max_retries": 0},
        }
    ]


@pytest.mark.asyncio
async def test_llm_callback_preserves_lower_per_call_output_limit(tmp_path, monkeypatch) -> None:
    import lightrag.llm.openai as sdk_openai

    calls: list[dict] = []

    async def fake_complete(model, prompt, **kwargs):
        calls.append(kwargs)
        return "complete"

    monkeypatch.setattr(sdk_openai, "openai_complete_if_cache", fake_complete)
    engine = LightRAGEngine(provider_settings(tmp_path))
    llm, _ = engine._provider_callbacks()

    await llm("summarize", max_tokens=1200)
    await llm("extract", max_tokens=10000)

    assert [call["max_tokens"] for call in calls] == [1200, 8192]


@pytest.mark.asyncio
async def test_llm_callback_rejects_length_truncated_output(tmp_path, monkeypatch) -> None:
    import lightrag.llm.openai as sdk_openai
    from lightrag.utils import TruncatedResponse

    async def fake_complete(*args, **kwargs):
        return TruncatedResponse('{"partial":')

    monkeypatch.setattr(sdk_openai, "openai_complete_if_cache", fake_complete)
    engine = LightRAGEngine(provider_settings(tmp_path))
    llm, _ = engine._provider_callbacks()

    with pytest.raises(RuntimeError, match="truncated"):
        await llm("extract")


@pytest.mark.asyncio
async def test_llm_callback_omits_thinking_option_for_legacy_profile(tmp_path, monkeypatch) -> None:
    import lightrag.llm.openai as sdk_openai

    calls: list[dict] = []

    async def fake_complete(model, prompt, **kwargs):
        calls.append(kwargs)
        return "complete"

    monkeypatch.setattr(sdk_openai, "openai_complete_if_cache", fake_complete)
    engine = LightRAGEngine(
        replace(
            provider_settings(tmp_path),
            llm_enable_thinking=None,
            llm_temperature=0.0,
            llm_top_p=None,
            llm_top_k=None,
            llm_min_p=None,
            llm_presence_penalty=None,
        )
    )
    llm, _ = engine._provider_callbacks()

    await llm("legacy")

    assert "extra_body" not in calls[0]
    assert calls[0]["temperature"] == 0.0
    assert "top_p" not in calls[0]
    assert "presence_penalty" not in calls[0]


@pytest.mark.asyncio
async def test_qwen35_callback_forwards_presence_penalty(tmp_path, monkeypatch) -> None:
    import lightrag.llm.openai as sdk_openai

    calls: list[dict] = []

    async def fake_complete(model, prompt, **kwargs):
        calls.append(kwargs)
        return "complete"

    monkeypatch.setattr(sdk_openai, "openai_complete_if_cache", fake_complete)
    engine = LightRAGEngine(
        replace(
            provider_settings(tmp_path),
            llm_model="Qwen/Qwen3.5-35B-A3B",
            llm_presence_penalty=1.5,
        )
    )
    llm, _ = engine._provider_callbacks()

    await llm("extract", presence_penalty=-1.0)

    assert calls[0]["presence_penalty"] == 1.5


@pytest.mark.asyncio
async def test_query_rejects_oversize_embedding_input_before_storage_access(tmp_path) -> None:
    engine = LightRAGEngine(replace(provider_settings(tmp_path), embedding_max_utf8_bytes=8))

    with pytest.raises(TooLarge, match="query"):
        await engine.query("workspace", "汉字汉")


@pytest.mark.asyncio
async def test_lightrag_receives_chunk_concurrency_and_sdk_timeout_settings(
    tmp_path, monkeypatch
) -> None:
    import lightrag

    captured: dict = {}

    class FakeLightRAG:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def initialize_storages(self):
            return None

        async def finalize_storages(self):
            return None

    monkeypatch.setattr(lightrag, "LightRAG", FakeLightRAG)
    settings = replace(
        provider_settings(tmp_path),
        chunk_tokens=700,
        chunk_overlap_tokens=70,
        llm_max_async=2,
        llm_sdk_timeout_seconds=300,
        embedding_sdk_timeout_seconds=120,
    )
    engine = LightRAGEngine(
        settings,
        llm_func=lambda: None,
        embedding_func=lambda: None,
    )

    await engine._acquire_rag("bounded")
    await engine.close()

    assert captured["chunk_token_size"] == 700
    assert captured["chunk_overlap_token_size"] == 70
    assert captured["embedding_chunk_overlap_token_size"] == 70
    assert captured["llm_model_max_async"] == 2
    assert captured["entity_extract_max_records"] == 40
    assert captured["entity_extract_max_entities"] == 20
    assert captured["entity_extract_max_gleaning"] == 0
    assert captured["default_llm_timeout"] == 300
    assert captured["default_embedding_timeout"] == 120
    assert captured["tiktoken_model_name"] == "gpt-4o-mini"
