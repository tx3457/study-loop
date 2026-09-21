import asyncio
import hashlib
import importlib
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

import routers.autonomous as autonomous
from models.knowledge_evidence import KnowledgeRunState
from services.knowledge_agent import KnowledgeAgentContext
from services.provider_config import ProviderConfigurationError
from services.tool_loop import TruncatedModelOutputError, run_tool_round
from test_knowledge_agent import EvidenceClient, KB_ID


def _configured(**overrides):
    values = {
        "KNOWLEDGE_LLM_API_KEY": "private-token",
        "KNOWLEDGE_LLM_BASE_URL": "https://api.siliconflow.cn/v1",
        "KNOWLEDGE_LLM_MODEL": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    }
    values.update(overrides)
    return values


def test_complete_override_builds_bounded_non_thinking_profile_without_exposing_secret():
    from services.knowledge_llm import load_knowledge_llm_profile

    captured = {}

    def build(config, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return object()

    profile = load_knowledge_llm_profile(_configured(), client_builder=build)

    assert profile.model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert profile.max_output_tokens == 4096
    assert profile.request_timeout_seconds == 120
    assert profile.total_timeout_seconds == 120
    assert profile.temperature == pytest.approx(0.7)
    assert profile.top_p == pytest.approx(0.8)
    assert profile.presence_penalty is None
    assert profile.extra_body == {
        "enable_thinking": False,
        "top_k": 20,
        "min_p": 0.0,
    }
    assert captured["timeout"].read == 120
    assert "private-token" not in repr(profile)
    assert "private-token" not in profile.semantic_fingerprint


@pytest.mark.parametrize(
    "missing",
    ["KNOWLEDGE_LLM_API_KEY", "KNOWLEDGE_LLM_BASE_URL", "KNOWLEDGE_LLM_MODEL"],
)
def test_partial_override_fails_closed_instead_of_mixing_provider_fields(missing):
    from services.knowledge_llm import load_knowledge_llm_profile

    values = _configured()
    values.pop(missing)
    with pytest.raises(ProviderConfigurationError):
        load_knowledge_llm_profile(values)


def test_partial_override_does_not_break_legacy_import_but_kb_use_rejects(monkeypatch):
    import services.knowledge_llm as knowledge_llm

    monkeypatch.setenv("KNOWLEDGE_LLM_API_KEY", "partial-secret")
    monkeypatch.delenv("KNOWLEDGE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("KNOWLEDGE_LLM_MODEL", raising=False)

    reloaded = importlib.reload(knowledge_llm)
    with pytest.raises(ProviderConfigurationError):
        reloaded.get_knowledge_llm_profile()


def test_partial_kb_override_does_not_break_main_import_when_feature_is_disabled():
    env = os.environ.copy()
    env.update({
        "KNOWLEDGE_BASES_ENABLED": "false",
        "KNOWLEDGE_LLM_API_KEY": "partial-secret",
        "KNOWLEDGE_LLM_BASE_URL": "",
        "KNOWLEDGE_LLM_MODEL": "",
    })
    result = subprocess.run(
        [sys.executable, "-c", "import main; print(main.app.title)"],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "partial-secret" not in result.stdout + result.stderr


def test_absent_override_preserves_legacy_client_model_and_fingerprint_identity():
    from services.knowledge_llm import load_knowledge_llm_profile

    legacy_client = object()
    profile = load_knowledge_llm_profile(
        {}, legacy_client=legacy_client, legacy_model="legacy-chat"
    )

    assert profile.client is legacy_client
    assert profile.model == "legacy-chat"
    assert profile.is_override is False
    assert profile.call_kwargs == {}
    assert profile.semantic_fingerprint == "legacy"


def test_semantic_fingerprint_changes_for_model_and_thinking_but_not_secret():
    from services.knowledge_llm import load_knowledge_llm_profile

    first = load_knowledge_llm_profile(_configured())
    other_secret = load_knowledge_llm_profile(
        _configured(KNOWLEDGE_LLM_API_KEY="another-secret")
    )
    other_model = load_knowledge_llm_profile(
        _configured(KNOWLEDGE_LLM_MODEL="Qwen/Qwen3-32B")
    )
    thinking = load_knowledge_llm_profile(
        _configured(KNOWLEDGE_LLM_ENABLE_THINKING="true")
    )

    assert first.semantic_fingerprint == other_secret.semantic_fingerprint
    assert first.semantic_fingerprint != other_model.semantic_fingerprint
    assert first.semantic_fingerprint != thinking.semantic_fingerprint


def test_explicit_sampling_parameters_propagate_and_change_fingerprint():
    from services.knowledge_llm import load_knowledge_llm_profile

    baseline = load_knowledge_llm_profile(_configured())
    tuned = load_knowledge_llm_profile(_configured(
        KNOWLEDGE_LLM_TEMPERATURE="0.6",
        KNOWLEDGE_LLM_TOP_P="0.75",
        KNOWLEDGE_LLM_TOP_K="12",
        KNOWLEDGE_LLM_MIN_P="0.05",
    ))

    assert tuned.call_kwargs["temperature"] == pytest.approx(0.6)
    assert tuned.call_kwargs["top_p"] == pytest.approx(0.75)
    assert tuned.call_kwargs["extra_body"] == {
        "enable_thinking": False,
        "top_k": 12,
        "min_p": pytest.approx(0.05),
    }
    assert tuned.semantic_fingerprint != baseline.semantic_fingerprint


def test_different_qa_model_ignores_graph_sampling_and_uses_qa_model_defaults():
    from services.knowledge_llm import load_knowledge_llm_profile

    captured = {}

    def build(config, **_kwargs):
        captured["config"] = config
        return object()

    profile = load_knowledge_llm_profile(_configured(
        KNOWLEDGE_LLM_MODEL="deepseek-ai/DeepSeek-V3.2",
        KNOWLEDGE_LLM_TEMPERATURE="0",
        KNOWLEDGE_LLM_TOP_P="0.2",
        KNOWLEDGE_LLM_TOP_K="3",
        KNOWLEDGE_LLM_MIN_P="0.4",
        KNOWLEDGE_LLM_PRESENCE_PENALTY="-1",
        KNOWLEDGE_LLM_ENABLE_THINKING="true",
        KNOWLEDGE_QA_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507",
    ), client_builder=build)

    assert profile.model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert captured["config"].api_key == "private-token"
    assert captured["config"].base_url == "https://api.siliconflow.cn/v1"
    assert captured["config"].model == profile.model
    assert profile.call_kwargs["temperature"] == pytest.approx(0.7)
    assert profile.call_kwargs["top_p"] == pytest.approx(0.8)
    assert "presence_penalty" not in profile.call_kwargs
    assert profile.call_kwargs["extra_body"] == {
        "enable_thinking": False,
        "top_k": 20,
        "min_p": 0.0,
    }


def test_same_qa_model_inherits_graph_sampling_overrides():
    from services.knowledge_llm import load_knowledge_llm_profile

    profile = load_knowledge_llm_profile(_configured(
        KNOWLEDGE_QA_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507",
        KNOWLEDGE_LLM_TEMPERATURE="0.4",
        KNOWLEDGE_LLM_TOP_P="0.6",
        KNOWLEDGE_LLM_TOP_K="9",
        KNOWLEDGE_LLM_MIN_P="0.1",
        KNOWLEDGE_LLM_PRESENCE_PENALTY="0.5",
        KNOWLEDGE_LLM_ENABLE_THINKING="true",
    ))

    assert profile.call_kwargs["temperature"] == pytest.approx(0.4)
    assert profile.call_kwargs["top_p"] == pytest.approx(0.6)
    assert profile.call_kwargs["presence_penalty"] == pytest.approx(0.5)
    assert profile.call_kwargs["extra_body"] == {
        "enable_thinking": True,
        "top_k": 9,
        "min_p": pytest.approx(0.1),
    }


def test_explicit_qa_sampling_overrides_graph_values():
    from services.knowledge_llm import load_knowledge_llm_profile

    profile = load_knowledge_llm_profile(_configured(
        KNOWLEDGE_LLM_TEMPERATURE="0.1",
        KNOWLEDGE_LLM_TOP_P="0.2",
        KNOWLEDGE_LLM_TOP_K="3",
        KNOWLEDGE_LLM_MIN_P="0.4",
        KNOWLEDGE_LLM_PRESENCE_PENALTY="-1",
        KNOWLEDGE_LLM_ENABLE_THINKING="true",
        KNOWLEDGE_QA_TEMPERATURE="0.55",
        KNOWLEDGE_QA_TOP_P="0.75",
        KNOWLEDGE_QA_TOP_K="17",
        KNOWLEDGE_QA_MIN_P="0.05",
        KNOWLEDGE_QA_PRESENCE_PENALTY="1.25",
        KNOWLEDGE_QA_ENABLE_THINKING="false",
    ))

    assert profile.call_kwargs["temperature"] == pytest.approx(0.55)
    assert profile.call_kwargs["top_p"] == pytest.approx(0.75)
    assert profile.call_kwargs["presence_penalty"] == pytest.approx(1.25)
    assert profile.call_kwargs["extra_body"] == {
        "enable_thinking": False,
        "top_k": 17,
        "min_p": pytest.approx(0.05),
    }


@pytest.mark.parametrize(
    "override",
    [
        {"KNOWLEDGE_QA_MODEL": "Qwen/Qwen3-30B-A3B-Instruct-2507"},
        {"KNOWLEDGE_QA_TEMPERATURE": "0.7"},
        {"KNOWLEDGE_QA_ENABLE_THINKING": "false"},
    ],
)
def test_qa_role_overrides_require_complete_parent_profile(override):
    from services.knowledge_llm import load_knowledge_llm_profile

    with pytest.raises(ProviderConfigurationError):
        load_knowledge_llm_profile(override)


def test_legacy_qa_timeout_and_output_knobs_alone_keep_legacy_fallback():
    from services.knowledge_llm import load_knowledge_llm_profile

    legacy_client = object()
    profile = load_knowledge_llm_profile(
        {
            "KNOWLEDGE_QA_TIMEOUT_SECONDS": "180",
            "KNOWLEDGE_QA_MAX_OUTPUT_TOKENS": "8192",
        },
        legacy_client=legacy_client,
        legacy_model="legacy-chat",
    )

    assert profile.client is legacy_client
    assert profile.model == "legacy-chat"
    assert profile.call_kwargs == {}


@pytest.mark.parametrize(
    "override",
    [
        {"KNOWLEDGE_LLM_MODEL": "replace-with-chat-model"},
        {"KNOWLEDGE_QA_MODEL": "replace-with-chat-model"},
    ],
)
def test_parent_and_selected_qa_models_are_both_validated(override):
    from services.knowledge_llm import load_knowledge_llm_profile

    with pytest.raises(ProviderConfigurationError):
        load_knowledge_llm_profile(_configured(**override))


def test_qa_model_and_sampling_changes_update_semantic_fingerprint():
    from services.knowledge_llm import load_knowledge_llm_profile

    baseline = load_knowledge_llm_profile(_configured())
    other_model = load_knowledge_llm_profile(_configured(
        KNOWLEDGE_QA_MODEL="Qwen/Qwen3.5-35B-A3B"
    ))
    other_sampling = load_knowledge_llm_profile(_configured(
        KNOWLEDGE_QA_TEMPERATURE="0.5"
    ))

    assert baseline.semantic_fingerprint != other_model.semantic_fingerprint
    assert baseline.semantic_fingerprint != other_sampling.semantic_fingerprint


def test_qa_only_model_and_generation_changes_do_not_change_graph_index_hash(
    monkeypatch,
):
    from graph_service.config import Settings
    from graph_service.engine import index_config_hash
    from services.knowledge_llm import load_knowledge_llm_profile

    graph_env = {
        "KNOWLEDGE_DATABASE_URL": "postgresql://graph@127.0.0.1:5432/graph",
        "KNOWLEDGE_SERVICE_TOKEN": "test-internal-token",
        "KNOWLEDGE_PROVIDER": "openai",
        "KNOWLEDGE_EMBEDDING_DIM": "1024",
        "KNOWLEDGE_LLM_API_KEY": "test-chat-key",
        "KNOWLEDGE_LLM_BASE_URL": "https://api.siliconflow.cn/v1",
        "KNOWLEDGE_LLM_MODEL": "deepseek-ai/DeepSeek-V3.2",
        "KNOWLEDGE_EMBEDDING_API_KEY": "test-embedding-key",
        "KNOWLEDGE_EMBEDDING_BASE_URL": "https://api.siliconflow.cn/v1",
        "KNOWLEDGE_EMBEDDING_MODEL": "BAAI/bge-m3",
    }
    for name, value in graph_env.items():
        monkeypatch.setenv(name, value)
    for name in (
        "KNOWLEDGE_QA_MODEL",
        "KNOWLEDGE_QA_TEMPERATURE",
        "KNOWLEDGE_QA_TOP_P",
        "KNOWLEDGE_QA_TOP_K",
        "KNOWLEDGE_QA_MIN_P",
        "KNOWLEDGE_QA_PRESENCE_PENALTY",
        "KNOWLEDGE_QA_ENABLE_THINKING",
    ):
        monkeypatch.delenv(name, raising=False)

    graph_before = Settings.from_env()
    graph_hash_before = index_config_hash(graph_before)
    qa_before = load_knowledge_llm_profile()

    qa_env = {
        "KNOWLEDGE_QA_MODEL": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "KNOWLEDGE_QA_TEMPERATURE": "0.65",
        "KNOWLEDGE_QA_TOP_P": "0.75",
        "KNOWLEDGE_QA_TOP_K": "17",
        "KNOWLEDGE_QA_MIN_P": "0.05",
        "KNOWLEDGE_QA_PRESENCE_PENALTY": "0.4",
        "KNOWLEDGE_QA_ENABLE_THINKING": "false",
    }
    for name, value in qa_env.items():
        monkeypatch.setenv(name, value)

    graph_after = Settings.from_env()
    graph_hash_after = index_config_hash(graph_after)
    qa_after = load_knowledge_llm_profile()

    assert graph_before.llm_model == "deepseek-ai/DeepSeek-V3.2"
    assert graph_after.llm_model == graph_before.llm_model
    assert graph_hash_after == graph_hash_before
    assert qa_before.model == graph_before.llm_model
    assert qa_after.model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert qa_after.semantic_fingerprint != qa_before.semantic_fingerprint


def test_other_models_have_no_sampling_override_unless_explicit():
    from services.knowledge_llm import load_knowledge_llm_profile

    profile = load_knowledge_llm_profile(
        _configured(KNOWLEDGE_LLM_MODEL="Qwen/another-model")
    )
    assert "temperature" not in profile.call_kwargs
    assert "top_p" not in profile.call_kwargs
    assert profile.call_kwargs["extra_body"] == {"enable_thinking": False}


def test_qwen35_uses_non_thinking_recipe_with_presence_penalty():
    from services.knowledge_llm import load_knowledge_llm_profile

    profile = load_knowledge_llm_profile(
        _configured(KNOWLEDGE_LLM_MODEL="Qwen/Qwen3.5-35B-A3B")
    )

    assert profile.call_kwargs["temperature"] == pytest.approx(0.7)
    assert profile.call_kwargs["top_p"] == pytest.approx(0.8)
    assert profile.call_kwargs["presence_penalty"] == pytest.approx(1.5)
    assert profile.call_kwargs["extra_body"] == {
        "enable_thinking": False,
        "top_k": 20,
        "min_p": 0.0,
    }


def test_presence_penalty_is_optional_for_other_models_and_explicitly_supported():
    from services.knowledge_llm import load_knowledge_llm_profile

    default = load_knowledge_llm_profile(
        _configured(KNOWLEDGE_LLM_MODEL="Qwen/another-model")
    )
    explicit = load_knowledge_llm_profile(_configured(
        KNOWLEDGE_LLM_MODEL="Qwen/another-model",
        KNOWLEDGE_LLM_PRESENCE_PENALTY="-0.5",
    ))

    assert "presence_penalty" not in default.call_kwargs
    assert explicit.call_kwargs["presence_penalty"] == pytest.approx(-0.5)
    assert explicit.semantic_fingerprint != default.semantic_fingerprint


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("KNOWLEDGE_LLM_TEMPERATURE", "nan"),
        ("KNOWLEDGE_LLM_TEMPERATURE", "2.1"),
        ("KNOWLEDGE_LLM_TOP_P", "0"),
        ("KNOWLEDGE_LLM_TOP_P", "1.1"),
        ("KNOWLEDGE_LLM_TOP_K", "0"),
        ("KNOWLEDGE_LLM_TOP_K", "1.5"),
        ("KNOWLEDGE_LLM_TOP_K", "101"),
        ("KNOWLEDGE_LLM_MIN_P", "-0.1"),
        ("KNOWLEDGE_LLM_MIN_P", "1.1"),
        ("KNOWLEDGE_LLM_PRESENCE_PENALTY", "nan"),
        ("KNOWLEDGE_LLM_PRESENCE_PENALTY", "-2.1"),
        ("KNOWLEDGE_LLM_PRESENCE_PENALTY", "2.1"),
    ],
)
def test_invalid_sampling_parameters_are_rejected(name, value):
    from services.knowledge_llm import load_knowledge_llm_profile

    with pytest.raises(ValueError):
        load_knowledge_llm_profile(_configured(**{name: value}))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("KNOWLEDGE_QA_TEMPERATURE", "nan"),
        ("KNOWLEDGE_QA_TOP_P", "0"),
        ("KNOWLEDGE_QA_TOP_K", "101"),
        ("KNOWLEDGE_QA_MIN_P", "-0.1"),
        ("KNOWLEDGE_QA_PRESENCE_PENALTY", "2.1"),
        ("KNOWLEDGE_QA_ENABLE_THINKING", "sometimes"),
    ],
)
def test_invalid_qa_sampling_parameters_are_rejected(name, value):
    from services.knowledge_llm import load_knowledge_llm_profile

    with pytest.raises(ValueError):
        load_knowledge_llm_profile(_configured(**{name: value}))


def test_invalid_bounds_and_boolean_are_rejected():
    from services.knowledge_llm import load_knowledge_llm_profile

    for override in (
        {"KNOWLEDGE_QA_TIMEOUT_SECONDS": "0"},
        {"KNOWLEDGE_QA_TIMEOUT_SECONDS": "301"},
        {"KNOWLEDGE_QA_MAX_OUTPUT_TOKENS": "0"},
        {"KNOWLEDGE_QA_MAX_OUTPUT_TOKENS": "999999"},
        {"KNOWLEDGE_LLM_ENABLE_THINKING": "sometimes"},
    ):
        with pytest.raises(ValueError):
            load_knowledge_llm_profile(_configured(**override))


def test_truncated_tool_batch_is_rejected_before_transcript_mutation_or_dispatch():
    tool_call = SimpleNamespace(
        id="partial",
        function=SimpleNamespace(name="danger", arguments='{"partial":'),
    )
    response = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="length",
        message=SimpleNamespace(content=None, tool_calls=[tool_call]),
    )])
    messages = []
    with patch("services.tool_loop.llm_chat", AsyncMock(return_value=response)):
        with pytest.raises(TruncatedModelOutputError):
            asyncio.run(run_tool_round(
                messages,
                tools=[],
                reject_truncated_output=True,
            ))
    assert messages == []


def test_kb_round_and_max_round_fallback_use_the_same_scoped_profile():
    from services.knowledge_llm import KnowledgeLLMProfile

    client = object()
    profile = KnowledgeLLMProfile(
        client=client,
        model="kb-model",
        max_output_tokens=321,
        request_timeout_seconds=12,
        total_timeout_seconds=15,
        extra_body={"enable_thinking": False},
        semantic_fingerprint="profile-hash",
        is_override=True,
    )
    context = KnowledgeAgentContext(KnowledgeRunState(
        knowledge_base_id=KB_ID,
        revision=2,
        epoch=3,
        owner_id="learner",
        session_id="kbs_test",
        web_enabled=False,
    ), client=EvidenceClient(), llm_profile=profile)
    calls = []

    async def provider(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content=(
                '{"final_answer":"bounded answer","citation_ids":[],'
                '"abstained":false}'
            ), tool_calls=None),
        )])

    with (
        patch.dict("os.environ", {"KNOWLEDGE_BASES_ENABLED": "true"}),
        patch.object(autonomous, "MAX_AUTONOMOUS_ROUNDS", 0),
        patch.object(autonomous, "llm_chat", side_effect=provider),
    ):
        from services.autonomous_snapshot import StepRecord

        result = asyncio.run(autonomous._run_react_loop(
            messages=[
                {"role": "system", "content": "Use the selected knowledge base."},
                {"role": "user", "content": "answer"},
            ],
            plan=[], steps=[StepRecord(
                round_index=0, tool_name="search_knowledge_base"
            )], tools_called=[], user_id="learner", document_id=None,
            starting_round=0, run_id="kb", evidence_registry={},
            grounding_required=False, knowledge_context=context,
        ))

    assert result.final_answer == "bounded answer"
    assert calls == [{
        "client": client,
        "model": "kb-model",
        "max_tokens": 321,
        "timeout": 12,
        "total_timeout": 15,
        "extra_body": {"enable_thinking": False},
        "require_nonempty_response": True,
        "response_format": {"type": "json_object"},
    }]


def test_kb_react_round_routes_only_through_scoped_profile():
    from services.knowledge_llm import KnowledgeLLMProfile

    client = object()
    profile = KnowledgeLLMProfile(
        client=client,
        model="kb-model",
        max_output_tokens=4096,
        request_timeout_seconds=120,
        total_timeout_seconds=120,
        extra_body={"enable_thinking": False},
        semantic_fingerprint="profile-hash",
        is_override=True,
    )
    context = KnowledgeAgentContext(KnowledgeRunState(
        knowledge_base_id=KB_ID,
        revision=2,
        epoch=3,
        owner_id="learner",
        session_id="kbs_test",
        web_enabled=False,
    ), client=EvidenceClient(), llm_profile=profile)
    calls = []

    async def provider(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            tool_call = SimpleNamespace(
                id="search",
                function=SimpleNamespace(
                    name="search_knowledge_base",
                    arguments='{"query":"answer"}',
                ),
            )
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(content=None, tool_calls=[tool_call]),
            )])
        evidence_id = next(iter(context.state.evidence))
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content=(
                '{"final_answer":"done","citation_ids":["'
                + evidence_id + '"],"abstained":false}'
            ), tool_calls=None),
        )])

    with (
        patch.dict("os.environ", {"KNOWLEDGE_BASES_ENABLED": "true"}),
        patch("services.tool_loop.llm_chat", side_effect=provider),
    ):
        result = asyncio.run(autonomous._run_react_loop(
            messages=[
                {"role": "system", "content": "Use the selected knowledge base."},
                {"role": "user", "content": "answer"},
            ],
            plan=[], steps=[], tools_called=[], user_id="learner", document_id=None,
            starting_round=0, run_id="kb", evidence_registry={},
            grounding_required=False, knowledge_context=context,
        ))

    assert result.final_answer == "done"
    assert len(calls) == 2
    assert calls[0]["client"] is client
    assert calls[0]["model"] == "kb-model"
    assert calls[0]["max_tokens"] == 4096
    assert calls[0]["timeout"] == 120
    assert calls[0]["total_timeout"] == 120
    assert calls[0]["extra_body"] == {"enable_thinking": False}
    assert calls[0]["require_nonempty_response"] is True
    assert calls[0]["tool_choice"] == "required"
    assert "response_format" not in calls[0]
    assert calls[1]["tool_choice"] == "auto"
    assert calls[1]["response_format"] == {"type": "json_object"}
    assert all(
        schema["function"]["name"] != "finalize"
        for call in calls for schema in call["tools"]
    )


def test_grounded_kb_round_requires_a_tool_choice():
    from services.knowledge_llm import KnowledgeLLMProfile

    profile = KnowledgeLLMProfile(
        client=object(),
        model="kb-model",
        semantic_fingerprint="profile-hash",
        is_override=True,
    )
    context = KnowledgeAgentContext(KnowledgeRunState(
        knowledge_base_id=KB_ID,
        revision=2,
        epoch=3,
        owner_id="learner",
        session_id="kbs_test",
        web_enabled=False,
    ), client=EvidenceClient(), llm_profile=profile)
    calls = []

    async def provider(*args, **kwargs):
        calls.append(kwargs)
        tool_call = SimpleNamespace(
            id="ask",
            function=SimpleNamespace(
                name="ask_user",
                arguments='{"question":"Which scope?"}',
            ),
        )
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="tool_calls",
            message=SimpleNamespace(content=None, tool_calls=[tool_call]),
        )])

    with (
        patch.dict("os.environ", {"KNOWLEDGE_BASES_ENABLED": "true"}),
        patch("services.tool_loop.llm_chat", side_effect=provider),
    ):
        asyncio.run(autonomous._run_react_loop(
            messages=[
                {"role": "system", "content": "Use the selected knowledge base."},
                {"role": "user", "content": "answer"},
            ],
            plan=[], steps=[], tools_called=[], user_id="learner", document_id=None,
            starting_round=0, run_id="kb", evidence_registry={},
            grounding_required=True, knowledge_context=context,
        ))

    assert calls[0]["tool_choice"] == "required"
    assert "response_format" not in calls[0]


def test_provider_text_cannot_publish_uncited_grounded_kb_answer():
    from services.knowledge_llm import KnowledgeLLMProfile

    context = KnowledgeAgentContext(KnowledgeRunState(
        knowledge_base_id=KB_ID,
        revision=2,
        epoch=3,
        owner_id="learner",
        session_id="kbs_test",
        web_enabled=False,
    ), client=EvidenceClient(), llm_profile=KnowledgeLLMProfile(
        client=object(),
        model="kb-model",
        semantic_fingerprint="profile-hash",
        is_override=True,
    ))

    async def ignores_required(*args, **kwargs):
        assert kwargs["tool_choice"] == "required"
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="unsupported claim", tool_calls=None),
        )])

    with (
        patch.dict("os.environ", {"KNOWLEDGE_BASES_ENABLED": "true"}),
        patch("services.tool_loop.llm_chat", side_effect=ignores_required),
    ):
        with pytest.raises(HTTPException) as raised:
            asyncio.run(autonomous._run_react_loop(
                messages=[
                    {"role": "system", "content": "Use the selected knowledge base."},
                    {"role": "user", "content": "answer"},
                ],
                plan=[], steps=[], tools_called=[], user_id="learner", document_id=None,
                starting_round=0, run_id="kb", evidence_registry={},
                grounding_required=True, knowledge_context=context,
            ))

    assert raised.value.status_code == 503
    assert "检索" in raised.value.detail


def test_knowledge_tool_timeout_is_scoped_and_bounded():
    profile = SimpleNamespace(semantic_fingerprint="legacy")
    with patch.dict("os.environ", {"KNOWLEDGE_TOOL_TIMEOUT_SECONDS": "105"}):
        context = KnowledgeAgentContext(KnowledgeRunState(
            knowledge_base_id=KB_ID,
            revision=2,
            epoch=3,
            owner_id="learner",
            session_id="kbs_test",
            web_enabled=False,
        ), client=EvidenceClient(), llm_profile=profile)
    assert context.registry.get("search_knowledge_base").metadata.timeout_sec == 105


def test_override_changes_kb_resume_fingerprint_without_changing_legacy_value():
    from services.knowledge_llm import KnowledgeLLMProfile

    state = KnowledgeRunState(
        knowledge_base_id=KB_ID,
        revision=2,
        epoch=3,
        owner_id="learner",
        session_id="kbs_test",
        web_enabled=False,
    )
    legacy = KnowledgeAgentContext(
        state.model_copy(deep=True),
        client=EvidenceClient(),
        llm_profile=SimpleNamespace(semantic_fingerprint="legacy"),
    )
    override = KnowledgeAgentContext(
        state.model_copy(deep=True),
        client=EvidenceClient(),
        llm_profile=KnowledgeLLMProfile(
            client=object(),
            model="kb-model",
            semantic_fingerprint="profile-hash",
            is_override=True,
        ),
    )
    other_override = KnowledgeAgentContext(
        state.model_copy(deep=True),
        client=EvidenceClient(),
        llm_profile=KnowledgeLLMProfile(
            client=object(),
            model="kb-model-2",
            semantic_fingerprint="other-profile-hash",
            is_override=True,
        ),
    )

    assert legacy.fingerprint() != override.fingerprint()
    assert override.fingerprint() != other_override.fingerprint()

    old_legacy = hashlib.sha256(
        ("knowledge-tools-v2:" + autonomous._current_registry_sha256(
            legacy.registry
        )).encode()
    ).hexdigest()
    old_override = hashlib.sha256(
        (
            "knowledge-tools-v3:"
            + autonomous._current_registry_sha256(override.registry)
            + ":profile-hash"
        ).encode()
    ).hexdigest()
    assert legacy.fingerprint() != old_legacy
    assert override.fingerprint() != old_override

    pre_projection_policy = "grounded-tool-choice-required-v1"
    pre_projection_legacy = hashlib.sha256(
        (
            "knowledge-tools-v3:"
            + autonomous._current_registry_sha256(legacy.registry)
            + ":"
            + pre_projection_policy
        ).encode()
    ).hexdigest()
    pre_projection_override = hashlib.sha256(
        (
            "knowledge-tools-v4:"
            + autonomous._current_registry_sha256(override.registry)
            + ":profile-hash:"
            + pre_projection_policy
        ).encode()
    ).hexdigest()
    assert legacy.fingerprint() != pre_projection_legacy
    assert override.fingerprint() != pre_projection_override


def test_ordinary_document_registry_fingerprint_is_unchanged():
    from services.autonomous_snapshot import current_registry_sha256

    assert autonomous._current_registry_sha256() == current_registry_sha256()
