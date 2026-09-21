import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import routers.autonomous as autonomous
from models.knowledge_evidence import KnowledgeRunState
from services.knowledge_agent import KnowledgeAgentContext
from services.knowledge_llm import KnowledgeLLMProfile
from services.provider_config import ProviderDeadlineExceeded
from test_knowledge_agent import EvidenceClient, KB_ID


def _response(*, content=None, tool_calls=None, finish_reason="stop", refusal=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(
            content=content,
            tool_calls=tool_calls,
            refusal=refusal,
        ),
    )])


def _tool_response(name, arguments, call_id):
    call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    return _response(tool_calls=[call], finish_reason="tool_calls")


def _client(*responses):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=list(responses))
    return client


def _context(client, evidence_client):
    profile = KnowledgeLLMProfile(
        client=client,
        model="kb-model",
        max_output_tokens=4096,
        request_timeout_seconds=120,
        total_timeout_seconds=120,
        extra_body={"enable_thinking": False},
        semantic_fingerprint="test-profile",
        is_override=True,
    )
    return KnowledgeAgentContext(KnowledgeRunState(
        knowledge_base_id=KB_ID,
        revision=2,
        epoch=3,
        owner_id="learner",
        session_id="kbs_empty_test",
        web_enabled=False,
    ), client=evidence_client, llm_profile=profile)


def _kb_messages(content="Explain"):
    return [
        {"role": "system", "content": "Use the selected knowledge base."},
        {"role": "user", "content": content},
    ]


def test_llm_chat_retries_one_empty_response_then_returns_valid_output():
    from services.llm import llm_chat

    client = _client(_response(), _response(content="recovered"))
    result = asyncio.run(llm_chat(
        [{"role": "user", "content": "answer"}],
        client=client,
        max_retries=0,
        require_nonempty_response=True,
    ))

    assert result.choices[0].message.content == "recovered"
    assert client.chat.completions.create.await_count == 2


def test_llm_chat_persistent_empty_raises_dedicated_error():
    from services.llm import EmptyModelOutputError, llm_chat

    client = _client(_response(), _response(content="  "))
    with pytest.raises(EmptyModelOutputError):
        asyncio.run(llm_chat(
            [{"role": "user", "content": "answer"}],
            client=client,
            max_retries=0,
            require_nonempty_response=True,
        ))
    assert client.chat.completions.create.await_count == 2


@pytest.mark.parametrize(
    "response",
    [
        _response(finish_reason="content_filter"),
        _response(refusal="I cannot answer"),
        _response(
            content="must not publish",
            tool_calls=[SimpleNamespace(
                id="blocked-filter",
                function=SimpleNamespace(name="danger", arguments="{}"),
            )],
            finish_reason="content_filter",
        ),
        _response(
            content="must not publish",
            tool_calls=[SimpleNamespace(
                id="blocked-refusal",
                function=SimpleNamespace(name="danger", arguments="{}"),
            )],
            refusal="I cannot answer",
        ),
    ],
)
def test_explicit_filter_or_refusal_is_not_retried(response):
    from services.llm import EmptyModelOutputError, llm_chat

    client = _client(response)
    with pytest.raises(EmptyModelOutputError):
        asyncio.run(llm_chat(
            [{"role": "user", "content": "answer"}],
            client=client,
            max_retries=0,
            require_nonempty_response=True,
        ))
    assert client.chat.completions.create.await_count == 1


def test_blank_length_response_is_left_for_existing_truncation_guard_without_retry():
    from services.llm import llm_chat

    response = _response(finish_reason="length")
    client = _client(response)
    result = asyncio.run(llm_chat(
        [{"role": "user", "content": "answer"}],
        client=client,
        max_retries=0,
        require_nonempty_response=True,
    ))
    assert result is response
    assert client.chat.completions.create.await_count == 1


def test_null_content_with_tool_calls_is_valid_and_not_retried():
    from services.llm import llm_chat

    response = _tool_response("search_knowledge_base", {"query": "x"}, "call-1")
    client = _client(response)
    result = asyncio.run(llm_chat(
        [{"role": "user", "content": "answer"}],
        client=client,
        max_retries=0,
        require_nonempty_response=True,
    ))
    assert result is response
    assert client.chat.completions.create.await_count == 1


def test_empty_retry_shares_one_total_deadline():
    from services.llm import llm_chat

    calls = 0

    async def slow_empty_then_valid(**_kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.03)
        return _response() if calls == 1 else _response(content="late")

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=slow_empty_then_valid)
    with pytest.raises(ProviderDeadlineExceeded):
        asyncio.run(llm_chat(
            [{"role": "user", "content": "answer"}],
            client=client,
            max_retries=0,
            total_timeout=0.04,
            require_nonempty_response=True,
        ))
    assert client.chat.completions.create.await_count == 2


def test_legacy_call_accepts_empty_response_without_retry():
    from services.llm import llm_chat

    response = _response()
    client = _client(response)
    result = asyncio.run(llm_chat(
        [{"role": "user", "content": "answer"}],
        client=client,
        max_retries=0,
    ))
    assert result is response
    assert client.chat.completions.create.await_count == 1


def test_kb_search_then_two_empty_responses_returns_503_without_second_tool_execution():
    client = _client(
        _tool_response("search_knowledge_base", {"query": "concepts"}, "search-1"),
        _response(),
        _response(),
    )
    evidence_client = EvidenceClient()
    context = _context(client, evidence_client)
    messages = _kb_messages()

    with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}):
        with pytest.raises(HTTPException) as raised:
            asyncio.run(autonomous._run_react_loop(
                messages=messages,
                plan=[],
                steps=[],
                tools_called=[],
                user_id="learner",
                document_id=None,
                starting_round=0,
                run_id="empty-kb",
                evidence_registry={},
                grounding_required=True,
                knowledge_context=context,
            ))

    assert raised.value.status_code == 503
    assert "空" in raised.value.detail
    assert client.chat.completions.create.await_count == 3
    assert len(evidence_client.queries) == 1
    assert len([message for message in messages if message["role"] == "assistant"]) == 1


@pytest.mark.parametrize(
    ("response", "detail_fragment"),
    [
        (
            _response(
                content="blocked",
                tool_calls=[SimpleNamespace(
                    id="filtered-tool",
                    function=SimpleNamespace(
                        name="search_knowledge_base",
                        arguments='{"query":"must-not-run"}',
                    ),
                )],
                finish_reason="content_filter",
            ),
            "过滤",
        ),
        (
            _response(
                content="blocked",
                tool_calls=[SimpleNamespace(
                    id="refused-tool",
                    function=SimpleNamespace(
                        name="search_knowledge_base",
                        arguments='{"query":"must-not-run"}',
                    ),
                )],
                refusal="I cannot answer",
            ),
            "拒绝",
        ),
    ],
)
def test_kb_filtered_or_refused_payload_never_dispatches(response, detail_fragment):
    client = _client(response)
    evidence_client = EvidenceClient()
    context = _context(client, evidence_client)
    messages = _kb_messages()

    with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}):
        with pytest.raises(HTTPException) as raised:
            asyncio.run(autonomous._run_react_loop(
                messages=messages,
                plan=[],
                steps=[],
                tools_called=[],
                user_id="learner",
                document_id=None,
                starting_round=0,
                run_id="blocked-kb",
                evidence_registry={},
                grounding_required=True,
                knowledge_context=context,
            ))

    assert raised.value.status_code == 503
    assert detail_fragment in raised.value.detail
    assert client.chat.completions.create.await_count == 1
    assert evidence_client.queries == []
    assert messages == _kb_messages()


def test_kb_blank_length_uses_truncation_classifier_without_empty_retry():
    client = _client(_response(finish_reason="length"))
    context = _context(client, EvidenceClient())
    with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}):
        with pytest.raises(HTTPException) as raised:
            asyncio.run(autonomous._run_react_loop(
                messages=_kb_messages(),
                plan=[],
                steps=[],
                tools_called=[],
                user_id="learner",
                document_id=None,
                starting_round=0,
                run_id="length-kb",
                evidence_registry={},
                grounding_required=True,
                knowledge_context=context,
            ))
    assert raised.value.status_code == 503
    assert "长度上限" in raised.value.detail
    assert client.chat.completions.create.await_count == 1


def test_kb_search_empty_then_valid_json_recovers_with_citation():
    evidence_client = EvidenceClient()
    context = None

    def final_response():
        evidence_id = next(iter(context.state.evidence))
        return _response(content=json.dumps({
            "final_answer": "Recovered",
            "citation_ids": [evidence_id],
            "abstained": False,
        }))

    async def provider(**_kwargs):
        call = provider.await_count
        if call == 1:
            return _tool_response(
                "search_knowledge_base", {"query": "concepts"}, "search-1"
            )
        if call == 2:
            return _response()
        return final_response()

    client = MagicMock()
    provider = AsyncMock(side_effect=provider)
    client.chat.completions.create = provider
    context = _context(client, evidence_client)

    with patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}):
        result = asyncio.run(autonomous._run_react_loop(
            messages=_kb_messages(),
            plan=[],
            steps=[],
            tools_called=[],
            user_id="learner",
            document_id=None,
            starting_round=0,
            run_id="recover-kb",
            evidence_registry={},
            grounding_required=True,
            knowledge_context=context,
        ))

    assert result.final_answer == "Recovered"
    assert result.abstained is False
    assert len(result.source_citations) == 1
    assert provider.await_count == 3


def test_kb_max_round_empty_response_returns_503():
    client = _client(_response(), _response())
    context = _context(client, EvidenceClient())
    from services.autonomous_snapshot import StepRecord

    with (
        patch.dict(os.environ, {"KNOWLEDGE_BASES_ENABLED": "true"}),
        patch.object(autonomous, "MAX_AUTONOMOUS_ROUNDS", 0),
        pytest.raises(HTTPException) as raised,
    ):
        asyncio.run(autonomous._run_react_loop(
            messages=_kb_messages(),
            plan=[],
            steps=[StepRecord(
                round_index=0, tool_name="search_knowledge_base"
            )],
            tools_called=[],
            user_id="learner",
            document_id=None,
            starting_round=0,
            run_id="empty-max",
            evidence_registry={},
            grounding_required=True,
            knowledge_context=context,
        ))
    assert raised.value.status_code == 503
    assert "空" in raised.value.detail
