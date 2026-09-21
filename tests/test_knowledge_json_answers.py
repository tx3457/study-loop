import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import routers.autonomous as autonomous
from models.knowledge_evidence import KnowledgeRunState
from services.knowledge_agent import KnowledgeAgentContext
from services.knowledge_llm import KnowledgeLLMProfile
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
    return _response(
        tool_calls=[SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
        )],
        finish_reason="tool_calls",
    )


def _context(client, *, evidence_client=None, web_enabled=False):
    return KnowledgeAgentContext(
        KnowledgeRunState(
            knowledge_base_id=KB_ID,
            revision=2,
            epoch=3,
            owner_id="learner",
            session_id="kbs_json_test",
            web_enabled=web_enabled,
            web_query="public query" if web_enabled else "",
        ),
        client=evidence_client or EvidenceClient(),
        llm_profile=KnowledgeLLMProfile(
            client=client,
            model="kb-model",
            semantic_fingerprint="json-answer-profile",
            is_override=True,
        ),
    )


def _run(context, *, max_rounds=None, steps=None):
    patches = [patch.dict("os.environ", {"KNOWLEDGE_BASES_ENABLED": "true"})]
    if max_rounds is not None:
        patches.append(patch.object(autonomous, "MAX_AUTONOMOUS_ROUNDS", max_rounds))
    with patches[0]:
        if len(patches) == 2:
            with patches[1]:
                return asyncio.run(_invoke(context, steps=steps))
        return asyncio.run(_invoke(context, steps=steps))


async def _invoke(context, *, steps=None):
    return await autonomous._run_react_loop(
        messages=[
            {"role": "system", "content": "Use the selected knowledge base."},
            {"role": "user", "content": "Explain graphs"},
        ],
        plan=[],
        steps=steps or [],
        tools_called=[],
        user_id="learner",
        document_id=None,
        starting_round=0,
        run_id="kb-json",
        evidence_registry={},
        grounding_required=True,
        knowledge_context=context,
    )


def test_first_round_requires_retrieval_without_json_mode_then_accepts_native_json():
    calls = []
    context = None

    async def provider(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _tool_response(
                "search_knowledge_base", {"query": "graphs"}, "search-1"
            )
        evidence_id = next(iter(context.state.evidence))
        return _response(content=json.dumps({
            "final_answer": "Graphs connect concepts across documents.",
            "citation_ids": [evidence_id],
            "abstained": False,
            "reason": "evidence supports the answer",
        }))

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=provider)
    context = _context(client)

    result = _run(context)

    assert result.final_answer == "Graphs connect concepts across documents."
    assert result.source_citations[0].evidence_id in context.state.evidence
    assert calls[0]["tool_choice"] == "required"
    assert "response_format" not in calls[0]
    assert calls[1]["tool_choice"] == "auto"
    assert calls[1]["response_format"] == {"type": "json_object"}
    post_gate_prompt = calls[1]["messages"][0]["content"]
    assert "唯一面向用户" in post_gate_prompt
    assert "每个子问题" in post_gate_prompt
    assert "引言" in post_gate_prompt
    assert "无关或不足" in post_gate_prompt
    assert [message["role"] for message in calls[1]["messages"]].count("system") == 1
    assert calls[1]["messages"][0]["role"] == "system"
    assert all(
        schema["function"]["name"] != "finalize"
        for call in calls for schema in call["tools"]
    )


def test_native_json_with_forged_citation_fails_closed():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[
        _tool_response("search_knowledge_base", {"query": "graphs"}, "search-1"),
        _response(content=json.dumps({
            "final_answer": "Unsupported claim",
            "citation_ids": ["forged"],
            "abstained": False,
        })),
    ])

    result = _run(_context(client))

    assert result.abstained is True
    assert result.source_citations == []
    assert result.invalid_citation_count == 1


@pytest.mark.parametrize("content", ["plain text", '{"final_answer":"x","extra":1}'])
def test_post_retrieval_nonconforming_content_returns_503(content):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[
        _tool_response("search_knowledge_base", {"query": "graphs"}, "search-1"),
        _response(content=content),
    ])

    with pytest.raises(HTTPException) as raised:
        _run(_context(client))

    assert raised.value.status_code == 503
    assert "JSON" in raised.value.detail


def test_premature_native_abstention_before_retrieval_returns_503():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_response(content=json.dumps({
        "final_answer": "No evidence",
        "citation_ids": [],
        "abstained": True,
    })))

    with pytest.raises(HTTPException) as raised:
        _run(_context(client))

    assert raised.value.status_code == 503
    assert "检索" in raised.value.detail


def test_successful_zero_result_search_unlocks_native_abstention():
    evidence_client = EvidenceClient()
    evidence_client.query = AsyncMock(return_value={
        "scope": {"knowledge_base_id": KB_ID, "revision": 2, "epoch": 3},
        "evidence": [],
    })
    client = MagicMock()
    calls = []

    async def provider(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _tool_response("search_knowledge_base", {"query": "missing"}, "search-1")
        return _response(content=json.dumps({
            "final_answer": "No evidence was found.",
            "citation_ids": [],
            "abstained": True,
        }))

    client.chat.completions.create = AsyncMock(side_effect=provider)
    result = _run(_context(client, evidence_client=evidence_client))

    assert result.abstained is True
    assert calls[1]["response_format"] == {"type": "json_object"}


def test_failed_retrieval_does_not_unlock_native_answer():
    evidence_client = EvidenceClient()
    evidence_client.query = AsyncMock(side_effect=RuntimeError("backend down"))
    client = MagicMock()
    calls = []

    async def provider(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _tool_response("search_knowledge_base", {"query": "graphs"}, "search-1")
        return _response(content=json.dumps({
            "final_answer": "No evidence",
            "citation_ids": [],
            "abstained": True,
        }))

    client.chat.completions.create = AsyncMock(side_effect=provider)
    with pytest.raises(HTTPException) as raised:
        _run(_context(client, evidence_client=evidence_client))

    assert raised.value.status_code == 503
    assert calls[1]["tool_choice"] == "required"
    assert "response_format" not in calls[1]


def test_projection_budget_failure_stays_sanitized_and_does_not_unlock_answer():
    evidence_client = EvidenceClient()
    evidence_client.query = AsyncMock(return_value={
        "scope": {"knowledge_base_id": KB_ID, "revision": 2, "epoch": 3},
        "evidence": [{
            "knowledge_base_id": KB_ID,
            "document_id": "doc-large",
            "source_version_id": "version-large",
            "chunk_id": f"chunk-{index}",
            "title": "Large",
            "text": str(index) + ("x" * 3_999),
        } for index in range(20)],
    })
    client = MagicMock()
    calls = []

    async def provider(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _tool_response(
                "search_knowledge_base", {"query": "too broad"}, "search-1"
            )
        tool_messages = [
            message for message in kwargs["messages"]
            if message.get("role") == "tool"
        ]
        sanitized = json.loads(tool_messages[-1]["content"])
        assert sanitized == {
            "error": "工具执行失败",
            "error_type": "HTTPException",
        }
        assert "缩小" not in tool_messages[-1]["content"]
        return _response(content=json.dumps({
            "final_answer": "must not publish",
            "citation_ids": [],
            "abstained": True,
        }))

    client.chat.completions.create = AsyncMock(side_effect=provider)
    context = _context(client, evidence_client=evidence_client)

    with pytest.raises(HTTPException) as raised:
        _run(context)

    assert raised.value.status_code == 503
    assert context.state.evidence == {}
    assert calls[1]["tool_choice"] == "required"
    assert "response_format" not in calls[1]


def test_partially_parsed_failed_search_cannot_unlock_or_publish_ghost_evidence():
    evidence_client = EvidenceClient()
    valid = {
        "knowledge_base_id": KB_ID,
        "document_id": "doc-1",
        "source_version_id": "version-1",
        "chunk_id": "chunk-1",
        "title": "Valid",
        "text": "valid supporting evidence",
    }
    evidence_client.query = AsyncMock(return_value={
        "scope": {"knowledge_base_id": KB_ID, "revision": 2, "epoch": 3},
        "evidence": [valid, {**valid, "chunk_id": "bad", "text": 123}],
    })
    client = MagicMock()
    calls = []
    context = None

    async def provider(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _tool_response("search_knowledge_base", {"query": "graphs"}, "search-1")
        evidence_id = next(iter(context.state.evidence), "ghost")
        return _response(content=json.dumps({
            "final_answer": "must not publish",
            "citation_ids": [evidence_id],
            "abstained": False,
        }))

    client.chat.completions.create = AsyncMock(side_effect=provider)
    context = _context(client, evidence_client=evidence_client)

    with pytest.raises(HTTPException) as raised:
        _run(context)

    assert raised.value.status_code == 503
    assert context.state.evidence == {}
    assert calls[1]["tool_choice"] == "required"
    assert "response_format" not in calls[1]


def test_web_search_snippets_do_not_unlock_native_answer():
    client = MagicMock()
    calls = []

    async def provider(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _tool_response("search_web", {}, "web-search-1")
        return _response(content=json.dumps({
            "final_answer": "No evidence",
            "citation_ids": [],
            "abstained": True,
        }))

    client.chat.completions.create = AsyncMock(side_effect=provider)
    with patch(
        "services.knowledge_web.search_web",
        AsyncMock(return_value=[{
            "title": "Result", "url": "https://example.org", "snippet": "summary"
        }]),
    ):
        with pytest.raises(HTTPException) as raised:
            _run(_context(client, web_enabled=True))

    assert raised.value.status_code == 503
    assert calls[1]["tool_choice"] == "required"
    assert "response_format" not in calls[1]


def test_scope_change_after_native_json_generation_prevents_publication():
    evidence_client = EvidenceClient()
    client = MagicMock()
    context = None

    async def provider(**_kwargs):
        if client.chat.completions.create.await_count == 1:
            return _tool_response("search_knowledge_base", {"query": "graphs"}, "search-1")
        evidence_client.epoch = 4
        return _response(content=json.dumps({
            "final_answer": "Graphs connect concepts.",
            "citation_ids": [next(iter(context.state.evidence))],
            "abstained": False,
        }))

    client.chat.completions.create = AsyncMock(side_effect=provider)
    context = _context(client, evidence_client=evidence_client)

    with pytest.raises(HTTPException) as raised:
        _run(context)

    assert raised.value.status_code == 409


def test_kb_max_round_fallback_uses_strict_json_after_retrieval():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_response(content=json.dumps({
        "final_answer": "No evidence was found.",
        "citation_ids": [],
        "abstained": True,
    })))
    from services.autonomous_snapshot import StepRecord

    result = _run(
        _context(client),
        max_rounds=0,
        steps=[StepRecord(round_index=0, tool_name="search_knowledge_base")],
    )

    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    roles = [message["role"] for message in kwargs["messages"]]
    assert roles[0] == "system"
    assert roles.count("system") == 1
    assert result.abstained is True
    assert result.truncated is True


def test_kb_ephemeral_summary_excludes_tool_arguments_and_observations():
    from services.autonomous_snapshot import StepRecord

    context = _context(MagicMock())
    summary = autonomous._knowledge_state_summary(
        context,
        [StepRecord(
            round_index=0,
            tool_name="search_knowledge_base",
            tool_args={"query": "UNTRUSTED_QUERY"},
            observation_preview="UNTRUSTED_OBSERVATION",
        )],
        1,
    )

    assert "search_knowledge_base" in summary
    assert "UNTRUSTED_QUERY" not in summary
    assert "UNTRUSTED_OBSERVATION" not in summary


def test_length_stopped_valid_json_is_rejected_before_parsing():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_response(
        content=json.dumps({
            "final_answer": "Looks complete but was truncated",
            "citation_ids": [],
            "abstained": True,
        }),
        finish_reason="length",
    ))
    from services.autonomous_snapshot import StepRecord

    with pytest.raises(HTTPException) as raised:
        _run(
            _context(client),
            steps=[StepRecord(round_index=0, tool_name="search_knowledge_base")],
        )

    assert raised.value.status_code == 503
    assert "长度上限" in raised.value.detail


def test_kb_max_round_json_with_observed_citation_is_accepted():
    client = MagicMock()
    context = _context(client)
    with patch.dict("os.environ", {"KNOWLEDGE_BASES_ENABLED": "true"}):
        asyncio.run(context.search("graphs"))
    evidence_id = next(iter(context.state.evidence))
    client.chat.completions.create = AsyncMock(return_value=_response(content=json.dumps({
        "final_answer": "Graphs connect concepts.",
        "citation_ids": [evidence_id],
        "abstained": False,
    })))

    from services.autonomous_snapshot import StepRecord

    result = _run(
        context,
        max_rounds=0,
        steps=[StepRecord(round_index=0, tool_name="search_knowledge_base")],
    )

    assert result.final_answer == "Graphs connect concepts."
    assert result.source_citations[0].evidence_id == evidence_id


def test_kb_max_round_invalid_json_returns_503():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_response(content="not json"))
    from services.autonomous_snapshot import StepRecord

    with pytest.raises(HTTPException) as raised:
        _run(
            _context(client),
            max_rounds=0,
            steps=[StepRecord(round_index=0, tool_name="search_knowledge_base")],
        )

    assert raised.value.status_code == 503
    assert "JSON" in raised.value.detail


def test_successful_web_fetch_unlocks_native_json_answer():
    url = "https://example.org/source"
    evidence_client = EvidenceClient()
    evidence_client.store_web_snapshot = AsyncMock(return_value={"id": "snapshot-1"})
    evidence_client.get_web_snapshot = AsyncMock(return_value={"content_hash": "a" * 64})
    page = SimpleNamespace(
        url=url,
        title="Public source",
        text="Public evidence supports the answer.",
        fetched_at="2026-09-21T00:00:00+00:00",
        content_hash="a" * 64,
    )
    client = MagicMock()
    calls = []
    context = None

    async def provider(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _tool_response("fetch_web", {"url": url}, "fetch-1")
        evidence_id = next(iter(context.state.evidence))
        return _response(content=json.dumps({
            "final_answer": "The public source supports the answer.",
            "citation_ids": [evidence_id],
            "abstained": False,
        }))

    client.chat.completions.create = AsyncMock(side_effect=provider)
    context = _context(client, evidence_client=evidence_client, web_enabled=True)
    context.state.approved_web_urls = [url]
    with patch("services.knowledge_web.fetch_public_page", AsyncMock(return_value=page)):
        result = _run(context)

    assert result.source_citations[0].kind == "web_snapshot"
    assert calls[1]["tool_choice"] == "auto"
    assert calls[1]["response_format"] == {"type": "json_object"}


def test_generic_document_no_tool_content_remains_plain_text():
    response = _response(content="ordinary document answer")
    with patch("services.tool_loop.llm_chat", AsyncMock(return_value=response)) as provider:
        result = asyncio.run(autonomous._run_react_loop(
            messages=[{"role": "user", "content": "Explain"}],
            plan=[], steps=[], tools_called=[], user_id="learner", document_id=None,
            starting_round=0, run_id="ordinary", evidence_registry={},
            grounding_required=False, knowledge_context=None,
        ))

    assert result.final_answer == "ordinary document answer"
    assert "response_format" not in provider.await_args.kwargs
