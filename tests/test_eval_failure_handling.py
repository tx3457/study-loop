"""Regression tests for fail-closed LLM-as-Judge aggregation."""

import asyncio

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import services.eval as eval_service
from models.eval import ABConfig, JudgeScore, JudgeVerdict


def _valid_verdict(
    *,
    relevance: int = 4,
    clarity: int = 5,
    covers_weak_point: bool = True,
    faithfulness: bool = True,
    difficulty: str = "medium",
) -> JudgeVerdict:
    return JudgeVerdict(
        relevance=relevance,
        clarity=clarity,
        difficulty_feel=difficulty,
        covers_weak_point=covers_weak_point,
        matched_point="RAG" if covers_weak_point else "无",
        faithfulness=faithfulness,
        reasoning="valid score",
    )


def _error_verdict(error_type: str = "RuntimeError") -> JudgeVerdict:
    return JudgeVerdict(
        status="error",
        reasoning="judge failed",
        error_type=error_type,
        error_message="provider unavailable",
    )


def test_judge_exception_returns_explicit_error_without_fake_scores():
    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(
        side_effect=RuntimeError("provider unavailable")
    )

    with patch.object(eval_service, "_client", client):
        verdict = asyncio.run(eval_service.judge_question("q", "a", "source", []))

    assert verdict.status == "error"
    assert verdict.error_type == "RuntimeError"
    assert verdict.faithfulness is None
    assert verdict.relevance is None


def test_successful_judge_call_preserves_flat_valid_verdict():
    score = JudgeScore(
        relevance=5,
        clarity=4,
        difficulty_feel="medium",
        covers_weak_point=True,
        matched_point="RAG",
        faithfulness=True,
        reasoning="supported by source",
    )
    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=score))]
        )
    )

    with patch.object(eval_service, "_client", client):
        verdict = asyncio.run(
            eval_service.judge_question("q", "a", "source", ["RAG"])
        )

    assert verdict.status == "valid"
    assert verdict.relevance == 5
    assert verdict.faithfulness is True
    assert verdict.error_type is None


def test_all_judge_failures_cannot_produce_positive_quality_metrics():
    metrics = eval_service.aggregate_metrics([_error_verdict(), _error_verdict()])

    assert metrics.total_count == 2
    assert metrics.valid_count == 0
    assert metrics.failed_count == 2
    assert metrics.judge_success_rate == 0.0
    assert metrics.faithfulness_rate == 0.0
    assert metrics.weak_point_coverage == 0.0
    assert metrics.avg_relevance == 0.0
    assert metrics.difficulty_dist == {}


def test_partial_failure_uses_only_valid_samples_as_denominator():
    metrics = eval_service.aggregate_metrics([
        _valid_verdict(relevance=5, clarity=4, faithfulness=True),
        _error_verdict("TimeoutError"),
    ])

    assert metrics.total_count == 2
    assert metrics.valid_count == 1
    assert metrics.failed_count == 1
    assert metrics.judge_success_rate == 0.5
    assert metrics.faithfulness_rate == 1.0
    assert metrics.weak_point_coverage == 1.0
    assert metrics.avg_relevance == 5.0
    assert metrics.avg_clarity == 4.0
    assert metrics.difficulty_dist == {"medium": 1}


def test_normal_valid_aggregation_preserves_existing_metric_semantics():
    metrics = eval_service.aggregate_metrics([
        _valid_verdict(
            relevance=5,
            clarity=4,
            covers_weak_point=True,
            faithfulness=True,
            difficulty="easy",
        ),
        _valid_verdict(
            relevance=3,
            clarity=2,
            covers_weak_point=False,
            faithfulness=False,
            difficulty="hard",
        ),
    ])

    assert metrics.total_count == 2
    assert metrics.valid_count == 2
    assert metrics.failed_count == 0
    assert metrics.judge_success_rate == 1.0
    assert metrics.weak_point_coverage == 0.5
    assert metrics.avg_relevance == 4.0
    assert metrics.avg_clarity == 3.0
    assert metrics.faithfulness_rate == 0.5
    assert metrics.difficulty_dist == {"easy": 1, "hard": 1}


def test_ab_quality_delta_is_not_reported_when_an_arm_has_no_valid_judges():
    result = eval_service._build_result(
        config=ABConfig(document_id="doc", experiment="ce"),
        baseline_label="baseline",
        treatment_label="treatment",
        baseline_verdicts=[_error_verdict()],
        treatment_verdicts=[_valid_verdict()],
    )

    assert result.delta["faithfulness_rate"] is None
    assert result.delta["avg_relevance"] is None
    assert result.delta["judge_success_rate"] == 1.0
