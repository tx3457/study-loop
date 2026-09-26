"""Offline checks of the MuSiQue benchmark's fixtures, scoring and decision rule."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "evaluate_multihop", Path(__file__).resolve().parent.parent / "scripts" / "evaluate_multihop.py"
)
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def test_frozen_sample_matches_its_manifest_and_protocol():
    loaded = bench.load_benchmark()
    assert len(loaded["questions"]) == 100
    assert {q["hop"] for q in loaded["questions"]} == {"2hop", "3hop", "4hop"}
    # Each question needs as many supporting paragraphs as it has hops.
    assert all(len(q["supporting_source_ids"]) == int(q["hop"][0]) for q in loaded["questions"])
    assert loaded["protocol"]["frozen_before_model_calls"] is True


def test_answer_scoring_follows_squad_normalisation_and_aliases():
    assert bench.exact_match("The Eiffel Tower!", ["eiffel tower"]) == 1.0
    assert bench.exact_match("Paris", ["London", "paris"]) == 1.0
    assert bench.exact_match("unknown", ["Paris"]) == 0.0
    assert bench.token_f1("the city of Paris", ["Paris"]) == pytest.approx(2 * (1 / 3) / (1 / 3 + 1))
    assert bench.token_f1("", ["Paris"]) == 0.0


def test_retrieval_metrics_count_supporting_paragraphs_in_the_top_k():
    ranked = ["a", "x", "b", "y", "c"]
    assert bench.recall_at(ranked, ["a", "b", "c"], 3) == pytest.approx(2 / 3)
    assert bench.full_support_at(ranked, ["a", "b", "c"], 3) == 0.0
    assert bench.full_support_at(ranked, ["a", "b", "c"], 5) == 1.0
    assert bench.dedupe(["a", "a", "", "b"]) == ["a", "b"]


def test_statistics_are_centred_and_exact():
    same = [0.2, 0.5, 0.9, 0.0] * 25
    assert bench.paired_bootstrap(same, same) == {
        "mean_difference": 0.0, "ci95_low": 0.0, "ci95_high": 0.0,
    }
    better = [x + 0.1 for x in same]
    interval = bench.paired_bootstrap(same, better)
    assert interval["ci95_low"] == pytest.approx(0.1) and interval["ci95_high"] == pytest.approx(0.1)
    # 8 discordant pairs all one way: two-sided exact p = 2 / 2**8.
    result = bench.mcnemar_exact([0] * 8 + [1] * 2, [1] * 8 + [1] * 2)
    assert result == {"only_baseline": 0, "only_graph": 8, "p_value": pytest.approx(2 / 256)}


@pytest.mark.parametrize(
    ("mean", "low", "high", "expected"),
    [
        (0.06, 0.01, 0.11, "invest"),
        (0.06, -0.01, 0.13, "inconclusive"),  # large but not distinguishable from zero
        (0.03, 0.005, 0.055, "inconclusive"),  # real but below the practical threshold
        (-0.02, -0.07, 0.019, "shrink"),
        (0.0, -0.03, 0.03, "inconclusive"),
    ],
)
def test_decision_rule_matches_the_frozen_protocol(mean, low, high, expected):
    comparison = {"mean_difference": mean, "ci95_low": low, "ci95_high": high}
    assert bench.decide(comparison, bench.load_benchmark()["protocol"]["decision_rule"]) == expected


def test_holdout_sample_is_frozen_and_disjoint_from_the_tuning_sample():
    holdout = bench.load_benchmark(root=bench.REPO_ROOT / "evaluation" / "musique_multihop_holdout")
    tuning = bench.load_benchmark()
    assert len(holdout["questions"]) == 100
    assert not {q["question_id"] for q in holdout["questions"]} & {
        q["question_id"] for q in tuning["questions"]
    }
    assert holdout["protocol"]["frozen_before_model_calls"] is True


def test_weight_selection_follows_the_frozen_tie_breaks():
    def metrics(**means):
        return {w: {"mean": {"full_support@10": fs, "recall@10": r}} for w, (fs, r) in means.items()}

    # Highest full support wins outright.
    assert bench.select_weight(metrics(**{"1:1": (0.25, 0.6), "2:1": (0.30, 0.5)})) == "2:1"
    # Equal full support: recall decides.
    assert bench.select_weight(metrics(**{"1:1": (0.3, 0.6), "3:1": (0.3, 0.7)})) == "3:1"
    # Fully tied: the weight closest to 1:1, so no change when nothing is better.
    assert bench.select_weight(metrics(**{"4:1": (0.3, 0.7), "1:1": (0.3, 0.7)})) == "1:1"
    assert bench.select_weight(metrics(**{"4:1": (0.3, 0.7), "1.5:1": (0.3, 0.7)})) == "1.5:1"
