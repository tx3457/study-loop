#!/usr/bin/env python3
"""Verify the published SciFact BM25 metric artifact without dataset access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Mapping, Sequence


METRICS = ("ndcg@10", "mrr@10", "recall@5", "recall@10", "recall@100")
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_MANIFEST = HERE / "manifest.json"
DEFAULT_RESULTS = HERE / "per_query_metrics.jsonl"


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate percentile of an empty sequence")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def paired_bootstrap(
    baseline: Mapping[str, float],
    current: Mapping[str, float],
    *,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    qids = sorted(set(baseline).intersection(current))
    deltas = [current[qid] - baseline[qid] for qid in qids]
    rng = random.Random(seed)
    sampled = [
        statistics.fmean(deltas[rng.randrange(len(deltas))] for _ in deltas)
        for _ in range(samples)
    ]
    return {
        "n_pairs": len(deltas),
        "samples": samples,
        "seed": seed,
        "mean_delta": statistics.fmean(deltas),
        "ci95_low": percentile(sampled, 0.025),
        "ci95_high": percentile(sampled, 0.975),
    }


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}: expected {expected!r}, found {actual!r}")


# Behavioural invariants available to invariant-bound sources. The manifest says
# WHICH invariants a file must satisfy; this table says HOW each one is checked.
# A manifest entry naming an invariant that is absent here fails verification, so
# a declaration can never silently go unenforced.
INVARIANT_CHECKS = {
    "vectorstore imports pure BM25 helpers": (
        "services/vectorstore.py",
        lambda text: "from services.bm25 import build_bm25_index, rank_bm25" in text,
    ),
    "legacy query character split is absent": (
        "services/vectorstore.py",
        lambda text: "get_scores(list(query))" not in text,
    ),
}


def verify_sources(manifest: dict) -> None:
    """Check the two source-binding tiers described by studyloop_source.binding_policy.

    Tier 1 (`files`) is hash-pinned: the evaluated ranking path is reproduced from
    those bytes, so any change invalidates the published numbers.

    Tier 2 (`invariant_bound_files`) is not hash-pinned. run.py never imports those
    modules, so their contents cannot move a metric; they are recorded to evidence
    that production retrieval uses the same pure BM25 helpers. Pinning a digest
    there bound the experiment to a file under active development and produced
    failures that said nothing about the measurement.
    """
    source = manifest["studyloop_source"]
    pinned = source["files"]
    invariant_bound = source.get("invariant_bound_files", {})

    pinned_paths = {entry["path"] for entry in pinned.values()}
    invariant_paths = {entry["path"] for entry in invariant_bound.values()}
    both = pinned_paths & invariant_paths
    if both:
        raise ValueError(
            f"sources cannot be both hash-pinned and invariant-bound: {sorted(both)}"
        )

    for name, entry in pinned.items():
        path = REPO_ROOT / entry["path"]
        if not path.is_file():
            raise ValueError(f"missing recorded source {name}: {entry['path']}")
        actual = file_digest(path)
        if actual != entry["sha256"]:
            raise ValueError(
                f"source hash mismatch for {entry['path']}: "
                f"expected {entry['sha256']}, found {actual}"
            )

    bm25_text = (REPO_ROOT / "services/bm25.py").read_text(encoding="utf-8")
    checks = {
        "bm25 imports the shared tokenizer": (
            "from services.tokenization import tokenize_for_bm25" in bm25_text
        ),
        "BM25 documents use the shared tokenizer": (
            "[tokenize_for_bm25(document) for document in documents]" in bm25_text
        ),
        "BM25 queries use the shared tokenizer": (
            "query_tokens = tokenize_for_bm25(query)" in bm25_text
        ),
    }

    for name, entry in invariant_bound.items():
        path = REPO_ROOT / entry["path"]
        if not path.is_file():
            raise ValueError(f"missing recorded source {name}: {entry['path']}")
        declared = entry.get("invariants") or []
        if not declared:
            raise ValueError(
                f"invariant-bound source {name} declares no invariants; it would be "
                "recorded without being checked at all"
            )
        text = path.read_text(encoding="utf-8")
        for label in declared:
            check = INVARIANT_CHECKS.get(label)
            if check is None:
                raise ValueError(
                    f"manifest declares invariant {label!r} for {name}, but verify.py "
                    "implements no check for it"
                )
            expected_path, predicate = check
            if expected_path != entry["path"]:
                raise ValueError(
                    f"invariant {label!r} is defined for {expected_path}, "
                    f"not {entry['path']}"
                )
            checks[label] = predicate(text)

    failed = [label for label, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"business-source binding checks failed: {failed}")


def verify_rows(rows: list[dict], manifest: dict) -> dict:
    expected_count = manifest["dataset"]["test_query_count"]
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} rows, found {len(rows)}")

    qids: set[str] = set()
    by_arm: dict[str, dict[str, dict[str, float]]] = {
        "baseline": {},
        "current": {},
    }
    for index, row in enumerate(rows, 1):
        if set(row) != {"qid", "baseline", "current"}:
            raise ValueError(f"row {index} has unexpected fields: {sorted(row)}")
        qid = str(row["qid"])
        if not qid or qid in qids:
            raise ValueError(f"row {index} has empty or duplicate qid: {qid!r}")
        qids.add(qid)
        for arm in ("baseline", "current"):
            values = row[arm]
            if set(values) != set(METRICS):
                raise ValueError(f"row {index} {arm} has unexpected metrics")
            normalized: dict[str, float] = {}
            for metric in METRICS:
                value = float(values[metric])
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"row {index} {arm}.{metric} is outside [0, 1]")
                normalized[metric] = value
            by_arm[arm][qid] = normalized

    ordered_qids = sorted(qids)
    aggregate: dict[str, dict[str, float]] = {"baseline": {}, "current": {}}
    for arm in aggregate:
        for metric in METRICS:
            value = statistics.fmean(
                by_arm[arm][qid][metric] for qid in ordered_qids
            )
            expected = float(manifest["results"][arm]["metrics"][metric])
            assert_close(value, expected, f"{arm}.{metric}")
            aggregate[arm][metric] = value

    protocol = manifest["protocol"]
    for metric in METRICS:
        actual = paired_bootstrap(
            {qid: by_arm["baseline"][qid][metric] for qid in qids},
            {qid: by_arm["current"][qid][metric] for qid in qids},
            seed=int(protocol["bootstrap_seed"]),
            samples=int(protocol["bootstrap_samples"]),
        )
        expected = manifest["results"]["paired_bootstrap_current_minus_baseline"][metric]
        for field in ("n_pairs", "samples", "seed"):
            if actual[field] != expected[field]:
                raise ValueError(
                    f"bootstrap {metric}.{field}: expected {expected[field]}, "
                    f"found {actual[field]}"
                )
        for field in ("mean_delta", "ci95_low", "ci95_high"):
            assert_close(
                float(actual[field]),
                float(expected[field]),
                f"bootstrap {metric}.{field}",
            )

    return aggregate


def compare_to_published(rows: list[dict]) -> None:
    published = {str(row["qid"]): row for row in load_rows(DEFAULT_RESULTS)}
    supplied = {str(row["qid"]): row for row in rows}
    if set(published) != set(supplied):
        raise ValueError("supplied result query IDs differ from the published artifact")
    for qid in published:
        for arm in ("baseline", "current"):
            for metric in METRICS:
                assert_close(
                    float(supplied[qid][arm][metric]),
                    float(published[qid][arm][metric]),
                    f"query {qid} {arm}.{metric}",
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    verify_sources(manifest)

    if args.results.resolve() == DEFAULT_RESULTS.resolve():
        expected_hash = manifest["artifacts"]["per_query_metrics"]["sha256"]
        actual_hash = file_digest(args.results)
        if actual_hash != expected_hash:
            raise ValueError(
                f"published result hash mismatch: expected {expected_hash}, found {actual_hash}"
            )

    runner = manifest["artifacts"]["runner"]
    runner_path = REPO_ROOT / runner["path"]
    if file_digest(runner_path) != runner["sha256"]:
        raise ValueError("runner hash does not match manifest")

    rows = load_rows(args.results)
    aggregate = verify_rows(rows, manifest)
    if args.results.resolve() != DEFAULT_RESULTS.resolve():
        compare_to_published(rows)

    print(
        "PASS "
        f"queries={len(rows)} "
        f"baseline_ndcg@10={aggregate['baseline']['ndcg@10']:.6f} "
        f"current_ndcg@10={aggregate['current']['ndcg@10']:.6f} "
        "scope=english_bm25_component_only"
    )


if __name__ == "__main__":
    main()
