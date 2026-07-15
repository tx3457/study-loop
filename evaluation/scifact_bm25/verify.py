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


def verify_sources(manifest: dict) -> None:
    for name, source in manifest["studyloop_source"]["files"].items():
        path = REPO_ROOT / source["path"]
        if not path.is_file():
            raise ValueError(f"missing recorded source {name}: {source['path']}")
        actual = file_digest(path)
        if actual != source["sha256"]:
            raise ValueError(
                f"source hash mismatch for {source['path']}: "
                f"expected {source['sha256']}, found {actual}"
            )

    bm25_text = (REPO_ROOT / "services/bm25.py").read_text(encoding="utf-8")
    vectorstore_text = (REPO_ROOT / "services/vectorstore.py").read_text(encoding="utf-8")
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
        "vectorstore imports pure BM25 helpers": (
            "from services.bm25 import build_bm25_index, rank_bm25" in vectorstore_text
        ),
        "legacy query character split is absent": (
            "get_scores(list(query))" not in vectorstore_text
        ),
    }
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
