#!/usr/bin/env python3
"""Recompute the published SciFact BM25 regression from user-supplied data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rank_bm25 import BM25Okapi  # noqa: E402
from services.bm25 import build_bm25_index, rank_bm25  # noqa: E402


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_scifact(
    data_dir: Path,
    manifest: dict,
) -> tuple[list[str], list[str], dict[str, str], dict[str, dict[str, int]]]:
    paths = {
        "corpus": data_dir / "corpus.jsonl",
        "queries": data_dir / "queries.jsonl",
        "test_qrels": data_dir / "qrels" / "test.tsv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing SciFact files: {missing}")
    for name, path in paths.items():
        expected = manifest["dataset"][f"{name}_sha256"]
        actual = file_digest(path)
        if actual != expected:
            raise SystemExit(
                f"SciFact {name} hash mismatch: expected {expected}, found {actual}"
            )

    corpus_rows = load_jsonl(paths["corpus"])
    query_rows = load_jsonl(paths["queries"])
    expected_documents = int(manifest["dataset"]["document_count"])
    if len(corpus_rows) != expected_documents:
        raise ValueError(
            f"expected {expected_documents} documents, found {len(corpus_rows)}"
        )

    doc_ids: list[str] = []
    documents: list[str] = []
    for row in corpus_rows:
        doc_id = str(row["_id"])
        text = "\n".join(
            part.strip()
            for part in (row.get("title", ""), row.get("text", ""))
            if part and part.strip()
        )
        if not text:
            raise ValueError(f"empty SciFact document: {doc_id}")
        doc_ids.append(doc_id)
        documents.append(text)

    all_queries = {
        str(row["_id"]): str(row.get("text") or row.get("title") or "").strip()
        for row in query_rows
    }
    qrels: dict[str, dict[str, int]] = {}
    with paths["test_qrels"].open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            qid = str(row["query-id"])
            qrels.setdefault(qid, {})[str(row["corpus-id"])] = int(row["score"])

    qids = sorted(qrels, key=lambda value: int(value) if value.isdigit() else value)
    expected_queries = int(manifest["dataset"]["test_query_count"])
    if len(qids) != expected_queries:
        raise ValueError(f"expected {expected_queries} queries, found {len(qids)}")
    if sum(len(values) for values in qrels.values()) != int(
        manifest["dataset"]["test_qrel_pairs"]
    ):
        raise ValueError("SciFact qrel-pair count differs from the manifest")

    queries = {qid: all_queries[qid] for qid in qids}
    if any(not query for query in queries.values()):
        raise ValueError("empty SciFact test query")
    missing_qrels = {
        doc_id for values in qrels.values() for doc_id in values
    } - set(doc_ids)
    if missing_qrels:
        raise ValueError(f"qrels reference missing corpus IDs: {sorted(missing_qrels)[:3]}")
    return doc_ids, documents, queries, qrels


def recall_at_k(qrels: Mapping[str, int], ranking: Sequence[str], k: int) -> float:
    relevant = {doc_id for doc_id, score in qrels.items() if score > 0}
    return len(relevant.intersection(ranking[:k])) / len(relevant) if relevant else 0.0


def reciprocal_rank_at_k(
    qrels: Mapping[str, int], ranking: Sequence[str], k: int
) -> float:
    for rank, doc_id in enumerate(ranking[:k], 1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(qrels: Mapping[str, int], ranking: Sequence[str], k: int) -> float:
    def gain(score: int, rank: int) -> float:
        return (2.0**score - 1.0) / math.log2(rank + 1.0)

    dcg = sum(
        gain(qrels.get(doc_id, 0), rank)
        for rank, doc_id in enumerate(ranking[:k], 1)
    )
    ideal_scores = sorted(
        (score for score in qrels.values() if score > 0), reverse=True
    )[:k]
    ideal = sum(gain(score, rank) for rank, score in enumerate(ideal_scores, 1))
    return dcg / ideal if ideal else 0.0


def query_metrics(qrels: Mapping[str, int], ranking: Sequence[str]) -> dict[str, float]:
    return {
        "ndcg@10": ndcg_at_k(qrels, ranking, 10),
        "mrr@10": reciprocal_rank_at_k(qrels, ranking, 10),
        "recall@5": recall_at_k(qrels, ranking, 5),
        "recall@10": recall_at_k(qrels, ranking, 10),
        "recall@100": recall_at_k(qrels, ranking, 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    published = HERE / "per_query_metrics.jsonl"
    if args.output.resolve() == published.resolve():
        raise SystemExit("refusing to overwrite the checked-in evidence artifact")
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    if args.zip_path is not None:
        if not args.zip_path.is_file():
            raise SystemExit(f"missing SciFact zip: {args.zip_path}")
        expected_md5 = manifest["dataset"]["official_zip_md5"]
        actual_md5 = file_digest(args.zip_path, "md5")
        if actual_md5 != expected_md5:
            raise SystemExit(
                f"SciFact zip MD5 mismatch: expected {expected_md5}, found {actual_md5}"
            )

    doc_ids, documents, queries, qrels = load_scifact(args.data_dir, manifest)
    legacy_index = BM25Okapi([list(document) for document in documents])
    current_index = build_bm25_index(documents)
    if current_index is None:
        raise ValueError("business tokenizer produced an empty corpus")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for number, (qid, query) in enumerate(queries.items(), 1):
            legacy_scores = legacy_index.get_scores(list(query))
            legacy_positions = sorted(
                range(len(legacy_scores)),
                key=lambda position: (-float(legacy_scores[position]), position),
            )[:100]
            current_positions = [
                position for position, _score in rank_bm25(current_index, query, 100)
            ]
            row = {
                "qid": qid,
                "baseline": query_metrics(
                    qrels[qid], [doc_ids[position] for position in legacy_positions]
                ),
                "current": query_metrics(
                    qrels[qid], [doc_ids[position] for position in current_positions]
                ),
            }
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )
            if number % 50 == 0:
                print(f"evaluated {number}/{len(queries)} queries", flush=True)

    print(f"wrote {len(queries)} query rows to {args.output}")
    print(
        "next: python evaluation/scifact_bm25/verify.py "
        f"--results {args.output}"
    )


if __name__ == "__main__":
    main()
