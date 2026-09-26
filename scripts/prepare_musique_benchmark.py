"""Freeze the MuSiQue multi-hop sample used to decide GraphRAG's future.

Run once, before any model call:

    python scripts/prepare_musique_benchmark.py path/to/musique_ans_v1.0_dev.jsonl

The source file is pinned by SHA-256. The sample is a seeded, hop-stratified
draw; the corpus is every paragraph (supporting and distractor) of the sampled
questions, pooled into one knowledge base so distractors of one question compete
with evidence of another. Writes evaluation/musique_multihop/{questions,corpus,
manifest}.json; protocol.json is written by hand and frozen alongside.

MuSiQue is licensed CC BY 4.0 (Trivedi et al., TACL 2022,
https://github.com/StonyBrookNLP/musique).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "evaluation" / "musique_multihop"

SOURCE_SHA256 = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"
SEED = 20260925
# Deeper chains are over-weighted relative to the dev set (52/31/17%) so each hop
# group is large enough to report on its own.
STRATA = {"2hop": 40, "3hop": 30, "4hop": 30}


def source_id(title: str, text: str) -> str:
    return "p_" + hashlib.sha256(f"{title}\n{text}".encode()).hexdigest()[:16]


def document_text(title: str, text: str) -> str:
    # What both retrieval methods index: the title carries the entity name that
    # multi-hop questions pivot on.
    return f"{title}\n{text}"


def build(
    rows: list[dict], seed: int = SEED, exclude: frozenset[str] = frozenset()
) -> tuple[list[dict], list[dict]]:
    by_stratum: dict[str, list[dict]] = {name: [] for name in STRATA}
    for row in rows:
        if not row["answerable"] or row["id"] in exclude:
            continue
        by_stratum[row["id"].split("__")[0][:4]].append(row)
    rng = random.Random(seed)
    selected: list[dict] = []
    for name, size in STRATA.items():
        pool = sorted(by_stratum[name], key=lambda row: row["id"])
        selected.extend(rng.sample(pool, size))

    corpus: dict[str, dict] = {}
    questions = []
    for row in selected:
        supporting = []
        for paragraph in row["paragraphs"]:
            pid = source_id(paragraph["title"], paragraph["paragraph_text"])
            corpus.setdefault(
                pid,
                {
                    "source_id": pid,
                    "title": paragraph["title"],
                    "text": document_text(paragraph["title"], paragraph["paragraph_text"]),
                },
            )
            if paragraph["is_supporting"]:
                supporting.append(pid)
        questions.append(
            {
                "question_id": row["id"],
                "hop": row["id"].split("__")[0][:4],
                "question": row["question"],
                "answer": row["answer"],
                "answer_aliases": row.get("answer_aliases") or [],
                "supporting_source_ids": list(dict.fromkeys(supporting)),
            }
        )
    return questions, sorted(corpus.values(), key=lambda item: item["source_id"])


def _write(path: Path, payload) -> str:
    data = json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    path.write_text(data, encoding="utf-8")
    return hashlib.sha256(data.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="musique_ans_v1.0_dev.jsonl")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--benchmark-id", default="musique_multihop_v1")
    parser.add_argument(
        "--exclude", type=Path, action="append", default=[],
        help="questions.json of a sample this one must not overlap (a held-out set)",
    )
    args = parser.parse_args()
    raw = args.source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != SOURCE_SHA256:
        print(f"source SHA-256 {actual} does not match the pinned {SOURCE_SHA256}", file=sys.stderr)
        return 2
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    excluded = frozenset(
        q["question_id"] for path in args.exclude for q in json.loads(path.read_text())
    )
    questions, corpus = build(rows, args.seed, excluded)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmark_id": args.benchmark_id,
        "source": {
            "dataset": "MuSiQue-Ans v1.0 dev",
            "file": "musique_ans_v1.0_dev.jsonl",
            "sha256": SOURCE_SHA256,
            "license": "CC BY 4.0",
            "citation": "Trivedi et al., MuSiQue: Multihop Questions via Single-hop "
                        "Question Composition, TACL 2022",
        },
        "seed": args.seed,
        "strata": STRATA,
        **({"excluded_question_count": len(excluded)} if excluded else {}),
        "question_count": len(questions),
        "corpus_size": len(corpus),
        "files": {
            "questions.json": _write(output / "questions.json", questions),
            "corpus.json": _write(output / "corpus.json", corpus),
        },
    }
    _write(output / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("question_count", "corpus_size", "strata")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
