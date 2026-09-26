"""MuSiQue multi-hop benchmark: existing hybrid retrieval versus GraphRAG (LightRAG).

The protocol, including the decision rule, is frozen in
evaluation/musique_multihop/protocol.json before any model call. Each phase runs
in the environment its method runs in production, because the main app and the
knowledge service pin different openai versions; phases hand over JSON files in
one state directory and every live phase resumes where it stopped.

    python scripts/evaluate_multihop.py validate                     # any env, offline
    python scripts/evaluate_multihop.py retrieve-baseline --state S  # main-app env
    python scripts/evaluate_multihop.py retrieve-graph --state S     # knowledge env
    python scripts/evaluate_multihop.py answer --state S             # knowledge env
    python scripts/evaluate_multihop.py score --state S              # any env, offline

The fusion-weight study (evaluation/musique_multihop_holdout/protocol.json) sweeps
the main app's RRF weights on the tuning sample and judges the pick on a held-out one:

    python scripts/evaluate_multihop.py sweep-baseline --state T                      # main-app env
    python scripts/evaluate_multihop.py sweep-baseline --state H \
        --benchmark evaluation/musique_multihop_holdout                               # main-app env
    python scripts/evaluate_multihop.py sweep-score --state T --holdout-state H \
        --benchmark evaluation/musique_multihop_holdout                               # offline

Live phases need KNOWLEDGE_EVAL_DATABASE_URL (a disposable pgvector database)
for the graph phase and the provider keys from .env. --limit N runs only the
first N questions and their paragraphs, for a smoke run.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import string
import sys
import time
from collections import Counter
from math import comb
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = REPO_ROOT / "evaluation" / "musique_multihop"
SEED = 20260925
K_VALUES = (5, 10, 20)
ANSWER_CONTEXT = 10
ARMS = ("hybrid_rrf", "lightrag_mix", "lightrag_naive", "lightrag_mix_with_graph_facts")
GRAPH_WORKSPACE = "musique_multihop_v1"


# ---------------------------------------------------------------- fixtures


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_benchmark(limit: int | None = None, root: Path = BENCHMARK) -> dict[str, Any]:
    """Load the frozen sample after checking it against its manifest."""
    manifest = _read(root / "manifest.json")
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"{name} changed after freezing: {actual} != {expected}")
    questions = _read(root / "questions.json")
    corpus = _read(root / "corpus.json")
    by_id = {item["source_id"]: item for item in corpus}
    for question in questions:
        missing = set(question["supporting_source_ids"]) - set(by_id)
        if missing:
            raise SystemExit(f"{question['question_id']} cites absent paragraphs {missing}")
    if limit is not None:
        questions = questions[:limit]
        # A smoke corpus keeps every paragraph of the chosen questions only; the
        # MuSiQue question file lists them through supporting IDs plus distractors,
        # which the frozen corpus no longer groups, so take the supporting ones and
        # a deterministic slice of the rest.
        keep = {sid for q in questions for sid in q["supporting_source_ids"]}
        others = sorted(set(by_id) - keep)
        keep |= set(others[: 18 * len(questions)])
        corpus = [item for item in corpus if item["source_id"] in keep]
    return {"manifest": manifest, "questions": questions, "corpus": corpus,
            "protocol": _read(root / "protocol.json")}


def _write(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)


# ---------------------------------------------------------------- scoring


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, golds: list[str]) -> float:
    return float(any(normalize_answer(prediction) == normalize_answer(g) for g in golds))


def token_f1(prediction: str, golds: list[str]) -> float:
    best = 0.0
    predicted = normalize_answer(prediction).split()
    for gold in golds:
        expected = normalize_answer(gold).split()
        common = Counter(predicted) & Counter(expected)
        overlap = sum(common.values())
        if not predicted or not expected:
            score = float(predicted == expected)
        elif overlap == 0:
            score = 0.0
        else:
            precision, recall = overlap / len(predicted), overlap / len(expected)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


def recall_at(ranked: list[str], supporting: list[str], k: int) -> float:
    return len(set(ranked[:k]) & set(supporting)) / len(supporting)


def full_support_at(ranked: list[str], supporting: list[str], k: int) -> float:
    return float(set(supporting) <= set(ranked[:k]))


def paired_bootstrap(a: list[float], b: list[float], resamples: int = 10_000) -> dict[str, float]:
    """95% percentile interval of mean(b - a) over resampled questions."""
    differences = [y - x for x, y in zip(a, b, strict=True)]
    n = len(differences)
    rng = random.Random(SEED)
    means = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    return {
        "mean_difference": sum(differences) / n,
        "ci95_low": means[int(0.025 * resamples)],
        "ci95_high": means[int(0.975 * resamples) - 1],
    }


def mcnemar_exact(a: list[float], b: list[float]) -> dict[str, Any]:
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = only_a + only_b
    if n == 0:
        return {"only_baseline": 0, "only_graph": 0, "p_value": 1.0}
    tail = sum(comb(n, i) for i in range(0, min(only_a, only_b) + 1)) / 2**n
    return {"only_baseline": only_a, "only_graph": only_b, "p_value": min(1.0, 2 * tail)}


def decide(f1_comparison: dict[str, float], rule: dict[str, Any]) -> str:
    """Apply the frozen decision rule (F1 in points, i.e. scores times 100)."""
    mean = f1_comparison["mean_difference"] * 100
    low = f1_comparison["ci95_low"] * 100
    high = f1_comparison["ci95_high"] * 100
    if mean >= 5 and low > 0:
        return "invest"
    if high < 2:
        return "shrink"
    return "inconclusive"


# ---------------------------------------------------------------- baseline


async def _baseline_setup(state: Path, bench: dict[str, Any]):
    _load_env()
    os.environ["LLM_EMBEDDING_MODEL"] = os.environ.get("KNOWLEDGE_EMBEDDING_MODEL", "BAAI/bge-m3")
    os.environ["CHROMA_DIR"] = str(state / "chroma")
    os.environ["RERANKER_ENABLED"] = "false"
    sys.path.insert(0, str(REPO_ROOT))
    from services import vectorstore

    owner, document_id = "musique-eval", "musique-multihop-v1"
    text_to_source = {item["text"]: item["source_id"] for item in bench["corpus"]}
    marker = state / "baseline_ingested.json"
    if not marker.exists():
        started = time.perf_counter()
        await vectorstore.deal_document(
            document_id, "musique.txt", [item["text"] for item in bench["corpus"]],
            owner_id=owner,
        )
        _write(marker, {"seconds": time.perf_counter() - started,
                        "embedding_model": os.environ["LLM_EMBEDDING_MODEL"],
                        "paragraphs": len(bench["corpus"])})
    return vectorstore, text_to_source, owner, document_id


async def _baseline_ranked(vectorstore, text_to_source, owner, document_id, question) -> list[str]:
    raw = await vectorstore.hybrid_query_document(
        document_id, question["question"], n_results=max(K_VALUES),
        enable_rerank=False, owner_id=owner,
    )
    texts = list(raw.get("documents", [[]])[0])
    ranked = dedupe([text_to_source.get(text, "") for text in texts])
    if len(ranked) != len(texts):
        raise RuntimeError("baseline returned text outside the frozen corpus")
    return ranked


async def _retrieve_baseline(state: Path, bench: dict[str, Any]) -> None:
    vectorstore, text_to_source, owner, document_id = await _baseline_setup(state, bench)
    rows = []
    for question in bench["questions"]:
        started = time.perf_counter()
        ranked = await _baseline_ranked(vectorstore, text_to_source, owner, document_id, question)
        latency = (time.perf_counter() - started) * 1000
        rows.append({"question_id": question["question_id"], "ranked": ranked,
                     "latency_ms": latency})
    _write(state / "retrieval_hybrid_rrf.json", {"embedding_model": os.environ["LLM_EMBEDDING_MODEL"],
                                                 "questions": rows})


async def _sweep_baseline(state: Path, bench: dict[str, Any], weights: list[str]) -> None:
    setup = await _baseline_setup(state, bench)
    sweep: dict[str, Any] = {"embedding_model": os.environ["LLM_EMBEDDING_MODEL"], "weights": {}}
    for weight in weights:
        dense, bm25 = weight.split(":")
        # hybrid_query_document reads the weights on every call.
        os.environ["HYBRID_DENSE_WEIGHT"], os.environ["HYBRID_BM25_WEIGHT"] = dense, bm25
        sweep["weights"][weight] = [
            {"question_id": q["question_id"], "ranked": await _baseline_ranked(*setup, q)}
            for q in bench["questions"]
        ]
        print(f"swept {weight}", flush=True)
    _write(state / "sweep_baseline.json", sweep)


def _weight_distance(weight: str) -> float:
    dense, bm25 = (float(part) for part in weight.split(":"))
    return abs(dense / (dense + bm25) - 0.5)


def _sweep_metrics(state: Path, bench: dict[str, Any]) -> dict[str, dict[str, Any]]:
    supporting = {q["question_id"]: q["supporting_source_ids"] for q in bench["questions"]}
    result = {}
    for weight, rows in _read(state / "sweep_baseline.json")["weights"].items():
        per = {k: [] for k in ("recall@5", "recall@10", "recall@20", "full_support@10")}
        for row in rows:
            gold = supporting[row["question_id"]]
            for k in (5, 10, 20):
                per[f"recall@{k}"].append(recall_at(row["ranked"], gold, k))
            per["full_support@10"].append(full_support_at(row["ranked"], gold, 10))
        result[weight] = {"columns": per,
                          "mean": {key: sum(v) / len(v) for key, v in per.items()}}
    return result


def select_weight(metrics: dict[str, dict[str, Any]]) -> str:
    """Frozen selection rule: full_support@10, then recall@10, then closest to 1:1."""
    return max(metrics, key=lambda w: (metrics[w]["mean"]["full_support@10"],
                                       metrics[w]["mean"]["recall@10"],
                                       -_weight_distance(w)))


def sweep_score(tuning: Path, holdout: Path, holdout_bench: dict[str, Any]) -> dict[str, Any]:
    tuning_metrics = _sweep_metrics(tuning, load_benchmark())
    holdout_metrics = _sweep_metrics(holdout, holdout_bench)
    chosen = select_weight(tuning_metrics)
    comparison = None
    decision = "no_change"
    if chosen != "1:1":
        base, pick = holdout_metrics["1:1"]["columns"], holdout_metrics[chosen]["columns"]
        comparison = {
            "recall@10": paired_bootstrap(base["recall@10"], pick["recall@10"]),
            "full_support@10": {
                # Named for this comparison; mcnemar_exact's keys name the GraphRAG one.
                ("only_1_1" if key == "only_baseline" else
                 "only_selected" if key == "only_graph" else key): value
                for key, value in mcnemar_exact(
                    base["full_support@10"], pick["full_support@10"]).items()
            },
        }
        if comparison["recall@10"]["ci95_low"] > 0:
            decision = "candidate"
    result = {
        "tuning": {w: m["mean"] for w, m in tuning_metrics.items()},
        "holdout": {w: m["mean"] for w, m in holdout_metrics.items()},
        "selected_on_tuning": chosen,
        "holdout_comparison_vs_1_1": comparison,
        "decision": decision,
    }
    _write(holdout / "sweep_results.json", result)
    return result


# ---------------------------------------------------------------- graph


async def _retrieve_graph(state: Path, bench: dict[str, Any], parallel: int) -> None:
    _load_env()
    database_url = os.environ.get("KNOWLEDGE_EVAL_DATABASE_URL", "").strip()
    if not database_url:
        raise SystemExit("KNOWLEDGE_EVAL_DATABASE_URL must name a disposable pgvector database")
    os.environ["KNOWLEDGE_DATABASE_URL"] = database_url
    os.environ.setdefault("KNOWLEDGE_SERVICE_TOKEN", "evaluation-only-" + "x" * 32)
    sys.path.insert(0, str(REPO_ROOT))
    from dataclasses import replace

    from lightrag import QueryParam
    from lightrag.utils import TokenTracker

    from graph_service.config import Settings
    from graph_service.engine import LightRAGEngine

    settings = replace(
        Settings.from_env(),
        materials_dir=state / "materials",
        working_dir=state / "lightrag",
        llm_max_async=parallel,
    )
    probe = LightRAGEngine(settings)
    production_llm, embedding = probe._provider_callbacks()
    trackers = {"index": TokenTracker(), "query": TokenTracker()}
    calls = Counter()
    phase = {"name": "index"}

    async def tracked_llm(prompt: str, **kwargs):
        calls[phase["name"]] += 1
        return await production_llm(prompt, token_tracker=trackers[phase["name"]], **kwargs)

    engine = LightRAGEngine(settings, llm_func=tracked_llm, embedding_func=embedding)
    usage_path = state / "graph_usage.json"
    previous = _read(usage_path) if usage_path.exists() else {}
    started = time.perf_counter()
    session: dict[str, Any] = {"llm_model": settings.llm_model,
                               "embedding_model": settings.embedding_model,
                               "llm_max_async": parallel}

    def save_usage() -> None:
        # After every batch, so a killed session still leaves its usage behind.
        session["index_seconds_this_session"] = time.perf_counter() - started
        _write(usage_path, _merge_usage(previous, {
            name: {**tracker.get_usage(), "llm_calls": calls[name]}
            for name, tracker in trackers.items()
        }, session))

    try:
        session["failed_documents"] = await _ingest_graph(
            engine, bench["corpus"], parallel, save_usage
        )
        save_usage()
        if session["failed_documents"]:
            raise SystemExit(
                f"{len(session['failed_documents'])} documents are not indexed; "
                "rerun this phase to retry them before querying"
            )

        phase["name"] = "query"
        rows = []
        for question in bench["questions"]:
            # Own clock: `started` times the whole session for save_usage.
            query_started = time.perf_counter()
            mix = await engine.query(GRAPH_WORKSPACE, question["question"])
            mix_ms = (time.perf_counter() - query_started) * 1000
            query_started = time.perf_counter()
            naive = await _naive(engine, question["question"], QueryParam)
            naive_ms = (time.perf_counter() - query_started) * 1000
            rows.append({
                "question_id": question["question_id"],
                "lightrag_mix": {"ranked": dedupe([c.get("full_doc_id", "") for c in mix["chunks"]]),
                                 "latency_ms": mix_ms,
                                 "entities": mix["entities"][:20],
                                 "relationships": mix["relationships"][:20]},
                "lightrag_naive": {"ranked": dedupe(naive), "latency_ms": naive_ms},
            })
        _write(state / "retrieval_graph.json", {"questions": rows})
    finally:
        save_usage()
        await engine.close()
        await probe.close()


def _merge_usage(previous: dict, usage: dict, extra: dict) -> dict:
    # Sessions of a resumed ingest add up; call counts without provider usage stay
    # visible as the gap between llm_calls and call_count.
    merged = {}
    for name, current in usage.items():
        before = previous.get(name, {})
        merged[name] = {key: before.get(key, 0) + value for key, value in current.items()}
    sessions = previous.get("sessions", []) + [extra]
    return {**merged, "sessions": sessions,
            "usage_complete": previous.get("usage_complete", True)}


async def _ingest_graph(engine, corpus: list[dict], parallel: int, checkpoint) -> list[str]:
    ids = [item["source_id"] for item in corpus]
    async with engine._use(GRAPH_WORKSPACE) as rag:
        # Documents processed concurrently; each paragraph is one chunk, so this and
        # llm_max_async set throughput without changing what is extracted.
        rag.max_parallel_insert = parallel
        for attempt in range(3):
            statuses = await rag.aget_docs_by_ids(ids)
            # The SDK's automatic sweep resumes interrupted documents but never
            # FAILED ones (those reset only through its manual exclusive reset), so
            # a failed document is deleted and inserted again as new. It failed
            # before its graph writes, so deleting it removes nothing it owned.
            failed = [i for i in ids if _status(statuses.get(i)) == "failed"]
            for source in failed:
                await rag.adelete_by_doc_id(source)
            unseen = [item for item in corpus
                      if statuses.get(item["source_id"]) is None or item["source_id"] in failed]
            unfinished = [i for i in ids if statuses.get(i) is not None
                          and _status(statuses[i]) not in {"processed", "failed"}]
            if not unseen and not unfinished:
                return []
            print(f"graph ingest attempt {attempt + 1}: {len(unseen)} new "
                  f"({len(failed)} of them failed before), {len(unfinished)} to resume",
                  flush=True)
            if unfinished:
                # Pending and interrupted documents: re-inserting their ids would be
                # skipped as duplicates, so run the SDK's queue instead.
                await rag.apipeline_process_enqueue_documents()
            for start in range(0, len(unseen), 50):
                batch = unseen[start:start + 50]
                await rag.ainsert(
                    [item["text"] for item in batch],
                    ids=[item["source_id"] for item in batch],
                    file_paths=[f"musique_{item['source_id']}" for item in batch],
                )
                print(f"  inserted {min(start + 50, len(unseen))}/{len(unseen)}", flush=True)
                checkpoint()
        statuses = await rag.aget_docs_by_ids(ids)
        return [i for i in ids if _status(statuses.get(i)) != "processed"]


def _status(record: Any) -> str | None:
    if not record:
        return None
    value = record.get("status") if isinstance(record, dict) else getattr(record, "status", None)
    return str(getattr(value, "value", value)).lower() if value is not None else None


async def _naive(engine, query: str, QueryParam) -> list[str]:
    async with engine._use(GRAPH_WORKSPACE) as rag:
        result = await rag.aquery_data(
            query, QueryParam(mode="naive", top_k=40, chunk_top_k=20, enable_rerank=False)
        )
        if result.get("status") != "success":
            raise RuntimeError("LightRAG naive query failed")
        ranked = []
        for chunk in result["data"].get("chunks", []):
            stored = await rag.text_chunks.get_by_id(chunk["chunk_id"])
            ranked.append((stored or {}).get("full_doc_id", ""))
        return ranked


# ---------------------------------------------------------------- answers


SYSTEM_PROMPT = (
    "Answer the question using only the passages. Reply with JSON "
    '{"answer": "<shortest phrase that answers the question>"}. If the passages do not '
    'contain the answer, reply {"answer": "unknown"}.'
)


def _passages(ranked: list[str], corpus: dict[str, dict]) -> str:
    return "\n\n".join(
        f"[{i}] {corpus[sid]['text']}" for i, sid in enumerate(ranked[:ANSWER_CONTEXT], start=1)
    )


def _graph_facts(row: dict) -> str:
    lines = []
    for entity in row.get("entities", []):
        lines.append(f"- {entity.get('entity_name', '')}: {entity.get('description', '')}")
    for relation in row.get("relationships", []):
        lines.append(f"- {relation.get('src_id', '')} -> {relation.get('tgt_id', '')}: "
                     f"{relation.get('description', '')}")
    return "Knowledge-graph facts:\n" + "\n".join(lines) if lines else ""


def _parse_answer(content: str) -> str:
    match = re.search(r"\{.*\}", content or "", re.DOTALL)
    if match:
        try:
            return str(json.loads(match.group(0)).get("answer", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            pass
    return (content or "").strip()


async def _answer(state: Path, bench: dict[str, Any], concurrency: int) -> None:
    _load_env()
    from openai import AsyncOpenAI, RateLimitError

    model = os.environ.get("KNOWLEDGE_QA_MODEL") or os.environ["KNOWLEDGE_LLM_MODEL"]
    thinking = (os.environ.get("KNOWLEDGE_QA_ENABLE_THINKING") or "").strip().strip("'\"").lower()
    extra_body = {"enable_thinking": thinking == "true"} if thinking in {"true", "false"} else None
    client = AsyncOpenAI(api_key=os.environ["KNOWLEDGE_LLM_API_KEY"],
                         base_url=os.environ["KNOWLEDGE_LLM_BASE_URL"].strip("'\""),
                         max_retries=0, timeout=120)
    corpus = {item["source_id"]: item for item in bench["corpus"]}
    hybrid = {r["question_id"]: r for r in _read(state / "retrieval_hybrid_rrf.json")["questions"]}
    graph = {r["question_id"]: r for r in _read(state / "retrieval_graph.json")["questions"]}
    path = state / "answers.jsonl"
    done = set()
    if path.exists():
        done = {(r["question_id"], r["arm"]) for r in map(json.loads, path.read_text().splitlines())}
    jobs = []
    for question in bench["questions"]:
        qid = question["question_id"]
        contexts = {
            "hybrid_rrf": _passages(hybrid[qid]["ranked"], corpus),
            "lightrag_mix": _passages(graph[qid]["lightrag_mix"]["ranked"], corpus),
            "lightrag_naive": _passages(graph[qid]["lightrag_naive"]["ranked"], corpus),
        }
        facts = _graph_facts(graph[qid]["lightrag_mix"])
        contexts["lightrag_mix_with_graph_facts"] = "\n\n".join(
            part for part in (contexts["lightrag_mix"], facts) if part
        )
        jobs.extend((question, arm, contexts[arm]) for arm in ARMS if (qid, arm) not in done)
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def run(question: dict, arm: str, context: str) -> None:
        async with semaphore:
            started = time.perf_counter()
            record: dict[str, Any] = {"question_id": question["question_id"], "arm": arm}
            # A 429 carries no model output, so re-issuing it is not re-sampling;
            # any call that produced an answer is never repeated.
            record["rate_limit_retries"] = 0
            while True:
                try:
                    response = await client.chat.completions.create(
                        model=model, temperature=0, max_tokens=256,
                        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": f"Passages:\n{context}\n\n"
                                                              f"Question: {question['question']}"}],
                        **({"extra_body": extra_body} if extra_body else {}),
                    )
                except RateLimitError:
                    if record["rate_limit_retries"] >= 8:
                        record.update(error="RateLimitError", answer="")
                        break
                    record["rate_limit_retries"] += 1
                    await asyncio.sleep(min(60, 5 * 2 ** record["rate_limit_retries"]))
                    continue
                except Exception as error:  # a failed call scores 0 and is reported
                    record.update(error=type(error).__name__, answer="")
                    break
                content = response.choices[0].message.content or ""
                record.update(raw=content, answer=_parse_answer(content),
                              usage=response.usage.model_dump() if response.usage else None)
                break
            record["latency_ms"] = (time.perf_counter() - started) * 1000
            async with lock:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    await asyncio.gather(*(run(*job) for job in jobs))
    _write(state / "answer_config.json", {"model": model, "enable_thinking": extra_body,
                                          "temperature": 0, "context_chunks": ANSWER_CONTEXT})


# ---------------------------------------------------------------- report


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def score(state: Path, bench: dict[str, Any]) -> dict[str, Any]:
    questions = bench["questions"]
    hybrid = {r["question_id"]: r for r in _read(state / "retrieval_hybrid_rrf.json")["questions"]}
    graph = {r["question_id"]: r for r in _read(state / "retrieval_graph.json")["questions"]}
    answers: dict[tuple[str, str], dict] = {}
    for line in (state / "answers.jsonl").read_text().splitlines():
        record = json.loads(line)
        answers[(record["question_id"], record["arm"])] = record

    def ranked(arm: str, qid: str) -> list[str]:
        return hybrid[qid]["ranked"] if arm == "hybrid_rrf" else graph[qid][arm]["ranked"]

    per_question = []
    for q in questions:
        qid, golds = q["question_id"], [q["answer"], *q["answer_aliases"]]
        row = {"question_id": qid, "hop": q["hop"]}
        for arm in ("hybrid_rrf", "lightrag_mix", "lightrag_naive"):
            for k in K_VALUES:
                row[f"{arm}.recall@{k}"] = recall_at(ranked(arm, qid), q["supporting_source_ids"], k)
                row[f"{arm}.full_support@{k}"] = full_support_at(
                    ranked(arm, qid), q["supporting_source_ids"], k)
        for arm in ARMS:
            record = answers.get((qid, arm), {"answer": "", "error": "missing"})
            row[f"{arm}.em"] = exact_match(record.get("answer", ""), golds)
            row[f"{arm}.f1"] = token_f1(record.get("answer", ""), golds)
            row[f"{arm}.answer"] = record.get("answer", "")
            row[f"{arm}.error"] = record.get("error")
        per_question.append(row)

    def mean(key: str, rows=per_question) -> float:
        return sum(r[key] for r in rows) / len(rows)

    metric_keys = [f"recall@{k}" for k in K_VALUES] + [f"full_support@{k}" for k in K_VALUES]
    summary: dict[str, Any] = {"n": len(questions), "arms": {}}
    for arm in ARMS:
        keys = (metric_keys if arm != "lightrag_mix_with_graph_facts" else []) + ["em", "f1"]
        summary["arms"][arm] = {
            "overall": {key: mean(f"{arm}.{key}") for key in keys},
            "by_hop": {hop: {key: mean(f"{arm}.{key}", [r for r in per_question if r["hop"] == hop])
                             for key in keys}
                       for hop in sorted({r["hop"] for r in per_question})},
            "answer_errors": sum(1 for r in per_question if r[f"{arm}.error"]),
        }
    column = lambda key: [r[key] for r in per_question]  # noqa: E731
    comparisons = {}
    for arm in ("lightrag_mix", "lightrag_naive", "lightrag_mix_with_graph_facts"):
        comparisons[arm] = {
            "f1": paired_bootstrap(column("hybrid_rrf.f1"), column(f"{arm}.f1")),
            "em": mcnemar_exact(column("hybrid_rrf.em"), column(f"{arm}.em")),
        }
        if arm != "lightrag_mix_with_graph_facts":
            comparisons[arm]["recall@10"] = paired_bootstrap(
                column("hybrid_rrf.recall@10"), column(f"{arm}.recall@10"))
            comparisons[arm]["full_support@10"] = mcnemar_exact(
                column("hybrid_rrf.full_support@10"), column(f"{arm}.full_support@10"))
    latency = {
        "hybrid_rrf": [r["latency_ms"] for r in hybrid.values()],
        "lightrag_mix": [r["lightrag_mix"]["latency_ms"] for r in graph.values()],
        "lightrag_naive": [r["lightrag_naive"]["latency_ms"] for r in graph.values()],
    }
    summary["latency_ms"] = {arm: {"p50": _percentile(v, 0.5), "p95": _percentile(v, 0.95)}
                             for arm, v in latency.items()}
    summary["comparisons_vs_hybrid"] = comparisons
    summary["decision"] = decide(comparisons["lightrag_mix"]["f1"],
                                 bench["protocol"]["decision_rule"])
    for name in ("graph_usage.json", "baseline_ingested.json", "answer_config.json"):
        if (state / name).exists():
            summary[name.removesuffix(".json")] = _read(state / name)
    result = {"benchmark_id": bench["manifest"]["benchmark_id"],
              "manifest_files": bench["manifest"]["files"],
              "summary": summary, "per_question": per_question}
    _write(state / "results.json", result)
    return result


# ---------------------------------------------------------------- cli


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=["validate", "retrieve-baseline", "retrieve-graph",
                                          "answer", "score", "sweep-baseline", "sweep-score"])
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK,
                        help="frozen sample directory (default: the GraphRAG comparison sample)")
    parser.add_argument("--holdout-state", type=Path, help="sweep-score: held-out state directory")
    parser.add_argument("--weights", default="1:0,4:1,3:1,2:1,1.5:1,1:1,1:1.5,1:2",
                        help="sweep-baseline: dense:bm25 fusion weights")
    parser.add_argument("--state", type=Path, help="state directory shared by the phases")
    parser.add_argument("--limit", type=int, help="smoke run on the first N questions")
    parser.add_argument("--parallel", type=int, default=8,
                        help="concurrent LightRAG documents and LLM calls while indexing")
    parser.add_argument("--concurrency", type=int, default=8, help="concurrent answer calls")
    args = parser.parse_args()
    bench = load_benchmark(args.limit, args.benchmark.resolve())
    if args.phase == "validate":
        print(json.dumps({"status": "fixtures_valid", "questions": len(bench["questions"]),
                          "corpus": len(bench["corpus"]), "live_model_calls": False}))
        return 0
    if args.state is None:
        parser.error("--state is required for this phase")
    args.state.mkdir(parents=True, exist_ok=True)
    if args.phase == "retrieve-baseline":
        asyncio.run(_retrieve_baseline(args.state, bench))
    elif args.phase == "retrieve-graph":
        asyncio.run(_retrieve_graph(args.state, bench, args.parallel))
    elif args.phase == "answer":
        asyncio.run(_answer(args.state, bench, args.concurrency))
    elif args.phase == "sweep-baseline":
        asyncio.run(_sweep_baseline(args.state, bench, args.weights.split(",")))
    elif args.phase == "sweep-score":
        if args.holdout_state is None:
            parser.error("--holdout-state is required for sweep-score")
        result = sweep_score(args.state, args.holdout_state, bench)
        print(json.dumps({k: result[k] for k in ("selected_on_tuning", "decision",
                                                 "holdout_comparison_vs_1_1")}, indent=1))
    else:
        summary = score(args.state, bench)["summary"]
        print(json.dumps({key: summary[key] for key in ("n", "decision", "latency_ms")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
