#!/usr/bin/env python3
"""Run the frozen small-corpus legacy-vs-LightRAG retrieval evaluation.

Without ``--allow-live-models`` this command only verifies the frozen fixture
hashes and protocol. The live path uses a temporary Chroma directory and unique
LightRAG workspaces; it never opens StudyLoop's configured Chroma directory.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = REPO_ROOT / "evaluation" / "knowledge"
FROZEN_FILES = ("protocol.json", "questions.json", "sources.json")


class EvaluationConfigurationError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationConfigurationError(
            f"invalid JSON fixture {path.name}: {type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict):
        raise EvaluationConfigurationError(f"fixture {path.name} must be an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_hash(files: dict[str, str]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_fixtures(root: Path) -> dict[str, Any]:
    manifest = _json(root / "manifest.json")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or set(expected_files) != set(FROZEN_FILES):
        raise EvaluationConfigurationError(
            f"manifest files must be exactly {list(FROZEN_FILES)}"
        )
    observed_files = {name: _sha256(root / name) for name in FROZEN_FILES}
    if observed_files != expected_files:
        mismatches = [
            name for name in FROZEN_FILES if observed_files[name] != expected_files.get(name)
        ]
        raise EvaluationConfigurationError(
            "frozen fixture hash mismatch: " + ", ".join(mismatches)
        )
    if _payload_hash(observed_files) != manifest.get("files_payload_sha256"):
        raise EvaluationConfigurationError("manifest files_payload_sha256 mismatch")
    if manifest.get("live_model_calls_before_freeze") is not False:
        raise EvaluationConfigurationError(
            "manifest must state that no live calls preceded fixture freeze"
        )

    sources_payload = _json(root / "sources.json")
    questions_payload = _json(root / "questions.json")
    protocol = _json(root / "protocol.json")
    domains = sources_payload.get("domains")
    questions = questions_payload.get("questions")
    if not isinstance(domains, list) or len(domains) != 2:
        raise EvaluationConfigurationError("sources must contain exactly two domains")
    if not isinstance(questions, list) or len(questions) != 12:
        raise EvaluationConfigurationError("questions must contain exactly 12 items")

    sources_by_domain: dict[str, list[dict[str, str]]] = {}
    all_source_ids: set[str] = set()
    for domain in domains:
        domain_id = domain.get("domain_id")
        sources = domain.get("sources")
        if not isinstance(domain_id, str) or not domain_id:
            raise EvaluationConfigurationError("every domain requires a domain_id")
        if domain_id in sources_by_domain:
            raise EvaluationConfigurationError(f"duplicate domain_id: {domain_id}")
        if not isinstance(sources, list) or len(sources) != 3:
            raise EvaluationConfigurationError(
                f"domain {domain_id} must contain exactly three sources"
            )
        normalized: list[dict[str, str]] = []
        for source in sources:
            source_id = source.get("source_id")
            title = source.get("title")
            text = source.get("text")
            if not all(isinstance(value, str) and value.strip() for value in (source_id, title, text)):
                raise EvaluationConfigurationError(
                    f"domain {domain_id} has an incomplete source"
                )
            if source_id in all_source_ids:
                raise EvaluationConfigurationError(f"duplicate source_id: {source_id}")
            all_source_ids.add(source_id)
            normalized.append({"source_id": source_id, "title": title, "text": text})
        sources_by_domain[domain_id] = normalized

    expected_case_counts = {
        "single_document": 3,
        "cross_document": 2,
        "unanswerable": 1,
    }
    question_ids: set[str] = set()
    case_counts = {
        domain_id: {case_type: 0 for case_type in expected_case_counts}
        for domain_id in sources_by_domain
    }
    for question in questions:
        question_id = question.get("question_id")
        domain_id = question.get("domain_id")
        case_type = question.get("case_type")
        query = question.get("query")
        required = question.get("required_source_ids")
        if not isinstance(question_id, str) or not question_id or question_id in question_ids:
            raise EvaluationConfigurationError(f"invalid question_id: {question_id!r}")
        question_ids.add(question_id)
        if domain_id not in sources_by_domain:
            raise EvaluationConfigurationError(
                f"question {question_id} uses unknown domain {domain_id!r}"
            )
        if case_type not in expected_case_counts:
            raise EvaluationConfigurationError(
                f"question {question_id} uses unknown case_type {case_type!r}"
            )
        if not isinstance(query, str) or not query.strip():
            raise EvaluationConfigurationError(f"question {question_id} has no query")
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            raise EvaluationConfigurationError(
                f"question {question_id} has invalid required_source_ids"
            )
        domain_source_ids = {
            source["source_id"] for source in sources_by_domain[domain_id]
        }
        if not set(required).issubset(domain_source_ids):
            raise EvaluationConfigurationError(
                f"question {question_id} requires a source outside its domain"
            )
        expected_required_count = {
            "single_document": 1,
            "cross_document": 2,
            "unanswerable": 0,
        }[case_type]
        if len(required) != expected_required_count or len(required) != len(set(required)):
            raise EvaluationConfigurationError(
                f"question {question_id} has the wrong required source count"
            )
        case_counts[domain_id][case_type] += 1
    for domain_id, observed in case_counts.items():
        if observed != expected_case_counts:
            raise EvaluationConfigurationError(
                f"domain {domain_id} question mix is {observed}, expected {expected_case_counts}"
            )
    if protocol.get("evaluation_id") != manifest.get("evaluation_id"):
        raise EvaluationConfigurationError("protocol and manifest evaluation_id differ")
    if protocol.get("retrieval", {}).get("reported_top_k") != 3:
        raise EvaluationConfigurationError("frozen protocol reported_top_k must be 3")
    return {
        "manifest": manifest,
        "protocol": protocol,
        "sources_by_domain": sources_by_domain,
        "questions": questions,
    }


def _safe_error(exc: BaseException) -> dict[str, str]:
    message = str(exc)
    message = re.sub(r"(?i)(api[_-]?key|token|password)\s*[=:]\s*\S+", r"\1=[REDACTED]", message)
    message = re.sub(r"(?i)(postgres(?:ql)?://[^:/\s]+:)[^@\s]+@", r"\1[REDACTED]@", message)
    return {"error_type": type(exc).__name__, "error_message": message[:500]}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _score(required: list[str], retrieved: list[str]) -> dict[str, Any]:
    ranked = _dedupe(retrieved)[:3]
    if not required:
        return {
            "recall_at_3": None,
            "retrieved_source_count": len(ranked),
            "returned_any_source": bool(ranked),
        }
    return {
        "recall_at_3": len(set(required) & set(ranked)) / len(required),
        "retrieved_source_count": len(ranked),
        "returned_any_source": bool(ranked),
    }


def _url_fingerprint(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.rstrip("/").lower().encode()).hexdigest()


async def _run_live(
    fixtures: dict[str, Any],
    output_dir: Path,
    *,
    enable_rerank: bool,
) -> int:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise EvaluationConfigurationError("live output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    database_url = os.environ.get("KNOWLEDGE_EVAL_DATABASE_URL", "").strip()
    if not database_url:
        raise EvaluationConfigurationError(
            "KNOWLEDGE_EVAL_DATABASE_URL must reference a disposable evaluation database"
        )

    run_id = uuid.uuid4().hex
    status: dict[str, Any] = {
        "evaluation_id": fixtures["manifest"]["evaluation_id"],
        "run_id": run_id,
        "status": "initializing",
        "live_models_authorized": True,
        "completed_questions": 0,
    }
    _write_json(output_dir / "STATUS.json", status)
    results: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_id": fixtures["manifest"]["evaluation_id"],
        "run_id": run_id,
        "manifest_files_payload_sha256": fixtures["manifest"]["files_payload_sha256"],
        "answer_generation": "not_run_retrieval_only",
        "questions": [],
    }

    graph_engine = None
    vectorstore = None
    stage = "provider_configuration"
    current_question_id: str | None = None
    with tempfile.TemporaryDirectory(prefix="study-loop-knowledge-eval-") as temp_root:
        temporary_root = Path(temp_root)
        os.environ["CHROMA_DIR"] = str(temporary_root / "chroma")
        os.environ.pop("POSTGRES_WORKSPACE", None)
        sys.path.insert(0, str(REPO_ROOT))
        try:
            from services.provider_config import load_provider_configs

            provider_configs = load_provider_configs()
            chat = provider_configs["chat"]
            embedding = provider_configs["embedding"]
            issues = [
                *(f"chat:{issue}" for issue in chat.issues),
                *(f"embedding:{issue}" for issue in embedding.issues),
            ]
            if issues:
                raise EvaluationConfigurationError(
                    "provider configuration invalid: " + ", ".join(issues)
                )
            if not chat.model or not embedding.model:
                raise EvaluationConfigurationError("chat and embedding models are required")

            from graph_service.config import Settings
            from graph_service.engine import LightRAGEngine
            from services import vectorstore as imported_vectorstore
            from services.reranker import rerank_docs, reranker_enabled

            vectorstore = imported_vectorstore
            if enable_rerank and not reranker_enabled():
                raise EvaluationConfigurationError(
                    "--enable-rerank requires RERANKER_ENABLED=true; refusing implicit fallback"
                )

            settings = Settings(
                database_url=database_url,
                internal_token="evaluation-only-not-used",
                materials_dir=temporary_root / "materials",
                working_dir=temporary_root / "lightrag",
                provider="openai",
                llm_model=chat.model,
                embedding_model=embedding.model,
                embedding_dim=int(os.environ.get("KNOWLEDGE_EMBEDDING_DIM") or os.environ.get("EMBEDDING_DIM", "0")),
                max_cached_instances=2,
                llm_timeout_seconds=int(os.environ.get("KNOWLEDGE_LLM_TIMEOUT_SECONDS", "60")),
                llm_api_key=chat.api_key,
                llm_base_url=chat.base_url,
                embedding_api_key=embedding.api_key,
                embedding_base_url=embedding.base_url,
            )
            if settings.embedding_dim <= 0:
                raise EvaluationConfigurationError(
                    "KNOWLEDGE_EMBEDDING_DIM or EMBEDDING_DIM must be positive"
                )
            graph_engine = LightRAGEngine(settings)
            results["configuration"] = {
                "lightrag_version": importlib.metadata.version("lightrag-hku"),
                "chat_model": chat.model,
                "embedding_model": embedding.model,
                "embedding_dim": settings.embedding_dim,
                "chat_base_url_sha256": _url_fingerprint(chat.base_url),
                "embedding_base_url_sha256": _url_fingerprint(embedding.base_url),
                "reranker_enabled": enable_rerank,
                "reranker_model": os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3") if enable_rerank else None,
                "secrets_recorded": False,
            }

            owner_id = f"knowledge-eval-{run_id}"
            domain_runtime: dict[str, dict[str, Any]] = {}
            stage = "corpus_ingestion"
            for domain_id, sources in fixtures["sources_by_domain"].items():
                baseline_document_id = f"knowledge-eval-{domain_id}-{run_id}"
                source_texts = [source["text"] for source in sources]
                source_by_text = {source["text"]: source["source_id"] for source in sources}
                started = time.perf_counter()
                await vectorstore.deal_document(
                    baseline_document_id,
                    f"{domain_id}.txt",
                    source_texts,
                    owner_id=owner_id,
                )
                baseline_ingest_ms = (time.perf_counter() - started) * 1000

                workspace = f"eval_{domain_id}_{run_id}"
                started = time.perf_counter()
                for source in sources:
                    await graph_engine.insert(
                        workspace,
                        source["source_id"],
                        source["text"],
                        f"eval_{source['source_id']}",
                    )
                graph_ingest_ms = (time.perf_counter() - started) * 1000
                domain_runtime[domain_id] = {
                    "baseline_document_id": baseline_document_id,
                    "workspace": workspace,
                    "source_by_text": source_by_text,
                    "baseline_ingest_latency_ms": baseline_ingest_ms,
                    "graph_ingest_latency_ms": graph_ingest_ms,
                }
            results["domain_ingestion"] = {
                domain_id: {
                    key: value
                    for key, value in runtime.items()
                    if key.endswith("latency_ms")
                }
                for domain_id, runtime in domain_runtime.items()
            }

            stage = "question_retrieval"
            for question in fixtures["questions"]:
                current_question_id = question["question_id"]
                runtime = domain_runtime[question["domain_id"]]
                started = time.perf_counter()
                baseline_raw = await vectorstore.hybrid_query_document(
                    runtime["baseline_document_id"],
                    question["query"],
                    n_results=3,
                    enable_rerank=False,
                    owner_id=owner_id,
                )
                baseline_latency_ms = (time.perf_counter() - started) * 1000
                baseline_docs = list(baseline_raw.get("documents", [[]])[0])
                baseline_ids = list(baseline_raw.get("ids", [[]])[0])
                if enable_rerank:
                    started = time.perf_counter()
                    reranked = await rerank_docs(
                        question["query"], baseline_docs, baseline_ids, top_k=3
                    )
                    baseline_latency_ms += (time.perf_counter() - started) * 1000
                    baseline_docs = [row[0] for row in reranked]
                    baseline_ids = [str(row[2]) for row in reranked]
                baseline_chunks = [
                    {
                        "rank": rank,
                        "chunk_id": chunk_id,
                        "source_id": runtime["source_by_text"].get(text),
                        "text": text,
                    }
                    for rank, (chunk_id, text) in enumerate(
                        zip(baseline_ids, baseline_docs, strict=True), start=1
                    )
                ]
                if any(row["source_id"] is None for row in baseline_chunks):
                    raise RuntimeError(
                        "baseline returned text outside the frozen source corpus"
                    )

                started = time.perf_counter()
                graph_raw = await graph_engine.query(
                    runtime["workspace"], question["query"]
                )
                graph_latency_ms = (time.perf_counter() - started) * 1000
                graph_chunks = [
                    {
                        "rank": rank,
                        "chunk_id": chunk.get("chunk_id"),
                        "source_id": chunk.get("full_doc_id"),
                        "text": chunk.get("content", ""),
                    }
                    for rank, chunk in enumerate(graph_raw.get("chunks", [])[:3], start=1)
                ]

                required = list(question["required_source_ids"])
                baseline_sources = [row["source_id"] for row in baseline_chunks]
                graph_sources = [row["source_id"] for row in graph_chunks]
                results["questions"].append(
                    {
                        "question_id": current_question_id,
                        "domain_id": question["domain_id"],
                        "case_type": question["case_type"],
                        "query": question["query"],
                        "required_source_ids": required,
                        "methods": {
                            "existing_hybrid_rrf": {
                                "retrieved_source_ids": _dedupe(baseline_sources)[:3],
                                "chunks": baseline_chunks,
                                "latency_ms": baseline_latency_ms,
                                "actual_token_usage": None,
                                "token_usage_unavailable_reason": "retrieval interface does not expose provider usage",
                                **_score(required, baseline_sources),
                            },
                            "lightrag_mix": {
                                "retrieved_source_ids": _dedupe(graph_sources)[:3],
                                "chunks": graph_chunks,
                                "latency_ms": graph_latency_ms,
                                "actual_token_usage": None,
                                "token_usage_unavailable_reason": "aquery_data interface does not expose provider usage",
                                **_score(required, graph_sources),
                            },
                        },
                        "answer": None,
                        "citations": [],
                        "answer_quality_status": "not_evaluated_retrieval_only",
                    }
                )
                status["completed_questions"] = len(results["questions"])
                _write_json(output_dir / "results.partial.json", results)
                _write_json(output_dir / "STATUS.json", status)

            answerable = [
                row
                for row in results["questions"]
                if row["case_type"] != "unanswerable"
            ]
            methods = ("existing_hybrid_rrf", "lightrag_mix")
            results["summary"] = {
                method: {
                    "answerable_question_count": len(answerable),
                    "mean_required_source_recall_at_3": sum(
                        row["methods"][method]["recall_at_3"] for row in answerable
                    )
                    / len(answerable),
                    "unanswerable_returned_any_source_count": sum(
                        bool(row["methods"][method]["returned_any_source"])
                        for row in results["questions"]
                        if row["case_type"] == "unanswerable"
                    ),
                    "mean_query_latency_ms": sum(
                        row["methods"][method]["latency_ms"]
                        for row in results["questions"]
                    )
                    / len(results["questions"]),
                }
                for method in methods
            }
            _write_json(output_dir / "results.json", results)
            partial = output_dir / "results.partial.json"
            if partial.exists():
                partial.unlink()
            status.update(status="complete", completed_questions=12)
            _write_json(output_dir / "STATUS.json", status)
            return 0
        except BaseException as exc:
            status.update(
                status="failed",
                failed_stage=stage,
                failed_question_id=current_question_id,
                **_safe_error(exc),
            )
            _write_json(output_dir / "results.partial.json", results)
            _write_json(output_dir / "STATUS.json", status)
            return 2
        finally:
            if graph_engine is not None:
                try:
                    await graph_engine.close()
                except Exception:
                    pass
            if vectorstore is not None:
                try:
                    await vectorstore.shutdown_vectorstore_io()
                except Exception:
                    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=DEFAULT_FIXTURES,
        help="Frozen fixture directory (default: repository evaluation/knowledge)",
    )
    parser.add_argument(
        "--allow-live-models",
        action="store_true",
        help="Explicitly authorize configured model calls for this bounded run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New/empty result directory; required for a live run",
    )
    parser.add_argument(
        "--enable-rerank",
        action="store_true",
        help="Apply the configured local reranker; failures stop the run",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        fixtures = validate_fixtures(args.fixtures_dir.resolve())
        if not args.allow_live_models:
            print(
                json.dumps(
                    {
                        "status": "fixtures_valid",
                        "evaluation_id": fixtures["manifest"]["evaluation_id"],
                        "files_payload_sha256": fixtures["manifest"]["files_payload_sha256"],
                        "live_model_calls": False,
                        "domain_count": len(fixtures["sources_by_domain"]),
                        "question_count": len(fixtures["questions"]),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.output_dir is None:
            raise EvaluationConfigurationError(
                "--output-dir is required with --allow-live-models"
            )
        return asyncio.run(
            _run_live(
                fixtures,
                args.output_dir.resolve(),
                enable_rerank=args.enable_rerank,
            )
        )
    except EvaluationConfigurationError as exc:
        print(f"evaluation configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
