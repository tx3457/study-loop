#!/usr/bin/env python3
"""Characterize answers over frozen retrieval results with at most 24 calls."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "evaluation" / "knowledge"
PROTOCOL_PATH = FIXTURE_ROOT / "answer_protocol.json"
ANSWER_MANIFEST_PATH = FIXTURE_ROOT / "answer_manifest.json"
RETRIEVAL_MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"
QUESTIONS_PATH = FIXTURE_ROOT / "questions.json"
SOURCES_PATH = FIXTURE_ROOT / "sources.json"


class AnswerEvaluationError(RuntimeError):
    def __init__(self, kind: str, message: str, *, record: dict[str, Any] | None = None):
        self.kind = kind
        self.record = record
        super().__init__(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnswerEvaluationError(
            "fixture_error", f"invalid JSON {path.name}: {type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict):
        raise AnswerEvaluationError("fixture_error", f"{path.name} must be an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_hash(files: dict[str, str]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_frozen_protocol() -> dict[str, Any]:
    answer_manifest = _read_json(ANSWER_MANIFEST_PATH)
    protocol = _read_json(PROTOCOL_PATH)
    retrieval_manifest = _read_json(RETRIEVAL_MANIFEST_PATH)
    questions_payload = _read_json(QUESTIONS_PATH)
    sources_payload = _read_json(SOURCES_PATH)

    files = answer_manifest.get("files")
    expected_files = {"answer_protocol.json": _sha256(PROTOCOL_PATH)}
    if files != expected_files:
        raise AnswerEvaluationError("fixture_error", "answer protocol hash mismatch")
    if answer_manifest.get("files_payload_sha256") != _payload_hash(expected_files):
        raise AnswerEvaluationError("fixture_error", "answer manifest payload hash mismatch")
    if answer_manifest.get("live_model_calls_before_freeze") is not False:
        raise AnswerEvaluationError(
            "fixture_error", "answer protocol was not frozen before live calls"
        )
    if answer_manifest.get("parent_retrieval_manifest_sha256") != _sha256(
        RETRIEVAL_MANIFEST_PATH
    ):
        raise AnswerEvaluationError("fixture_error", "retrieval manifest hash mismatch")
    parent_hashes = answer_manifest.get("required_parent_hashes", {})
    if parent_hashes != {
        "manifest_files_payload_sha256": retrieval_manifest.get(
            "files_payload_sha256"
        ),
        "questions.json": retrieval_manifest.get("files", {}).get("questions.json"),
        "sources.json": retrieval_manifest.get("files", {}).get("sources.json"),
    }:
        raise AnswerEvaluationError("fixture_error", "parent fixture hashes changed")
    if _sha256(QUESTIONS_PATH) != parent_hashes["questions.json"]:
        raise AnswerEvaluationError("fixture_error", "questions.json content changed")
    if _sha256(SOURCES_PATH) != parent_hashes["sources.json"]:
        raise AnswerEvaluationError("fixture_error", "sources.json content changed")

    questions = questions_payload.get("questions")
    if not isinstance(questions, list) or len(questions) != 12:
        raise AnswerEvaluationError("fixture_error", "expected exactly 12 questions")
    question_ids = [question.get("question_id") for question in questions]
    retrieval_inputs = protocol.get("retrieval_inputs", {})
    if question_ids != retrieval_inputs.get("question_order"):
        raise AnswerEvaluationError("fixture_error", "question order changed")
    methods = retrieval_inputs.get("methods_in_order")
    if methods != ["existing_hybrid_rrf", "lightrag_mix"]:
        raise AnswerEvaluationError("fixture_error", "method order changed")
    if protocol.get("generation", {}).get("maximum_calls") != 24:
        raise AnswerEvaluationError("fixture_error", "maximum_calls must remain 24")

    answerable_ids = set(protocol.get("rubrics", {}).get("answerable", {}))
    unanswerable_ids = set(
        protocol.get("rubrics", {}).get("unanswerable_question_ids", [])
    )
    if answerable_ids | unanswerable_ids != set(question_ids):
        raise AnswerEvaluationError("fixture_error", "rubrics do not cover all questions")
    if answerable_ids & unanswerable_ids or len(answerable_ids) != 10:
        raise AnswerEvaluationError("fixture_error", "rubric answerability split changed")
    for question_id, rubric in protocol["rubrics"]["answerable"].items():
        concepts = rubric.get("concepts")
        if not isinstance(concepts, list) or not concepts:
            raise AnswerEvaluationError(
                "fixture_error", f"question {question_id} has no concept rubric"
            )
        for concept in concepts:
            patterns = concept.get("patterns")
            if not isinstance(patterns, list) or not patterns:
                raise AnswerEvaluationError(
                    "fixture_error", f"question {question_id} has an empty concept"
                )
            for pattern in patterns:
                re.compile(pattern, re.IGNORECASE)

    source_domain: dict[str, str] = {}
    for domain in sources_payload.get("domains", []):
        for source in domain.get("sources", []):
            source_domain[source["source_id"]] = domain["domain_id"]
    return {
        "answer_manifest": answer_manifest,
        "protocol": protocol,
        "questions": questions,
        "source_domain": source_domain,
    }


def score_answer(
    *,
    question_id: str,
    answer: str,
    citation_ids: list[str],
    abstained: bool,
    retrieved_source_ids: list[str],
    required_source_ids: list[str],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Score lexical concepts and citation contracts without judging truth."""
    allowed = set(retrieved_source_ids)
    cited = set(citation_ids)
    required = set(required_source_ids)
    fabricated = sorted(cited - allowed)
    unanswerable = question_id in set(
        protocol["rubrics"]["unanswerable_question_ids"]
    )
    result: dict[str, Any] = {
        "semantic_truth_evaluated": False,
        "abstention_expected": unanswerable,
        "abstention_correct": abstained if unanswerable else not abstained,
        "fabricated_citation_ids": fabricated,
        "citations_within_retrieval": not fabricated,
        "cited_required_source_ids": sorted(cited & required),
        "uncited_required_source_ids": sorted(required - cited),
        "required_citation_coverage": (
            None if not required else len(cited & required) / len(required)
        ),
    }
    if unanswerable:
        result.update(
            lexical_concepts=None,
            lexical_rubric_coverage=None,
            unanswerable_has_citations=bool(citation_ids),
            unanswerable_has_answer_text=bool(answer.strip()),
        )
        return result

    concept_results: list[dict[str, Any]] = []
    for concept in protocol["rubrics"]["answerable"][question_id]["concepts"]:
        matched_pattern = next(
            (
                pattern
                for pattern in concept["patterns"]
                if re.search(pattern, answer, re.IGNORECASE)
            ),
            None,
        )
        concept_results.append(
            {
                "concept_id": concept["concept_id"],
                "matched": matched_pattern is not None,
                "matched_pattern": matched_pattern,
            }
        )
    result["lexical_concepts"] = concept_results
    result["lexical_rubric_coverage"] = sum(
        concept["matched"] for concept in concept_results
    ) / len(concept_results)
    return result


def _validate_retrieval_results(
    path: Path, frozen: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    results = _read_json(path)
    protocol = frozen["protocol"]
    if results.get("evaluation_id") != "knowledge_retrieval_small_v1":
        raise AnswerEvaluationError("input_error", "unexpected retrieval evaluation_id")
    if results.get("manifest_files_payload_sha256") != protocol["retrieval_inputs"][
        "required_manifest_files_payload_sha256"
    ]:
        raise AnswerEvaluationError("input_error", "retrieval fixture hash mismatch")
    rows = results.get("questions")
    if not isinstance(rows, list):
        raise AnswerEvaluationError("input_error", "retrieval questions must be an array")
    expected_ids = protocol["retrieval_inputs"]["question_order"]
    if [row.get("question_id") for row in rows] != expected_ids:
        raise AnswerEvaluationError("input_error", "retrieval question order mismatch")
    frozen_questions = {row["question_id"]: row for row in frozen["questions"]}
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = row["question_id"]
        expected = frozen_questions[question_id]
        for field in ("domain_id", "query", "required_source_ids"):
            if row.get(field) != expected.get(field):
                raise AnswerEvaluationError(
                    "input_error", f"retrieval {question_id} changed {field}"
                )
        methods = row.get("methods")
        if not isinstance(methods, dict) or list(methods) != protocol[
            "retrieval_inputs"
        ]["methods_in_order"]:
            raise AnswerEvaluationError(
                "input_error", f"retrieval {question_id} method order mismatch"
            )
        for method_name, method in methods.items():
            retrieved_ids = method.get("retrieved_source_ids")
            chunks = method.get("chunks")
            if not isinstance(retrieved_ids, list) or not all(
                isinstance(value, str) for value in retrieved_ids
            ):
                raise AnswerEvaluationError(
                    "input_error", f"{question_id}/{method_name} source IDs invalid"
                )
            if len(retrieved_ids) != len(set(retrieved_ids)):
                raise AnswerEvaluationError(
                    "input_error", f"{question_id}/{method_name} source IDs duplicate"
                )
            if not isinstance(chunks, list):
                raise AnswerEvaluationError(
                    "input_error", f"{question_id}/{method_name} chunks invalid"
                )
            for source_id in retrieved_ids:
                if frozen["source_domain"].get(source_id) != row["domain_id"]:
                    raise AnswerEvaluationError(
                        "input_error",
                        f"{question_id}/{method_name} source outside frozen domain",
                    )
            for chunk in chunks:
                if (
                    not isinstance(chunk, dict)
                    or chunk.get("source_id") not in retrieved_ids
                    or not isinstance(chunk.get("text"), str)
                ):
                    raise AnswerEvaluationError(
                        "input_error", f"{question_id}/{method_name} chunk invalid"
                    )
        by_id[question_id] = row
    return results, by_id


def _parse_response_content(content: Any) -> dict[str, Any]:
    if not isinstance(content, str):
        raise AnswerEvaluationError("format_failure", "response content is not text")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AnswerEvaluationError("format_failure", "response is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "answer",
        "citation_ids",
        "abstained",
    }:
        raise AnswerEvaluationError("format_failure", "response keys violate schema")
    if not isinstance(payload["answer"], str):
        raise AnswerEvaluationError("format_failure", "answer must be a string")
    citation_ids = payload["citation_ids"]
    if not isinstance(citation_ids, list) or any(
        not isinstance(value, str) or not value for value in citation_ids
    ):
        raise AnswerEvaluationError("format_failure", "citation_ids must be strings")
    if len(citation_ids) != len(set(citation_ids)):
        raise AnswerEvaluationError("format_failure", "citation_ids must be unique")
    if not isinstance(payload["abstained"], bool):
        raise AnswerEvaluationError("format_failure", "abstained must be boolean")
    return payload


def _usage_payload(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    elif isinstance(usage, dict):
        raw = usage
    else:
        raw = {
            name: getattr(usage, name, None)
            for name in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
        }
    allowed = {
        key: value
        for key, value in raw.items()
        if key
        in {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "cost",
        }
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    return allowed or None


def _safe_error(exc: BaseException) -> dict[str, str]:
    message = str(exc)
    message = re.sub(
        r"(?i)(api[_-]?key|token|password)\s*[=:]\s*\S+",
        r"\1=[REDACTED]",
        message,
    )
    return {"error_type": type(exc).__name__, "error_message": message[:500]}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _messages(protocol: dict[str, Any], question: str, chunks: list[dict]) -> list[dict]:
    evidence = [
        {"source_id": chunk["source_id"], "text": chunk["text"]}
        for chunk in chunks
    ]
    return [
        {"role": "system", "content": protocol["generation"]["shared_system_prompt"]},
        {
            "role": "user",
            "content": json.dumps(
                {"question": question, "evidence": evidence},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


async def _run_live(
    *,
    retrieval_path: Path,
    output_dir: Path,
    frozen: dict[str, Any],
) -> int:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise AnswerEvaluationError("configuration_error", "output directory is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_results, retrieval_by_id = _validate_retrieval_results(
        retrieval_path, frozen
    )
    protocol = frozen["protocol"]
    status: dict[str, Any] = {
        "evaluation_id": protocol["evaluation_id"],
        "status": "running",
        "live_models_authorized": True,
        "completed_calls": 0,
        "maximum_calls": 24,
    }
    output: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_id": protocol["evaluation_id"],
        "answer_manifest_files_payload_sha256": frozen["answer_manifest"][
            "files_payload_sha256"
        ],
        "retrieval_results_sha256": _sha256(retrieval_path),
        "retrieval_run_id": retrieval_results.get("run_id"),
        "semantic_truth_evaluated": False,
        "calls": [],
    }
    _write_json(output_dir / "STATUS.json", status)
    _write_json(output_dir / "results.partial.json", output)

    current_question_id: str | None = None
    current_method: str | None = None
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from services.llm import _chat_config, base_url, llm_chat, model

        if _chat_config.issues or not model:
            raise AnswerEvaluationError(
                "provider_failure",
                "current chat provider configuration is incomplete",
            )
        output["configuration"] = {
            "model": model,
            "base_url_sha256": (
                hashlib.sha256(base_url.rstrip("/").lower().encode()).hexdigest()
                if base_url
                else None
            ),
            "max_retries": 0,
            "temperature": 0,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
            "secrets_recorded": False,
        }
        methods = protocol["retrieval_inputs"]["methods_in_order"]
        questions = {row["question_id"]: row for row in frozen["questions"]}
        for question_id in protocol["retrieval_inputs"]["question_order"]:
            current_question_id = question_id
            question = questions[question_id]
            retrieval_row = retrieval_by_id[question_id]
            for method_name in methods:
                current_method = method_name
                method = retrieval_row["methods"][method_name]
                call_record: dict[str, Any] = {
                    "question_id": question_id,
                    "method": method_name,
                    "retrieved_source_ids": method["retrieved_source_ids"],
                    "required_source_ids": question["required_source_ids"],
                }
                started = time.perf_counter()
                try:
                    response = await llm_chat(
                        _messages(protocol, question["query"], method["chunks"]),
                        max_retries=0,
                        temperature=0,
                        max_tokens=2048,
                        response_format={"type": "json_object"},
                    )
                except Exception as exc:
                    call_record.update(
                        status="failed",
                        failure_kind="provider_failure",
                        latency_ms=(time.perf_counter() - started) * 1000,
                        **_safe_error(exc),
                    )
                    raise AnswerEvaluationError(
                        "provider_failure", "chat provider call failed", record=call_record
                    ) from exc

                latency_ms = (time.perf_counter() - started) * 1000
                choice = response.choices[0] if response.choices else None
                finish_reason = getattr(choice, "finish_reason", None)
                raw_content = (
                    getattr(getattr(choice, "message", None), "content", None)
                    if choice is not None
                    else None
                )
                usage = _usage_payload(getattr(response, "usage", None))
                provider_cost = (usage or {}).get("cost")
                call_record.update(
                    latency_ms=latency_ms,
                    finish_reason=finish_reason,
                    raw_response_content=raw_content,
                    actual_response_usage=usage,
                    actual_response_usage_unavailable_reason=(
                        None if usage is not None else "provider response did not report usage"
                    ),
                    actual_provider_cost=provider_cost,
                    provider_cost_unavailable_reason=(
                        None
                        if provider_cost is not None
                        else "provider response did not report monetary cost"
                    ),
                )
                if finish_reason != "stop":
                    kind = "length_failure" if finish_reason == "length" else "finish_failure"
                    call_record.update(status="failed", failure_kind=kind)
                    raise AnswerEvaluationError(
                        kind,
                        f"unexpected finish_reason: {finish_reason!r}",
                        record=call_record,
                    )
                try:
                    parsed = _parse_response_content(raw_content)
                except AnswerEvaluationError as exc:
                    call_record.update(status="failed", failure_kind=exc.kind)
                    exc.record = call_record
                    raise
                call_record.update(
                    status="complete",
                    answer=parsed["answer"],
                    citation_ids=parsed["citation_ids"],
                    abstained=parsed["abstained"],
                    scoring=score_answer(
                        question_id=question_id,
                        answer=parsed["answer"],
                        citation_ids=parsed["citation_ids"],
                        abstained=parsed["abstained"],
                        retrieved_source_ids=method["retrieved_source_ids"],
                        required_source_ids=question["required_source_ids"],
                        protocol=protocol,
                    ),
                )
                output["calls"].append(call_record)
                status["completed_calls"] = len(output["calls"])
                _write_json(output_dir / "results.partial.json", output)
                _write_json(output_dir / "STATUS.json", status)

        answerable_ids = set(protocol["rubrics"]["answerable"])
        unanswerable_ids = set(protocol["rubrics"]["unanswerable_question_ids"])
        output["summary"] = {}
        for method_name in methods:
            calls = [row for row in output["calls"] if row["method"] == method_name]
            answerable = [row for row in calls if row["question_id"] in answerable_ids]
            unanswerable = [
                row for row in calls if row["question_id"] in unanswerable_ids
            ]
            usage_rows = [
                row["actual_response_usage"]
                for row in calls
                if row["actual_response_usage"] is not None
            ]
            output["summary"][method_name] = {
                "call_count": len(calls),
                "mean_handchecked_concept_regex_coverage": sum(
                    row["scoring"]["lexical_rubric_coverage"] for row in answerable
                )
                / len(answerable),
                "mean_required_citation_coverage": sum(
                    row["scoring"]["required_citation_coverage"] for row in answerable
                )
                / len(answerable),
                "fabricated_citation_count": sum(
                    len(row["scoring"]["fabricated_citation_ids"]) for row in calls
                ),
                "unanswerable_correct_abstention_count": sum(
                    row["scoring"]["abstention_correct"] for row in unanswerable
                ),
                "mean_latency_ms": sum(row["latency_ms"] for row in calls) / len(calls),
                "actual_usage_reported_call_count": len(usage_rows),
                "actual_provider_cost_total": (
                    sum(
                        row["actual_provider_cost"]
                        for row in calls
                        if row["actual_provider_cost"] is not None
                    )
                    if any(row["actual_provider_cost"] is not None for row in calls)
                    else None
                ),
                "semantic_truth_evaluated": False,
            }
        _write_json(output_dir / "results.json", output)
        partial = output_dir / "results.partial.json"
        if partial.exists():
            partial.unlink()
        status.update(status="complete", completed_calls=24)
        _write_json(output_dir / "STATUS.json", status)
        return 0
    except AnswerEvaluationError as exc:
        if exc.record is not None:
            output["calls"].append(exc.record)
        status.update(
            status="failed",
            failed_question_id=current_question_id,
            failed_method=current_method,
            failure_kind=exc.kind,
            **_safe_error(exc),
        )
        _write_json(output_dir / "results.partial.json", output)
        _write_json(output_dir / "STATUS.json", status)
        return 2
    except Exception as exc:
        status.update(
            status="failed",
            failed_question_id=current_question_id,
            failed_method=current_method,
            failure_kind="unexpected_failure",
            **_safe_error(exc),
        )
        _write_json(output_dir / "results.partial.json", output)
        _write_json(output_dir / "STATUS.json", status)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval-results",
        type=Path,
        help="Completed frozen retrieval results.json (required for live execution)",
    )
    parser.add_argument(
        "--allow-live-models",
        action="store_true",
        help="Explicitly authorize the bounded 24-call answer run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New or empty output directory (required for live execution)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        frozen = validate_frozen_protocol()
        if not args.allow_live_models:
            print(
                json.dumps(
                    {
                        "status": "answer_protocol_valid",
                        "evaluation_id": frozen["protocol"]["evaluation_id"],
                        "files_payload_sha256": frozen["answer_manifest"][
                            "files_payload_sha256"
                        ],
                        "maximum_calls": 24,
                        "live_model_calls": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.retrieval_results is None or args.output_dir is None:
            raise AnswerEvaluationError(
                "configuration_error",
                "--retrieval-results and --output-dir are required with --allow-live-models",
            )
        return asyncio.run(
            _run_live(
                retrieval_path=args.retrieval_results.resolve(),
                output_dir=args.output_dir.resolve(),
                frozen=frozen,
            )
        )
    except AnswerEvaluationError as exc:
        print(f"answer evaluation error [{exc.kind}]: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
