from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import re
import uuid
from pathlib import Path
from typing import Any


EXPECTED_DIST_VERSION = "1.5.7"
REQUIRED_POSTGRES_STORAGES = (
    "PGKVStorage",
    "PGDocStatusStorage",
    "PGTableGraphStorage",
    "PGVectorStorage",
)
REQUIRED_METHOD_PARAMETERS = {
    "ainsert": ("input", "ids", "file_paths"),
    "aquery_data": ("query", "param"),
    "adelete_by_doc_id": ("doc_id",),
    "aedit_entity": ("entity_name", "updated_data", "allow_rename", "allow_merge"),
    "amerge_entities": ("source_entities", "target_entity"),
    "adelete_by_relation": ("source_entity", "target_entity"),
}


def sdk_contract_violations() -> list[str]:
    """Return hard contract mismatches without selecting a fallback backend."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        installed_version = version("lightrag-hku")
    except PackageNotFoundError:
        return ["lightrag-hku is not installed"]

    violations: list[str] = []
    if installed_version != EXPECTED_DIST_VERSION:
        violations.append(
            f"lightrag-hku must be {EXPECTED_DIST_VERSION}, got {installed_version}"
        )

    from lightrag import LightRAG
    from lightrag.kg import STORAGES

    missing_storages: list[str] = []
    for name in REQUIRED_POSTGRES_STORAGES:
        module_name = STORAGES.get(name)
        if module_name is None:
            missing_storages.append(name)
            continue
        try:
            module = importlib.import_module(module_name, package="lightrag")
        except (ImportError, ModuleNotFoundError):
            missing_storages.append(name)
            continue
        if not hasattr(module, name):
            missing_storages.append(name)
    if missing_storages:
        violations.append(
            "required PostgreSQL storage class(es) absent: "
            + ", ".join(missing_storages)
            + "; refusing to substitute another graph storage"
        )

    for method_name, required_parameters in REQUIRED_METHOD_PARAMETERS.items():
        method = getattr(LightRAG, method_name, None)
        if method is None:
            violations.append(f"LightRAG.{method_name} is absent")
            continue
        actual_parameters = inspect.signature(method).parameters
        missing_parameters = [
            name for name in required_parameters if name not in actual_parameters
        ]
        if missing_parameters:
            violations.append(
                f"LightRAG.{method_name} lacks parameter(s): "
                + ", ".join(missing_parameters)
            )
    return violations


def unique_workspace(label: str) -> str:
    """Build a backend-safe workspace that cannot collide across test runs."""
    return f"contract_{label}_{uuid.uuid4().hex}"


def immutable_doc_id() -> str:
    return str(uuid.uuid4())


def opaque_source_token() -> str:
    return f"src_{uuid.uuid4().hex}"


async def deterministic_embedding(texts: list[str]):
    """Deterministic local embedding; only the external model boundary is faked."""
    import numpy as np

    dimensions = 64
    rows: list[list[float]] = []
    for text in texts:
        vector = np.zeros(dimensions, dtype=np.float32)
        for token in re.findall(r"[A-Za-z0-9_]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:2], "big") % dimensions] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            vector[0] = 1.0
        else:
            vector /= norm
        rows.append(vector.tolist())
    return np.asarray(rows, dtype=np.float32)


def _input_text(prompt: str) -> str:
    match = re.search(r"---Input Text---\s*```\s*(.*?)\s*```", prompt, re.DOTALL)
    return match.group(1) if match else ""


async def deterministic_llm(prompt: str, system_prompt: str | None = None, **_: Any) -> str:
    """Produce deterministic extraction records without network/model calls."""
    combined = f"{system_prompt or ''}\n{prompt}"
    if "high_level_keywords" in combined and "low_level_keywords" in combined:
        words = re.findall(r"[A-Za-z][A-Za-z0-9_]*", prompt)
        keywords = list(dict.fromkeys(word.title() for word in words[-8:]))
        return json.dumps(
            {"high_level_keywords": keywords, "low_level_keywords": keywords}
        )

    if "Extract entities and relationships" in combined:
        text = _input_text(prompt)
        entities = list(dict.fromkeys(re.findall(r"ENTITY\[([^\]]+)\]", text)))
        relations = re.findall(r"REL\[([^|\]]+)\|([^\]]+)\]", text)
        rows = [
            f"entity<|#|>{name}<|#|>Concept<|#|>{name} appears in the source."
            for name in entities
        ]
        rows.extend(
            f"relation<|#|>{source}<|#|>{target}<|#|>supports<|#|>"
            f"{source} supports {target}."
            for source, target in relations
        )
        return "\n".join([*rows, "<|COMPLETE|>"])

    if "missed or incorrectly formatted" in combined:
        return "<|COMPLETE|>"
    if "synthesize" in combined.lower():
        return "Deterministic merged description."
    return "Deterministic response."


def build_embedding_func():
    from lightrag.utils import EmbeddingFunc

    return EmbeddingFunc(
        embedding_dim=64,
        max_token_size=8192,
        model_name="deterministic-contract-v1",
        func=deterministic_embedding,
    )


def make_rag(*, workspace: str, working_dir: Path):
    """Construct the exact approved storage combination; never fall back."""
    from lightrag import LightRAG

    return LightRAG(
        working_dir=str(working_dir),
        workspace=workspace,
        kv_storage="PGKVStorage",
        doc_status_storage="PGDocStatusStorage",
        graph_storage="PGTableGraphStorage",
        vector_storage="PGVectorStorage",
        llm_model_func=deterministic_llm,
        embedding_func=build_embedding_func(),
        entity_extract_max_gleaning=0,
        enable_llm_cache=False,
        cosine_better_than_threshold=-1.0,
    )


async def chunk_doc_ids(rag, chunks: list[dict[str, Any]]) -> dict[str, str | None]:
    """Resolve public retrieval chunk IDs through the real text_chunks store."""
    result: dict[str, str | None] = {}
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        stored = await rag.text_chunks.get_by_id(chunk_id)
        result[chunk_id] = stored.get("full_doc_id") if stored else None
    return result
