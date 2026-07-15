"""Server-side evidence registry and citation-ID validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from models.citation import CitationView


_CHUNK_INDEX_RE = re.compile(r"_chunk_(\d+)$")


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    document_id: str
    text: str
    rank: int


@dataclass(frozen=True)
class CitationResolution:
    citations: list[CitationView]
    invalid_ids: list[str]


EvidenceRegistry = dict[str, EvidenceChunk]


def collect_search_evidence(result: str, registry: EvidenceRegistry) -> int:
    """Add aligned ``chunks/chunk_ids`` from a search observation.

    Malformed payloads fail closed and leave the registry unchanged.  The
    caller may still expose the tool error to the model, but it cannot become
    citation evidence.
    """

    try:
        payload = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return 0
    if not isinstance(payload, dict):
        return 0

    document_id = payload.get("document_id")
    chunks = payload.get("chunks")
    chunk_ids = payload.get("chunk_ids")
    if (
        not isinstance(document_id, str)
        or not document_id
        or not isinstance(chunks, list)
        or not isinstance(chunk_ids, list)
        or len(chunks) != len(chunk_ids)
        or any(not isinstance(chunk, str) for chunk in chunks)
        or any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in chunk_ids)
    ):
        return 0

    additions = {
        chunk_id: EvidenceChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=chunk,
            rank=rank,
        )
        for rank, (chunk_id, chunk) in enumerate(zip(chunk_ids, chunks), 1)
    }
    for chunk_id, evidence in additions.items():
        registry.setdefault(chunk_id, evidence)
    return len(additions)


def resolve_citations(
    requested_ids: Iterable[str] | None,
    registry: EvidenceRegistry,
) -> CitationResolution:
    """Resolve only IDs observed in this run; deduplicate in model order."""

    citations: list[CitationView] = []
    invalid_ids: list[str] = []
    seen: set[str] = set()
    for raw_id in requested_ids or []:
        if not isinstance(raw_id, str) or raw_id in seen:
            continue
        seen.add(raw_id)
        evidence = registry.get(raw_id)
        if evidence is None:
            invalid_ids.append(raw_id)
            continue
        index_match = _CHUNK_INDEX_RE.search(evidence.chunk_id)
        citations.append(CitationView(
            chunk_id=evidence.chunk_id,
            document_id=evidence.document_id,
            chunk_index=int(index_match.group(1)) if index_match else None,
            rank=evidence.rank,
            snippet=evidence.text[:240],
        ))
    return CitationResolution(citations=citations, invalid_ids=invalid_ids)
