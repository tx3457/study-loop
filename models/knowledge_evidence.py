"""Bounded, versionable source references; raw web bodies live outside sessions."""

from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Where a passage came from, ordered by how accountable that origin is. This is
# provenance, not a truth claim: it says who put the text within reach, never
# that the text is correct. Kept separate from ``source_status``, which answers
# whether the source is still live.
ProvenanceOrigin = Literal["user_upload", "web_import", "web_snapshot"]


class SourceCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["kb_chunk", "web_snapshot"]
    origin: ProvenanceOrigin = "user_upload"
    evidence_id: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=512)
    snippet: str = Field(min_length=1, max_length=2000)
    knowledge_base_id: str | None = Field(default=None, max_length=128)
    document_id: str | None = Field(default=None, max_length=512)
    source_version_id: str | None = Field(default=None, max_length=128)
    chunk_id: str | None = Field(default=None, max_length=512)
    snapshot_id: str | None = Field(default=None, max_length=128)
    url: str | None = Field(default=None, max_length=4096)
    fetched_at: str | None = Field(default=None, max_length=64)
    content_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    locator: dict[str, str | int] | None = None
    source_status: Literal["current", "deleted", "historical", "unavailable"] = "current"

    @model_validator(mode="before")
    @classmethod
    def backfill_origin(cls, data):
        """Derive ``origin`` for records written before provenance tiers existed.

        A persisted HITL snapshot must still deserialize; refusing to resume it
        is the controller policy version's job, not a validation error here.
        """
        if isinstance(data, dict) and not data.get("origin"):
            return {
                **data,
                "origin": (
                    "web_snapshot" if data.get("kind") == "web_snapshot" else "user_upload"
                ),
            }
        return data

    @model_validator(mode="after")
    def validate_source(self):
        if not self.title.strip() or not self.snippet.strip():
            raise ValueError("citation text must not be blank")
        if (self.origin == "web_snapshot") != (self.kind == "web_snapshot"):
            raise ValueError("citation origin contradicts its kind")
        if self.origin == "web_import" and not self.url:
            raise ValueError("an ingested web citation requires its source URL")
        if self.kind == "kb_chunk":
            if not all((self.knowledge_base_id, self.document_id,
                        self.source_version_id, self.chunk_id)) or self.snapshot_id:
                raise ValueError("knowledge citation has invalid provenance")
        elif not all((self.snapshot_id, self.url, self.fetched_at, self.content_hash)):
            raise ValueError("web citation has incomplete snapshot provenance")
        if self.url:
            parsed = urlsplit(self.url)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or parsed.username is not None or parsed.password is not None):
                raise ValueError("citation URL must be an uncredentialed HTTP URL")
        if self.locator and len(self.locator) > 8:
            raise ValueError("source locator is too large")
        return self


class KnowledgeEvidence(SourceCitation):
    text: str = Field(min_length=1, max_length=12000)

    def public_citation(self) -> SourceCitation:
        return SourceCitation.model_validate(self.model_dump(exclude={"text"}))


class KnowledgeRunState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    knowledge_base_id: str = Field(min_length=1, max_length=128)
    owner_id: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=0)
    epoch: int = Field(ge=0)
    web_enabled: bool = False
    session_id: str = Field(min_length=1, max_length=128)
    web_query: str = Field(default="", max_length=500)
    approved_web_urls: list[str] = Field(default_factory=list, max_length=32)
    outbound_blocked: bool = False
    # Egress counters. Defaulted so a session persisted before they existed
    # still deserializes; it starts from zero rather than being refused here.
    web_searches_used: int = Field(default=0, ge=0)
    web_fetches_used: int = Field(default=0, ge=0)
    evidence: dict[str, KnowledgeEvidence] = Field(default_factory=dict, max_length=64)

    @model_validator(mode="after")
    def validate_evidence_scope(self):
        if self.web_enabled and not self.web_query.strip():
            raise ValueError("web-enabled sessions require the original user search query")
        for url in self.approved_web_urls:
            parsed = urlsplit(url)
            if (len(url) > 2048 or parsed.scheme not in {"http", "https"}
                    or not parsed.hostname or parsed.username is not None or parsed.password is not None):
                raise ValueError("approved web URL is invalid")
        for key, evidence in self.evidence.items():
            if key != evidence.evidence_id:
                raise ValueError("evidence key does not match its identity")
            if evidence.kind == "kb_chunk" and evidence.knowledge_base_id != self.knowledge_base_id:
                raise ValueError("knowledge evidence escapes the persisted scope")
            if evidence.kind == "web_snapshot" and not self.web_enabled:
                raise ValueError("web evidence is not authorized in this session")
        return self

    def tool_names(self) -> set[str]:
        names = {"search_knowledge_base"}
        if self.web_enabled:
            names.update({"search_web", "fetch_web"})
        return names
