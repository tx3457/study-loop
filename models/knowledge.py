"""Typed public contracts for the optional knowledge-base gateway."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class KnowledgeCapabilities(_StrictModel):
    enabled: bool
    available: bool
    web_search_available: bool


class KnowledgeScope(_StrictModel):
    knowledge_base_id: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=0, strict=True)
    epoch: int = Field(ge=0, strict=True)
    status: Literal["ready", "updating", "dirty", "deleting"] = "ready"


class KnowledgeBaseCreate(_StrictModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)


class KnowledgeBaseUpdate(_StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=4000)
    expected_revision: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def require_metadata_change(self):
        if self.name is None and self.description is None:
            raise ValueError("at least one metadata field is required")
        return self


class RevisionRequest(_StrictModel):
    expected_revision: int = Field(ge=0, strict=True)


class LegacyImportRequest(RevisionRequest):
    legacy_document_id: str = Field(min_length=1, max_length=255)


class WebImportRequest(RevisionRequest):
    snapshot_id: str = Field(min_length=1, max_length=128)


class ParsedBlock(_StrictModel):
    text: str = Field(min_length=1, max_length=2_000_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CorrectionRequest(RevisionRequest):
    kind: Literal[
        "rename_entity", "merge_entities", "delete_entity", "delete_relation"
    ]
    entity_id: str | None = Field(default=None, min_length=1, max_length=128)
    entity_ids: list[str] | None = Field(default=None, min_length=1, max_length=100)
    target_id: str | None = Field(default=None, min_length=1, max_length=128)
    edge_id: str | None = Field(default=None, min_length=1, max_length=128)
    label: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_kind_fields(self):
        required = {
            "rename_entity": ("entity_id", "label"),
            "merge_entities": ("entity_ids", "target_id"),
            "delete_entity": ("entity_id",),
            "delete_relation": ("edge_id",),
        }[self.kind]
        if any(getattr(self, field) is None for field in required):
            raise ValueError(f"missing fields for {self.kind}")
        if self.kind == "merge_entities" and (
            self.target_id in self.entity_ids
            or len(set(self.entity_ids)) != len(self.entity_ids)
        ):
            raise ValueError("merge sources must be distinct and exclude the target")
        allowed = set(required) | {"kind", "expected_revision"}
        for field in ("entity_id", "entity_ids", "target_id", "edge_id", "label"):
            if field not in allowed and getattr(self, field) is not None:
                raise ValueError(f"unexpected field for {self.kind}")
        return self
