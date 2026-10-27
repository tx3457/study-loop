"""Public response models for verifiable chunk-ID citations."""

from enum import Enum

from pydantic import BaseModel, Field


class GroundingStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    PENDING = "pending"
    CITATION_IDS_VALID = "citation_ids_valid"
    ABSTAINED = "abstained"


class CitationView(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int | None = None
    # PDF 片段所在页码，从 1 开始；非 PDF 或旧索引没有页码。
    page_number: int | None = Field(default=None, ge=1)
    rank: int = Field(ge=1)
    snippet: str
