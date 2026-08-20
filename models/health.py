"""Stable public health-response models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class StorageComponentHealth(BaseModel):
    """One storage dependency without operational secrets."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "unavailable", "not_configured"]
    required: bool
    code: Literal[
        "chroma_ready",
        "chroma_unavailable",
        "postgres_ready",
        "postgres_unavailable",
        "postgres_not_configured",
        "local_state_ready",
        "local_state_unavailable",
        "local_state_not_configured",
    ]


class StorageHealthChecks(BaseModel):
    """The storage components required by the selected deployment mode."""

    model_config = ConfigDict(extra="forbid")

    chroma: StorageComponentHealth
    postgres: StorageComponentHealth
    local_state: StorageComponentHealth


class StorageReadinessResponse(BaseModel):
    """Readiness envelope shared by the 200 and 503 responses."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["StudyLoop"]
    status: Literal["ready", "unready"]
    code: Literal["storage_ready", "storage_unavailable"]
    checks: StorageHealthChecks
