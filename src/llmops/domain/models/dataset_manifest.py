"""Dataset version manifest providing full auditability and provenance."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class DatasetFileRecord(BaseModel):
    """File record within a dataset version."""

    filename: str
    sha256_hash: str
    byte_size: int


class DatasetVersionManifest(BaseModel):
    """Immutable dataset version manifest linking Git, DVC, files, and storage."""

    dataset_version: str
    dataset_name: str = "documentation-assistant-dataset"
    git_commit: str = Field(default="")
    dvc_revision: str = Field(default="")
    content_hash: str
    storage_uri: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    num_documents: int = Field(default=0, ge=0)
    num_chunks: int = Field(default=0, ge=0)
    files: list[DatasetFileRecord] = Field(default_factory=list)
    quality_report_uri: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)
