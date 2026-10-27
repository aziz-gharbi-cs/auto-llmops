"""Canonical document and chunk data models representing ingested knowledge."""

import hashlib
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


def compute_sha256(text: str) -> str:
    """Compute deterministic SHA-256 hex digest for a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Section(BaseModel):
    """A semantic section within a canonical document."""

    section_id: str
    heading: str = Field(default="")
    level: int = Field(default=1, ge=1, le=6)
    content: str
    content_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Ensure content hash is computed if not provided."""
        if not self.content_hash and self.content:
            object.__setattr__(self, "content_hash", compute_sha256(self.content))


class Document(BaseModel):
    """Canonical document representation independent of raw acquisition source."""

    document_id: str
    source_name: str
    source_url: str
    title: str = Field(default="")
    version: str = Field(default="1.0.0")
    content: str
    content_hash: str = Field(default="")
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    parser_version: str = Field(default="1.0.0")
    metadata: dict[str, Any] = Field(default_factory=dict)
    sections: list[Section] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Ensure content hash is computed if not provided."""
        if not self.content_hash and self.content:
            object.__setattr__(self, "content_hash", compute_sha256(self.content))


class ChunkProvenance(BaseModel):
    """Provenance metadata tracking exactly where a chunk originated."""

    document_id: str
    document_version: str
    chunk_id: str
    source_name: str
    source_url: str
    dataset_version: str = Field(default="v1")
    ingestion_run_id: str
    content_hash: str
    text: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """A granular chunk of a canonical document ready for embedding and retrieval."""

    chunk_id: str
    document_id: str
    content: str
    content_hash: str = Field(default="")
    token_count: int = Field(default=0, ge=0)
    section_heading: str = Field(default="")
    provenance: ChunkProvenance

    def model_post_init(self, __context: Any) -> None:
        """Ensure content hash is computed."""
        if not self.content_hash and self.content:
            object.__setattr__(self, "content_hash", compute_sha256(self.content))
