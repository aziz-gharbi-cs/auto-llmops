"""Unit tests for the canonical Document, Section, and Chunk models."""

import hashlib

from llmops.domain.models.document import (
    ChunkProvenance,
    Document,
    DocumentChunk,
    Section,
    compute_sha256,
)


def test_compute_sha256() -> None:
    """Verify deterministic SHA-256 digest computation."""
    text = "Hello LLMOps"
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert compute_sha256(text) == expected


def test_section_auto_computes_hash() -> None:
    """Verify Section computes content_hash automatically."""
    section = Section(
        section_id="sec-1",
        heading="Introduction",
        level=1,
        content="This is the introduction section.",
    )
    assert section.content_hash == compute_sha256("This is the introduction section.")


def test_document_model_creation_and_hash() -> None:
    """Verify Document auto-computes content_hash and serializes."""
    content = "# FastAPI\nFastAPI is a modern web framework."
    doc = Document(
        document_id="doc-fastapi-001",
        source_name="fastapi",
        source_url="https://fastapi.tiangolo.com/",
        title="FastAPI Overview",
        version="1.0.0",
        content=content,
        metadata={"author": "tiangolo"},
    )
    assert doc.content_hash == compute_sha256(content)
    assert doc.source_name == "fastapi"
    assert doc.metadata["author"] == "tiangolo"

    # Serialization test
    doc_dict = doc.model_dump()
    assert doc_dict["document_id"] == "doc-fastapi-001"
    assert doc_dict["content_hash"] == doc.content_hash


def test_document_chunk_and_provenance() -> None:
    """Verify DocumentChunk with nested ChunkProvenance."""
    chunk_text = "FastAPI uses Pydantic for data validation."
    provenance = ChunkProvenance(
        document_id="doc-fastapi-001",
        document_version="1.0.0",
        chunk_id="chunk-001",
        source_name="fastapi",
        source_url="https://fastapi.tiangolo.com/",
        dataset_version="v1",
        ingestion_run_id="run-12345",
        content_hash=compute_sha256(chunk_text),
        text=chunk_text,
    )

    chunk = DocumentChunk(
        chunk_id="chunk-001",
        document_id="doc-fastapi-001",
        content=chunk_text,
        token_count=8,
        section_heading="Data Validation",
        provenance=provenance,
    )

    assert chunk.content_hash == compute_sha256(chunk_text)
    assert chunk.provenance.source_name == "fastapi"
    assert chunk.provenance.dataset_version == "v1"
    assert chunk.provenance.ingestion_run_id == "run-12345"
