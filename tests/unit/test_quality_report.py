"""Unit tests for dataset quality report generation."""

import json

from llmops.domain.models.document import ChunkProvenance, Document, DocumentChunk
from llmops.infrastructure.parser.quality_report import generate_quality_report


def test_generate_quality_report_stats() -> None:
    """Verify quality report correctly computes distribution and token statistics."""
    doc1 = Document(
        document_id="doc-1",
        source_name="fastapi",
        source_url="https://fastapi.tiangolo.com/",
        content="Content 1",
    )
    doc2 = Document(
        document_id="doc-2",
        source_name="pydantic",
        source_url="https://docs.pydantic.dev/",
        content="Content 2",
    )

    prov = ChunkProvenance(
        document_id="doc-1",
        document_version="1.0.0",
        chunk_id="c-1",
        source_name="fastapi",
        source_url="https://fastapi.tiangolo.com/",
        dataset_version="v1",
        ingestion_run_id="run-01",
        content_hash="hash1",
        text="text",
    )
    chunk1 = DocumentChunk(
        chunk_id="c-1",
        document_id="doc-1",
        content="Short chunk",
        token_count=10,
        provenance=prov,
    )
    chunk2 = DocumentChunk(
        chunk_id="c-2",
        document_id="doc-2",
        content="A much longer chunk with more words",
        token_count=30,
        provenance=prov,
    )

    report = generate_quality_report(
        documents=[doc1, doc2],
        chunks=[chunk1, chunk2],
        exact_dups=2,
        near_dups=1,
        rejected=0,
        dataset_version="v1",
    )

    assert report.total_documents == 2
    assert report.total_chunks == 2
    assert report.exact_duplicates_removed == 2
    assert report.near_duplicates_removed == 1
    assert report.source_distribution == {"fastapi": 1, "pydantic": 1}
    assert report.chunk_token_stats.min_tokens == 10
    assert report.chunk_token_stats.max_tokens == 30
    assert report.chunk_token_stats.mean_tokens == 20.0
    assert report.chunk_token_stats.total_tokens == 40

    # Ensure valid JSON output
    data = json.loads(report.model_dump_json())
    assert data["total_chunks"] == 2
    assert data["chunk_token_stats"]["total_tokens"] == 40
