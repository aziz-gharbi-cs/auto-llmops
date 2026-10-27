"""Unit tests for retrieval domain models."""

from datetime import UTC, datetime

from llmops.domain.models.document import ChunkProvenance
from llmops.domain.models.retrieval import (
    RetrievalFilter,
    RetrievalMetrics,
    RetrievalQuery,
    RetrievedChunk,
)


def test_retrieved_chunk_model() -> None:
    """Verify RetrievedChunk structure, provenance, and scores."""
    prov = ChunkProvenance(
        document_id="doc-101",
        document_version="1.0.0",
        chunk_id="chunk-001",
        source_name="fastapi",
        source_url="https://fastapi.tiangolo.com/",
        dataset_version="v1",
        ingestion_run_id="run-01",
        content_hash="hash-101",
        text="Sample chunk content.",
        created_at=datetime.now(UTC),
    )

    chunk = RetrievedChunk(
        chunk_id="chunk-001",
        document_id="doc-101",
        content="Sample chunk content.",
        score=0.88,
        rerank_score=0.95,
        section_heading="Overview",
        provenance=prov,
    )

    assert chunk.score == 0.88
    assert chunk.rerank_score == 0.95
    assert chunk.provenance.source_name == "fastapi"


def test_retrieval_query_and_filter() -> None:
    """Verify RetrievalQuery default bounds and filtering."""
    flt = RetrievalFilter(source_name="fastapi", dataset_version="v1")
    query = RetrievalQuery(
        query_text="How to define path parameters?",
        top_n=15,
        top_k=4,
        filter_criteria=flt,
    )
    assert query.top_n == 15
    assert query.top_k == 4
    assert query.filter_criteria is not None
    assert query.filter_criteria.source_name == "fastapi"


def test_retrieval_metrics_model() -> None:
    """Verify RetrievalMetrics fields."""
    metrics = RetrievalMetrics(k=3, recall_at_k=1.0, precision_at_k=0.6667, mrr=1.0, ndcg=0.85)
    assert metrics.k == 3
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 1.0
