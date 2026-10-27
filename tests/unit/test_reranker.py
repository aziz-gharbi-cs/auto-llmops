"""Unit tests for the second-stage reranker."""

from datetime import UTC, datetime

from llmops.domain.models.document import ChunkProvenance
from llmops.domain.models.retrieval import RetrievedChunk
from llmops.infrastructure.reranker.cross_encoder import (
    CrossEncoderReranker,
    compute_lexical_overlap_score,
)


def make_chunk(cid: str, text: str, initial_score: float) -> RetrievedChunk:
    """Helper to construct RetrievedChunk."""
    prov = ChunkProvenance(
        document_id="doc-1",
        document_version="1.0.0",
        chunk_id=cid,
        source_name="fastapi",
        source_url="https://fastapi.tiangolo.com/",
        dataset_version="v1",
        ingestion_run_id="run-1",
        content_hash="hash",
        text=text,
        created_at=datetime.now(UTC),
    )
    return RetrievedChunk(
        chunk_id=cid,
        document_id="doc-1",
        content=text,
        score=initial_score,
        provenance=prov,
    )


def test_compute_lexical_overlap_score() -> None:
    """Verify lexical overlap scoring."""
    query = "path parameters in fastapi"
    text_relevant = "FastAPI allows defining path parameters easily."
    text_irrelevant = "SQLAlchemy database migrations with Alembic."

    score_rel = compute_lexical_overlap_score(query, text_relevant)
    score_irrel = compute_lexical_overlap_score(query, text_irrelevant)
    assert score_rel > score_irrel


def test_reranker_reorders_and_cuts_top_k() -> None:
    """Verify reranker modifies ranking and returns exact top_k."""
    chunk_low_sim = make_chunk(
        "c1",
        "FastAPI path parameters allow extracting typed variables from the URL path.",
        initial_score=0.4,
    )
    chunk_high_sim = make_chunk(
        "c2",
        "Completely unrelated text with high random vector overlap.",
        initial_score=0.9,
    )
    chunk_mid = make_chunk(
        "c3",
        "Pydantic base model configuration.",
        initial_score=0.5,
    )

    reranker = CrossEncoderReranker(use_fallback=True)
    results = reranker.rerank(
        query="path parameters in fastapi",
        chunks=[chunk_high_sim, chunk_low_sim, chunk_mid],
        top_k=2,
    )

    assert len(results) == 2
    # The relevant chunk (c1) should be boosted to top rank due to term matching
    assert results[0].chunk_id == "c1"
    assert results[0].rerank_score is not None
