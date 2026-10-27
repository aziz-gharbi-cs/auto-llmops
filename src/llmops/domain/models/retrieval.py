"""Domain models for vector search, reranking, and retrieval metrics."""

from typing import Any

from pydantic import BaseModel, Field

from llmops.domain.models.document import ChunkProvenance


class RetrievedChunk(BaseModel):
    """A retrieved chunk enriched with similarity score, rerank score, and provenance."""

    chunk_id: str
    document_id: str
    content: str
    score: float = Field(default=0.0)
    rerank_score: float | None = None
    section_heading: str = Field(default="")
    provenance: ChunkProvenance
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalFilter(BaseModel):
    """Metadata filter criteria for vector search."""

    source_name: str | None = None
    document_id: str | None = None
    dataset_version: str | None = None


class RetrievalQuery(BaseModel):
    """Specification of a retrieval query across vector store and reranker."""

    query_text: str
    top_n: int = Field(default=10, ge=1, le=100)
    top_k: int = Field(default=3, ge=1, le=50)
    filter_criteria: RetrievalFilter | None = None


class RetrievalMetrics(BaseModel):
    """Standard information retrieval metrics (Recall@K, Precision@K, MRR, nDCG)."""

    k: int
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
