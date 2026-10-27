"""Domain models for RAG requests, citations, and grounded responses."""

from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Source citation identifying evidence used in a RAG answer."""

    document_id: str
    title: str = Field(default="")
    url: str
    chunk_id: str
    section_heading: str = Field(default="")
    relevance_score: float = Field(default=0.0)


class RAGRequest(BaseModel):
    """Request payload for RAG question-answering."""

    question: str
    source_name: str | None = None
    top_k: int = Field(default=3, ge=1, le=20)
    top_n: int = Field(default=10, ge=1, le=50)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=50, le=4096)
    prompt_version: str = Field(default="1.0.0")


class RAGResponse(BaseModel):
    """Grounded RAG response matching Section 26 & Section 67 of the specification."""

    answer: str
    model_version: str
    prompt_version: str
    run_id: str
    sources: list[Citation] = Field(default_factory=list)
    has_sufficient_context: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
