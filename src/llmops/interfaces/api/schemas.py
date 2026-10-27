"""Pydantic request and response schemas for public inference and admin APIs (Section 26)."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from llmops.domain.models.curation import ChatMessage

# --- Health & Status Schemas ---


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str = "ok"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ReadinessResponse(BaseModel):
    """Readiness probe response reflecting dependency health."""

    status: str
    checks: dict[str, bool]
    details: dict[str, str] = Field(default_factory=dict)


class VersionResponse(BaseModel):
    """System and model version information."""

    version: str
    git_commit: str
    active_model: str
    environment: str = "production"


# --- Inference & RAG Schemas ---


class SourceCitationResponse(BaseModel):
    """Provenance citation in controlled form for RAG responses (Section 26)."""

    document_id: str
    title: str
    url: str
    chunk_id: str


class ChatRequest(BaseModel):
    """Conversation inference request supporting multi-turn chat and optional RAG."""

    messages: list[ChatMessage]
    model: str | None = None
    stream: bool = False
    use_rag: bool = True
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)


class ChatResponse(BaseModel):
    """Controlled RAG chat answer schema conforming directly to Section 26."""

    answer: str
    model_version: str
    run_id: str
    sources: list[SourceCitationResponse] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    """Raw prompt generation request."""

    prompt: str
    model: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)


class GenerateResponse(BaseModel):
    """Raw generation response."""

    text: str
    model_version: str


# --- Retrieval Schemas ---


class RetrieveRequest(BaseModel):
    """Direct vector search query request."""

    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    source_name: str | None = None


class RetrievedChunkResponse(BaseModel):
    """A retrieved chunk with score and provenance."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    source_name: str
    source_url: str
    section_heading: str = ""


class RetrieveResponse(BaseModel):
    """Vector search results."""

    query: str
    chunks: list[RetrievedChunkResponse]
    latency_ms: float


# --- Model Management & Registry Schemas ---


class ModelInfo(BaseModel):
    """Model registry entry information."""

    model_version: str
    base_model: str
    stage: str
    approved: bool
    created_at: str
    evaluation_summary: dict[str, Any] | None = None


class ModelsListResponse(BaseModel):
    """List of available models in registry."""

    models: list[ModelInfo]
    active_model: str


# --- User Feedback Schemas ---


class FeedbackRequest(BaseModel):
    """User evaluation feedback on model responses (Section 26)."""

    query_id: str
    rating: int = Field(ge=1, le=5)
    comment: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackResponse(BaseModel):
    """Feedback receipt confirmation."""

    status: str = "received"
    feedback_id: str
