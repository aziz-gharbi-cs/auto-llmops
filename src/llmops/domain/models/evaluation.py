"""Domain models for multi-layer evaluation and quality gate (Section 19/20/22)."""

from datetime import UTC, datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, Field


class LayerAResults(BaseModel):
    """Layer A: Deterministic data validation checks (Section 19)."""

    total_records: int
    valid_records: int
    schema_errors: int = 0
    duplicate_count: int = 0
    leakage_detected: bool = False
    passed: bool = True
    details: dict[str, Any] = Field(default_factory=dict)


class LayerBResults(BaseModel):
    """Layer B: Information retrieval evaluation (Section 19)."""

    recall_at_k: float = Field(ge=0.0, le=1.0)
    precision_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    ndcg: float = Field(ge=0.0, le=1.0)
    context_relevance: float = Field(ge=0.0, le=5.0)
    passed: bool = True


class LayerCResults(BaseModel):
    """Layer C: Task generation quality metrics (Section 19)."""

    correctness: float = Field(ge=1.0, le=5.0)
    relevance: float = Field(ge=1.0, le=5.0)
    groundedness: float = Field(ge=1.0, le=5.0)
    instruction_following: float = Field(ge=1.0, le=5.0)
    clarity: float = Field(ge=1.0, le=5.0)
    overall: float = Field(ge=1.0, le=5.0)
    passed: bool = True


class LayerDResults(BaseModel):
    """Layer D: Head-to-head model comparison on held-out benchmark (Section 19)."""

    base_model_name: str
    candidate_model_name: str
    base_overall_score: float = Field(ge=1.0, le=5.0)
    candidate_overall_score: float = Field(ge=1.0, le=5.0)
    delta: float
    candidate_won: bool
    passed: bool = True


class LayerEResults(BaseModel):
    """Layer E: System-level comparison (Base vs Fine-Tuned vs Fine-Tuned + RAG)."""

    base_score: float = Field(ge=1.0, le=5.0)
    fine_tuned_score: float = Field(ge=1.0, le=5.0)
    rag_augmented_score: float = Field(ge=1.0, le=5.0)
    rag_grounding_rate: float = Field(ge=0.0, le=1.0)
    passed: bool = True


class QualityGateThresholds(BaseModel):
    """Configurable pass/fail criteria for model promotion."""

    min_retrieval_mrr: float = Field(
        default=0.5,
        validation_alias=AliasChoices("min_retrieval_mrr", "min_mrr"),
    )
    min_generation_groundedness: float = Field(
        default=3.5,
        validation_alias=AliasChoices("min_generation_groundedness", "min_groundedness"),
    )
    min_generation_correctness: float = Field(
        default=3.5,
        validation_alias=AliasChoices("min_generation_correctness", "min_correctness"),
    )
    min_candidate_win_margin: float = 0.0  # Non-negative delta vs baseline
    min_rag_grounding_rate: float = 0.75


class EvaluationReport(BaseModel):
    """Machine-readable multi-layer evaluation report artifact."""

    evaluation_id: str
    model_version: str
    dataset_version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    layer_a: LayerAResults
    layer_b: LayerBResults
    layer_c: LayerCResults
    layer_d: LayerDResults
    layer_e: LayerEResults
    quality_gate_passed: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    recommendation: str  # "APPROVE_FOR_STAGING" or "REJECT"
