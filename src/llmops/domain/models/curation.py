"""Domain models for synthetic data generation, LLM judging, and curated datasets."""

import hashlib
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class JudgeScore(BaseModel):
    """Structured LLM-as-a-Judge evaluation score conforming to Section 17 of the specification."""

    model_config = ConfigDict(populate_by_name=True)

    relevance: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    instruction_following: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    overall: float = Field(ge=1.0, le=5.0)
    is_pass: bool = Field(
        validation_alias=AliasChoices("pass", "is_pass"),
        serialization_alias="pass",
    )
    reason: str = Field(default="")


class SyntheticQACandidate(BaseModel):
    """A generated developer question and answer candidate pair."""

    instruction: str
    response: str


class ChatMessage(BaseModel):
    """A single message in standard role-content conversation format."""

    role: str
    content: str


class ExampleMetadata(BaseModel):
    """Audit metadata tracked outside the model-facing message payload (Section 11)."""

    example_id: str
    source_document_id: str
    source_url: str
    chunk_id: str
    generation_model: str
    generation_prompt_version: str
    judge_model: str
    judge_prompt_version: str
    judge_score: JudgeScore
    dataset_version: str
    content_hash: str


class TrainingExample(BaseModel):
    """Complete training example with model-facing messages and audit metadata."""

    messages: list[ChatMessage]
    metadata: ExampleMetadata

    @classmethod
    def create(
        cls,
        example_id: str,
        system_prompt: str,
        instruction: str,
        response: str,
        source_document_id: str,
        source_url: str,
        chunk_id: str,
        generation_model: str,
        generation_prompt_version: str,
        judge_model: str,
        judge_prompt_version: str,
        judge_score: JudgeScore,
        dataset_version: str = "v1",
    ) -> "TrainingExample":
        """Factory helper creating a valid training example with deterministic content hash."""
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=instruction),
            ChatMessage(role="assistant", content=response),
        ]
        combined_text = f"{instruction}\n---\n{response}"
        content_hash = hashlib.sha256(combined_text.encode("utf-8")).hexdigest()

        meta = ExampleMetadata(
            example_id=example_id,
            source_document_id=source_document_id,
            source_url=source_url,
            chunk_id=chunk_id,
            generation_model=generation_model,
            generation_prompt_version=generation_prompt_version,
            judge_model=judge_model,
            judge_prompt_version=judge_prompt_version,
            judge_score=judge_score,
            dataset_version=dataset_version,
            content_hash=content_hash,
        )
        return cls(messages=messages, metadata=meta)


class DatasetSplitSummary(BaseModel):
    """Statistical summary of curated dataset split."""

    total_candidates: int = 0
    passed_candidates: int = 0
    rejected_candidates: int = 0
    pass_rate: float = 0.0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
