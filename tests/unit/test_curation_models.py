"""Unit tests for curation domain models conforming to Sections 11 and 17."""

import pytest
from pydantic import ValidationError

from llmops.domain.models.curation import (
    ChatMessage,
    DatasetSplitSummary,
    ExampleMetadata,
    JudgeScore,
    SyntheticQACandidate,
    TrainingExample,
)


def test_judge_score_valid() -> None:
    score = JudgeScore(
        relevance=5,
        correctness=5,
        groundedness=4,
        instruction_following=5,
        clarity=4,
        overall=4.6,
        is_pass=True,
        reason="Excellent grounded answer",
    )
    assert score.is_pass is True
    assert score.overall == 4.6
    assert score.groundedness == 4


def test_judge_score_pass_alias() -> None:
    """Verify alias 'pass' works for JudgeScore serialization/deserialization."""
    raw = {
        "relevance": 4,
        "correctness": 4,
        "groundedness": 5,
        "instruction_following": 4,
        "clarity": 4,
        "overall": 4.2,
        "pass": True,
        "reason": "Meets criteria",
    }
    score = JudgeScore.model_validate(raw)
    assert score.is_pass is True


def test_judge_score_out_of_bounds() -> None:
    with pytest.raises(ValidationError):
        JudgeScore(
            relevance=6,  # max is 5
            correctness=5,
            groundedness=4,
            instruction_following=5,
            clarity=4,
            overall=4.6,
            is_pass=True,
        )


def test_synthetic_qa_candidate() -> None:
    candidate = SyntheticQACandidate(
        instruction="How do I use Depends in FastAPI?",
        response="Use Depends(dependency_function) in your route handler signature.",
    )
    assert "FastAPI" in candidate.instruction
    assert "Depends" in candidate.response


def test_chat_message() -> None:
    msg = ChatMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_training_example_create_factory() -> None:
    score = JudgeScore(
        relevance=5,
        correctness=5,
        groundedness=5,
        instruction_following=5,
        clarity=5,
        overall=5.0,
        is_pass=True,
    )
    example = TrainingExample.create(
        example_id="ex-1",
        system_prompt="You are an assistant.",
        instruction="What is Pydantic BaseModel?",
        response="BaseModel is the core data validation class in Pydantic.",
        source_document_id="doc-pydantic-1",
        source_url="https://docs.pydantic.dev/intro",
        chunk_id="chk-123",
        generation_model="gemma-3-1b",
        generation_prompt_version="1.0.0",
        judge_model="gemma-3-4b",
        judge_prompt_version="1.0.0",
        judge_score=score,
    )

    # Check 3-turn message structure
    assert len(example.messages) == 3
    assert example.messages[0].role == "system"
    assert example.messages[1].role == "user"
    assert example.messages[2].role == "assistant"

    # Check metadata isolation
    assert isinstance(example.metadata, ExampleMetadata)
    assert example.metadata.example_id == "ex-1"
    assert example.metadata.source_document_id == "doc-pydantic-1"
    assert len(example.metadata.content_hash) == 64  # SHA-256


def test_dataset_split_summary() -> None:
    summary = DatasetSplitSummary(
        total_candidates=100,
        passed_candidates=85,
        rejected_candidates=15,
        pass_rate=0.85,
        train_count=68,
        val_count=8,
        test_count=9,
    )
    assert summary.passed_candidates == 85
    assert summary.pass_rate == 0.85
