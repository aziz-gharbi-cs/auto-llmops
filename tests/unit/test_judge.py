"""Unit tests for DataCurationJudge evaluating Q&A candidates."""

from llmops.infrastructure.curation.judge import DataCurationJudge
from llmops.infrastructure.llm.mock_llm import MockLLMClient


def test_judge_structured_evaluation() -> None:
    score_payload = {
        "relevance": 5,
        "correctness": 5,
        "groundedness": 5,
        "instruction_following": 5,
        "clarity": 5,
        "overall": 5.0,
        "pass": True,
        "reason": "Perfect factual groundedness and clear code examples.",
    }
    client = MockLLMClient(
        model_name="judge-model-v1",
        structured_payload=score_payload,
    )
    judge = DataCurationJudge(llm_client=client)

    score, version = judge.evaluate(
        context="FastAPI provides APIRouter to structure large applications into multiple files.",
        instruction="What does APIRouter do in FastAPI?",
        response="APIRouter allows structuring large applications into multiple files and modules.",
        threshold=4.0,
    )

    assert score.is_pass is True
    assert score.overall == 5.0
    assert version == "1.0.0"
    assert judge.model_name == "judge-model-v1"


def test_judge_heuristic_fallback_pass() -> None:
    """When LLM returns empty payload, heuristic fallback passes well-grounded response."""
    client = MockLLMClient(structured_payload={})
    judge = DataCurationJudge(llm_client=client)

    context = (
        "SQLAlchemy AsyncSession provides non-blocking database queries with greenlet support."
    )
    instruction = "How does SQLAlchemy support non-blocking async queries?"
    response = "SQLAlchemy provides AsyncSession for non-blocking database queries."

    score, _ = judge.evaluate(
        context=context,
        instruction=instruction,
        response=response,
        threshold=3.5,
    )

    assert score.is_pass is True
    assert score.overall >= 3.5
    assert "heuristic" in score.reason.lower()


def test_judge_heuristic_fallback_fail() -> None:
    """Heuristic fallback fails completely irrelevant or very short responses."""
    client = MockLLMClient(structured_payload={})
    judge = DataCurationJudge(llm_client=client)

    score, _ = judge.evaluate(
        context="Some complex documentation about database connections and pooling.",
        instruction="Short?",
        response="No.",
        threshold=3.5,
    )

    assert score.is_pass is False
    assert score.overall < 3.5
