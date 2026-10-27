"""Unit tests for the RAGService."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from llmops.application.use_cases.rag_service import INSUFFICIENT_CONTEXT_MESSAGE, RAGService
from llmops.domain.models.document import ChunkProvenance
from llmops.domain.models.rag import RAGRequest
from llmops.domain.models.retrieval import RetrievedChunk
from llmops.infrastructure.llm.mock_llm import MockLLMClient


def make_retrieved_chunk(cid: str, text: str, score: float = 0.9) -> RetrievedChunk:
    prov = ChunkProvenance(
        document_id=f"doc_{cid}",
        document_version="1.0.0",
        chunk_id=cid,
        source_name="fastapi",
        source_url=f"https://fastapi.tiangolo.com/{cid}",
        dataset_version="v1",
        ingestion_run_id="run-1",
        content_hash="hash",
        text=text,
        created_at=datetime.now(UTC),
    )
    return RetrievedChunk(
        chunk_id=cid,
        document_id=f"doc_{cid}",
        content=text,
        score=score,
        rerank_score=score,
        section_heading="Tutorial",
        provenance=prov,
    )


def test_rag_service_success() -> None:
    """Verify RAGService returns grounded answer and attached citations."""
    mock_retrieval = MagicMock()
    mock_chunk = make_retrieved_chunk("c1", "FastAPI path parameters tutorial text.", score=0.95)
    mock_retrieval.retrieve.return_value = [mock_chunk]

    mock_llm = MockLLMClient(
        model_name="gemma-3-1b",
        default_response="To declare a path parameter, use Python format syntax as seen in [c1].",
    )

    rag = RAGService(retrieval_service=mock_retrieval, llm_client=mock_llm)
    req = RAGRequest(question="How do I declare path parameters in FastAPI?", top_k=1)
    response = rag.answer_question(req)

    assert response.has_sufficient_context is True
    assert "path parameter" in response.answer
    assert response.model_version == "gemma-3-1b"
    assert len(response.sources) == 1
    assert response.sources[0].chunk_id == "c1"
    assert response.sources[0].url == "https://fastapi.tiangolo.com/c1"


def test_rag_service_insufficient_evidence() -> None:
    """Verify RAGService returns graceful fallback when no chunks pass relevance threshold."""
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = []  # No candidate chunks found

    mock_llm = MockLLMClient()
    rag = RAGService(retrieval_service=mock_retrieval, llm_client=mock_llm)
    req = RAGRequest(question="What is the internal code of unrelated topic?")
    response = rag.answer_question(req)

    assert response.has_sufficient_context is False
    assert response.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert len(response.sources) == 0
