"""Unit tests for synthetic Q&A generation from document chunks."""

from llmops.domain.models.document import ChunkProvenance, DocumentChunk, compute_sha256
from llmops.infrastructure.curation.synthetic_generator import SyntheticQAGenerator
from llmops.infrastructure.llm.mock_llm import MockLLMClient


def _sample_chunk() -> DocumentChunk:
    chunk_text = (
        "FastAPI allows defining query parameters with default values and type annotations."
    )
    return DocumentChunk(
        chunk_id="chunk-fastapi-001",
        document_id="doc-fastapi-001",
        content=chunk_text,
        token_count=12,
        section_heading="Query Parameters",
        provenance=ChunkProvenance(
            document_id="doc-fastapi-001",
            document_version="1.0.0",
            chunk_id="chunk-fastapi-001",
            source_name="fastapi",
            source_url="https://fastapi.tiangolo.com/tutorial/query-params/",
            dataset_version="v1",
            ingestion_run_id="run-001",
            content_hash=compute_sha256(chunk_text),
            text=chunk_text,
        ),
    )


def test_synthetic_generator_structured() -> None:
    payload = {
        "instruction": "How do I define query parameters with defaults in FastAPI?",
        "response": "Specify default values directly in the path operation function signature.",
    }
    client = MockLLMClient(structured_payload=payload)
    generator = SyntheticQAGenerator(llm_client=client)

    chunk = _sample_chunk()
    candidate, version = generator.generate_candidate(chunk)

    assert candidate.instruction == payload["instruction"]
    assert candidate.response == payload["response"]
    assert version == "1.0.0"


def test_synthetic_generator_fallback() -> None:
    """When structured generation raises error, fallback heuristic generates Q&A."""
    client = MockLLMClient(structured_payload={})  # Missing required fields triggers fallback
    generator = SyntheticQAGenerator(llm_client=client)

    chunk = _sample_chunk()
    candidate, version = generator.generate_candidate(chunk)

    assert "Query Parameters" in candidate.instruction
    assert "fastapi" in candidate.instruction
    assert candidate.response == chunk.content
    assert version == "1.0.0"
