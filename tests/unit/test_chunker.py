"""Unit tests for the semantic document chunker."""

from llmops.domain.models.document import Document, Section
from llmops.infrastructure.chunker.semantic_chunker import SemanticChunker


def test_chunker_keeps_small_sections_intact() -> None:
    """Verify small sections produce single cohesive chunks."""
    sec1 = Section(
        section_id="sec-01",
        heading="Intro",
        level=1,
        content="FastAPI is a modern, fast web framework.",
    )
    sec2 = Section(
        section_id="sec-02",
        heading="Pydantic",
        level=2,
        content="Pydantic provides data validation using Python type annotations.",
    )
    doc = Document(
        document_id="doc-001",
        source_name="fastapi",
        source_url="https://fastapi.tiangolo.com/",
        title="FastAPI Overview",
        content="Combined text",
        sections=[sec1, sec2],
    )

    chunker = SemanticChunker(target_chunk_size=100, max_chunk_size=200)
    chunks = chunker.chunk_document(doc, dataset_version="v1", ingestion_run_id="run-101")

    assert len(chunks) == 2
    assert chunks[0].section_heading == "Intro"
    assert chunks[0].provenance.document_id == "doc-001"
    assert chunks[0].provenance.dataset_version == "v1"
    assert chunks[0].provenance.ingestion_run_id == "run-101"
    assert chunks[1].section_heading == "Pydantic"


def test_chunker_splits_large_content() -> None:
    """Verify content exceeding max_chunk_size is broken into multiple chunks."""
    paragraphs = [f"This is paragraph {i} explaining database queries." for i in range(20)]
    long_content = "\n\n".join(paragraphs)

    sec = Section(
        section_id="sec-01",
        heading="SQLAlchemy Queries",
        level=1,
        content=long_content,
    )
    doc = Document(
        document_id="doc-sql-001",
        source_name="sqlalchemy",
        source_url="https://docs.sqlalchemy.org/tutorial/",
        title="SQLAlchemy Tutorial",
        content=long_content,
        sections=[sec],
    )

    chunker = SemanticChunker(target_chunk_size=30, max_chunk_size=50)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 60
        assert chunk.provenance.source_name == "sqlalchemy"
        assert chunk.section_heading == "SQLAlchemy Queries"
