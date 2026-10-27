"""Unit tests for the exact and near-duplicate deduplication engine."""

from llmops.domain.models.document import Document
from llmops.infrastructure.parser.deduplicator import (
    Deduplicator,
    compute_shingles,
    jaccard_similarity,
)


def test_jaccard_similarity() -> None:
    """Verify Jaccard calculation on overlapping and disjoint sets."""
    set1 = {"apple", "banana", "cherry"}
    set2 = {"apple", "banana", "orange"}
    # intersection: 2 (apple, banana), union: 4 (apple, banana, cherry, orange) -> 0.5
    assert jaccard_similarity(set1, set2) == 0.5
    assert jaccard_similarity(set1, set1) == 1.0
    assert jaccard_similarity(set1, set()) == 0.0


def test_compute_shingles() -> None:
    """Verify character k-shingle generation."""
    shingles = compute_shingles("fastapi", k=3)
    assert "fas" in shingles
    assert "ast" in shingles
    assert "api" in shingles


def test_deduplicator_removes_exact_duplicates() -> None:
    """Verify exact duplicate documents with identical content_hash are filtered."""
    doc1 = Document(
        document_id="doc-1",
        source_name="fastapi",
        source_url="https://fastapi.tiangolo.com/1",
        content="Identical content for testing deduplication.",
    )
    doc2 = Document(
        document_id="doc-2",
        source_name="fastapi",
        source_url="https://fastapi.tiangolo.com/2",
        content="Identical content for testing deduplication.",
    )
    doc3 = Document(
        document_id="doc-3",
        source_name="pydantic",
        source_url="https://docs.pydantic.dev/1",
        content="Completely different content about Pydantic models.",
    )

    dedup = Deduplicator()
    unique, exact_count, near_count = dedup.deduplicate([doc1, doc2, doc3])

    assert len(unique) == 2
    assert exact_count == 1
    assert near_count == 0
    assert unique[0].document_id == "doc-1"
    assert unique[1].document_id == "doc-3"


def test_deduplicator_removes_near_duplicates() -> None:
    """Verify near-identical documents with tiny variance are filtered by Jaccard."""
    base_text = (
        "FastAPI is a modern, fast (high-performance), web framework for building APIs "
        "with Python 3.8+ based on standard Python type hints."
    )
    variant_text = (
        "FastAPI is a modern, fast (high-performance), web framework for building APIs "
        "with Python 3.8+ based on standard Python type hints! Extra word."
    )
    different_text = "SQLAlchemy is the Python SQL toolkit and Object Relational Mapper."

    doc1 = Document(
        document_id="doc-1",
        source_name="fastapi",
        source_url="https://fastapi.tiangolo.com/1",
        content=base_text,
    )
    doc2 = Document(
        document_id="doc-2",
        source_name="fastapi",
        source_url="https://fastapi.tiangolo.com/2",
        content=variant_text,
    )
    doc3 = Document(
        document_id="doc-3",
        source_name="sqlalchemy",
        source_url="https://docs.sqlalchemy.org/",
        content=different_text,
    )

    dedup = Deduplicator(near_duplicate_threshold=0.80, shingle_size=4)
    unique, exact_count, near_count = dedup.deduplicate([doc1, doc2, doc3])

    assert len(unique) == 2
    assert exact_count == 0
    assert near_count == 1
    assert unique[0].document_id == "doc-1"
    assert unique[1].document_id == "doc-3"
