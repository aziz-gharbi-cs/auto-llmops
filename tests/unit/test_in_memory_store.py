"""Unit tests for the InMemoryVectorStore adapter."""

from datetime import UTC, datetime

from llmops.domain.models.document import ChunkProvenance, DocumentChunk
from llmops.infrastructure.vector.in_memory_store import InMemoryVectorStore


def make_doc_chunk(cid: str, source: str, content: str) -> DocumentChunk:
    """Helper to create a test DocumentChunk."""
    prov = ChunkProvenance(
        document_id=f"doc_{cid}",
        document_version="1.0.0",
        chunk_id=cid,
        source_name=source,
        source_url=f"https://example.com/{cid}",
        dataset_version="v1",
        ingestion_run_id="run-1",
        content_hash="h",
        text=content,
        created_at=datetime.now(UTC),
    )
    return DocumentChunk(
        chunk_id=cid,
        document_id=f"doc_{cid}",
        content=content,
        provenance=prov,
    )


def test_in_memory_store_workflow() -> None:
    """Verify in-memory vector store collection creation, upsert, search, and delete."""
    store = InMemoryVectorStore()
    col = "test_col"

    # 1. Collection creation
    store.create_collection(col, vector_size=2)
    assert store.collection_exists(col) is True

    # 2. Upsert chunks
    c1 = make_doc_chunk("c1", "fastapi", "FastAPI route handling")
    c2 = make_doc_chunk("c2", "pydantic", "Pydantic field validation")

    vec1 = [1.0, 0.0]
    vec2 = [0.0, 1.0]

    count = store.upsert_chunks(col, [c1, c2], [vec1, vec2])
    assert count == 2

    # 3. Search closest to vec1
    hits = store.search(col, query_vector=[1.0, 0.1], limit=1)
    assert len(hits) == 1
    assert hits[0].chunk_id == "c1"
    assert hits[0].score > 0.9

    # 4. Filtered search
    filtered_hits = store.search(
        col,
        query_vector=[1.0, 0.0],
        limit=2,
        filter_criteria={"source_name": "pydantic"},
    )
    assert len(filtered_hits) == 1
    assert filtered_hits[0].chunk_id == "c2"

    # 5. Delete collection
    assert store.delete_collection(col) is True
    assert store.collection_exists(col) is False
