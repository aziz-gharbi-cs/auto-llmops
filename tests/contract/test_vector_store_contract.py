"""Contract test for BaseVectorStore implementations."""

from datetime import UTC, datetime

import pytest

from llmops.domain.interfaces.vector_store import BaseVectorStore
from llmops.domain.models.document import ChunkProvenance, DocumentChunk
from llmops.infrastructure.vector.in_memory_store import InMemoryVectorStore


@pytest.fixture
def vector_store() -> BaseVectorStore:
    """Fixture providing a BaseVectorStore instance."""
    return InMemoryVectorStore()


def verify_vector_store_contract(store: BaseVectorStore) -> None:
    """Generic contract verification asserting expected BaseVectorStore behavior."""
    col = "contract_collection"

    # 1. Collection creation
    assert store.collection_exists(col) is False
    store.create_collection(col, vector_size=3)
    assert store.collection_exists(col) is True

    # 2. Upsert chunks
    prov = ChunkProvenance(
        document_id="doc-contract-1",
        document_version="1.0.0",
        chunk_id="chunk-c1",
        source_name="fastapi",
        source_url="https://fastapi.tiangolo.com/",
        dataset_version="v1",
        ingestion_run_id="run-c1",
        content_hash="hash-c1",
        text="FastAPI validation logic",
        created_at=datetime.now(UTC),
    )
    chunk = DocumentChunk(
        chunk_id="chunk-c1",
        document_id="doc-contract-1",
        content="FastAPI validation logic",
        provenance=prov,
    )
    vector = [0.0, 1.0, 0.0]

    upserted = store.upsert_chunks(col, [chunk], [vector])
    assert upserted == 1

    # 3. Search
    results = store.search(col, query_vector=[0.0, 0.9, 0.1], limit=5)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-c1"
    assert results[0].score > 0.8
    assert results[0].provenance.source_name == "fastapi"

    # 4. Filter search
    filtered_empty = store.search(
        col, query_vector=[0.0, 1.0, 0.0], filter_criteria={"source_name": "sqlalchemy"}
    )
    assert len(filtered_empty) == 0

    # 5. Delete collection
    assert store.delete_collection(col) is True
    assert store.collection_exists(col) is False


def test_in_memory_store_fulfills_vector_contract(vector_store: BaseVectorStore) -> None:
    """Verify InMemoryVectorStore fulfills BaseVectorStore contract."""
    verify_vector_store_contract(vector_store)
