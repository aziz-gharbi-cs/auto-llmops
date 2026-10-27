"""Unit tests for QdrantVectorStore using mock HTTP transport."""

from datetime import UTC, datetime

import httpx

from llmops.domain.models.document import ChunkProvenance, DocumentChunk
from llmops.infrastructure.vector.qdrant_store import QdrantVectorStore


def test_qdrant_vector_store_workflow() -> None:
    """Verify QdrantVectorStore sends correct REST payloads and parses responses."""

    def handler(request: httpx.Request) -> httpx.Response:
        url_path = request.url.path
        method = request.method

        if method == "PUT" and url_path == "/collections/test_col":
            return httpx.Response(200, json={"result": True, "status": "ok"})

        if method == "GET" and url_path == "/collections/test_col":
            return httpx.Response(200, json={"result": {"status": "green"}})

        if method == "PUT" and url_path == "/collections/test_col/points":
            return httpx.Response(200, json={"result": {"operation_id": 1, "status": "completed"}})

        if method == "POST" and url_path == "/collections/test_col/points/search":
            return httpx.Response(
                200,
                json={
                    "result": [
                        {
                            "id": "c9bf9e57-1685-4c89-bafb-ff5af830be8a",
                            "score": 0.94,
                            "payload": {
                                "chunk_id": "chunk-101",
                                "document_id": "doc-101",
                                "text": "FastAPI dependency injection system.",
                                "source_name": "fastapi",
                                "source_url": "https://fastapi.tiangolo.com/dependencies/",
                                "section_heading": "Dependencies",
                            },
                        }
                    ]
                },
            )

        if method == "DELETE" and url_path == "/collections/test_col":
            return httpx.Response(200, json={"result": True, "status": "ok"})

        return httpx.Response(404, json={"status": "not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:6333")
    store = QdrantVectorStore(client=client)

    # 1. create & exists
    store.create_collection("test_col", vector_size=384)
    assert store.collection_exists("test_col") is True

    # 2. upsert
    prov = ChunkProvenance(
        document_id="doc-101",
        document_version="1.0.0",
        chunk_id="chunk-101",
        source_name="fastapi",
        source_url="https://fastapi.tiangolo.com/dependencies/",
        dataset_version="v1",
        ingestion_run_id="run-1",
        content_hash="h101",
        text="FastAPI dependency injection system.",
        created_at=datetime.now(UTC),
    )
    chunk = DocumentChunk(
        chunk_id="chunk-101",
        document_id="doc-101",
        content="FastAPI dependency injection system.",
        provenance=prov,
    )
    upserted = store.upsert_chunks("test_col", [chunk], [[0.1] * 384])
    assert upserted == 1

    # 3. search
    results = store.search("test_col", [0.1] * 384, limit=5)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-101"
    assert results[0].score == 0.94
    assert results[0].section_heading == "Dependencies"
    assert results[0].provenance.source_name == "fastapi"

    # 4. delete
    assert store.delete_collection("test_col") is True
