"""Unit tests for the end-to-end RetrievalService."""

from datetime import UTC, datetime

from llmops.application.use_cases.retrieval_service import RetrievalService
from llmops.domain.models.document import ChunkProvenance, DocumentChunk
from llmops.domain.models.retrieval import RetrievalQuery
from llmops.infrastructure.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from llmops.infrastructure.reranker.cross_encoder import CrossEncoderReranker
from llmops.infrastructure.vector.in_memory_store import InMemoryVectorStore


def test_retrieval_service_flow() -> None:
    """Verify RetrievalService index_chunks and two-stage retrieve workflow."""
    vector_store = InMemoryVectorStore()
    embedding_model = SentenceTransformerEmbeddings(use_fallback=True)
    reranker = CrossEncoderReranker(use_fallback=True)

    service = RetrievalService(
        vector_store=vector_store,
        embedding_model=embedding_model,
        reranker=reranker,
        default_collection="test_retrieval_collection",
    )

    prov1 = ChunkProvenance(
        document_id="doc-fastapi",
        document_version="1.0.0",
        chunk_id="c-fastapi-01",
        source_name="fastapi",
        source_url="https://fastapi.tiangolo.com/tutorial/",
        dataset_version="v1",
        ingestion_run_id="run-1",
        content_hash="h1",
        text="FastAPI makes it easy to build APIs with Python typing.",
        created_at=datetime.now(UTC),
    )
    c1 = DocumentChunk(
        chunk_id="c-fastapi-01",
        document_id="doc-fastapi",
        content="FastAPI makes it easy to build APIs with Python typing.",
        provenance=prov1,
    )

    prov2 = ChunkProvenance(
        document_id="doc-sqlalchemy",
        document_version="1.0.0",
        chunk_id="c-sql-01",
        source_name="sqlalchemy",
        source_url="https://docs.sqlalchemy.org/tutorial/",
        dataset_version="v1",
        ingestion_run_id="run-1",
        content_hash="h2",
        text="SQLAlchemy provides a complete suite of persistence patterns.",
        created_at=datetime.now(UTC),
    )
    c2 = DocumentChunk(
        chunk_id="c-sql-01",
        document_id="doc-sqlalchemy",
        content="SQLAlchemy provides a complete suite of persistence patterns.",
        provenance=prov2,
    )

    # 1. Index chunks
    indexed = service.index_chunks([c1, c2])
    assert indexed == 2

    # 2. Retrieve query
    query = RetrievalQuery(
        query_text="build APIs with FastAPI",
        top_n=5,
        top_k=2,
    )
    results = service.retrieve(query)
    assert len(results) == 2
    # c1 is top match
    assert results[0].chunk_id == "c-fastapi-01"
    assert results[0].rerank_score is not None
    assert results[0].provenance.source_name == "fastapi"
