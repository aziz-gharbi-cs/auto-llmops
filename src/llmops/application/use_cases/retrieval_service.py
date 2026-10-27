"""Retrieval service orchestrating query embedding, top-N search, and reranking."""

import logging
import time
from collections.abc import Sequence

from llmops.domain.interfaces.embeddings import BaseEmbeddingModel
from llmops.domain.interfaces.reranker import BaseReranker
from llmops.domain.interfaces.vector_store import BaseVectorStore
from llmops.domain.models.document import DocumentChunk
from llmops.domain.models.retrieval import RetrievalQuery, RetrievedChunk
from llmops.infrastructure.observability.metrics import registry
from llmops.infrastructure.observability.tracer import tracer

logger = logging.getLogger(__name__)


class RetrievalService:
    """Orchestrates two-stage semantic retrieval: vector store top-N + cross-encoder top-K."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_model: BaseEmbeddingModel,
        reranker: BaseReranker,
        default_collection: str = "docs_fastapi_pydantic_sqlalchemy",
    ) -> None:
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.default_collection = default_collection

    def index_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        collection_name: str | None = None,
        batch_size: int = 64,
    ) -> int:
        """Embed and upsert document chunks into vector store."""
        col_name = collection_name or self.default_collection

        if not self.vector_store.collection_exists(col_name):
            self.vector_store.create_collection(
                collection_name=col_name,
                vector_size=self.embedding_model.dimensions,
            )

        total_upserted = 0
        chunks_list = list(chunks)

        for i in range(0, len(chunks_list), batch_size):
            batch = chunks_list[i : i + batch_size]
            texts = [chunk.content for chunk in batch]
            embeddings = self.embedding_model.embed_texts(texts)
            count = self.vector_store.upsert_chunks(
                collection_name=col_name,
                chunks=batch,
                embeddings=embeddings,
            )
            total_upserted += count

        logger.info("Successfully indexed %d chunks into '%s'", total_upserted, col_name)
        return total_upserted

    def retrieve(
        self,
        query: RetrievalQuery,
        collection_name: str | None = None,
    ) -> list[RetrievedChunk]:
        """Execute two-stage retrieval: top-N vector search followed by top-K reranking."""
        t_start = time.perf_counter()
        col_name = collection_name or self.default_collection

        # Step 1: Compute query embedding
        t_embed_start = time.perf_counter()
        with tracer.start_span("retrieval.embed", attributes={"query_len": len(query.query_text)}):
            query_vector = self.embedding_model.embed_query(query.query_text)
        embed_duration = time.perf_counter() - t_embed_start
        registry.histogram("embedding_latency").observe(embed_duration)

        # Step 2: Prepare metadata filter
        filter_dict = None
        if query.filter_criteria:
            filter_dict = {
                k: v for k, v in query.filter_criteria.model_dump().items() if v is not None
            }

        # Step 3: Top-N vector similarity retrieval
        with tracer.start_span(
            "retrieval.search",
            attributes={"collection": col_name, "top_n": query.top_n},
        ):
            candidates = self.vector_store.search(
                collection_name=col_name,
                query_vector=query_vector,
                limit=query.top_n,
                filter_criteria=filter_dict,
            )

        if not candidates:
            total_duration = time.perf_counter() - t_start
            registry.histogram("retrieval_latency").observe(total_duration)
            return []

        # Step 4: Second-stage reranking (top-K)
        with tracer.start_span(
            "retrieval.rerank",
            attributes={"candidates_count": len(candidates), "top_k": query.top_k},
        ):
            results = self.reranker.rerank(
                query=query.query_text,
                chunks=candidates,
                top_k=query.top_k,
            )

        total_duration = time.perf_counter() - t_start
        registry.histogram("retrieval_latency").observe(total_duration)
        return results
