"""In-memory vector store implementing BaseVectorStore using exact cosine similarity."""

import math
from collections.abc import Sequence
from typing import Any

from llmops.domain.exceptions import VectorStoreError
from llmops.domain.interfaces.vector_store import BaseVectorStore
from llmops.domain.models.document import DocumentChunk
from llmops.domain.models.retrieval import RetrievedChunk


def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    dot_product = sum(a * b for a, b in zip(v1, v2, strict=True))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)


class InMemoryVectorStore(BaseVectorStore):
    """Zero-dependency in-memory vector store for unit tests and local experimentation."""

    def __init__(self) -> None:
        self.collections: dict[str, dict[str, Any]] = {}

    def create_collection(
        self, collection_name: str, vector_size: int = 384, distance: str = "Cosine"
    ) -> None:
        if collection_name not in self.collections:
            self.collections[collection_name] = {
                "vector_size": vector_size,
                "distance": distance,
                "chunks": [],
                "embeddings": [],
            }

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[list[float]],
    ) -> int:
        if collection_name not in self.collections:
            raise VectorStoreError(f"Collection '{collection_name}' does not exist.")

        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunk count does not match embedding count.")

        col = self.collections[collection_name]
        col["chunks"].extend(chunks)
        col["embeddings"].extend(embeddings)
        return len(chunks)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if collection_name not in self.collections:
            raise VectorStoreError(f"Collection '{collection_name}' does not exist.")

        col = self.collections[collection_name]
        scored_hits: list[tuple[float, DocumentChunk]] = []

        for chunk, emb in zip(col["chunks"], col["embeddings"], strict=True):
            # Check filters
            if filter_criteria:
                match = True
                prov = chunk.provenance
                field_map = {
                    "source_name": prov.source_name,
                    "document_id": prov.document_id,
                    "dataset_version": prov.dataset_version,
                }
                for k, v in filter_criteria.items():
                    if v is not None and field_map.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            score = cosine_similarity(query_vector, emb)
            scored_hits.append((score, chunk))

        # Sort by score descending
        scored_hits.sort(key=lambda x: x[0], reverse=True)
        top_hits = scored_hits[:limit]

        retrieved: list[RetrievedChunk] = []
        for score, chunk in top_hits:
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    score=score,
                    section_heading=chunk.section_heading,
                    provenance=chunk.provenance,
                )
            )
        return retrieved

    def delete_collection(self, collection_name: str) -> bool:
        if collection_name in self.collections:
            del self.collections[collection_name]
            return True
        return False
