"""Abstract vector store port for interchangeable vector database adapters."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from llmops.domain.models.document import DocumentChunk
from llmops.domain.models.retrieval import RetrievedChunk


class BaseVectorStore(ABC):
    """Abstract interface defining the vector store contract."""

    @abstractmethod
    def create_collection(
        self, collection_name: str, vector_size: int, distance: str = "Cosine"
    ) -> None:
        """Create a new collection if it does not already exist."""

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Return True if collection exists, False otherwise."""

    @abstractmethod
    def upsert_chunks(
        self,
        collection_name: str,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[list[float]],
    ) -> int:
        """Upsert document chunks and vectors into collection with metadata payload."""

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Perform semantic similarity search with optional metadata filtering."""

    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """Delete collection. Returns True if deleted."""
