"""Abstract embedding model port for generating semantic vector representations."""

from abc import ABC, abstractmethod
from collections.abc import Sequence


class BaseEmbeddingModel(ABC):
    """Abstract interface defining the embedding provider contract."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector dimensionality produced by this embedding model."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Compute embeddings for a sequence of texts."""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Compute embedding for a single search query."""
