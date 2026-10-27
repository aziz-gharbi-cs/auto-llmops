"""Abstract reranker port for second-stage cross-encoder context reranking."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from llmops.domain.models.retrieval import RetrievedChunk


class BaseReranker(ABC):
    """Abstract interface defining the second-stage reranker contract."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """Rerank candidates based on cross-attention / relevance scores. Returns top_k."""
