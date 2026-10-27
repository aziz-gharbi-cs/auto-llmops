"""Cross-encoder reranker adapter implementing BaseReranker."""

import logging
import re
from collections.abc import Sequence

from llmops.domain.interfaces.reranker import BaseReranker
from llmops.domain.models.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


def compute_lexical_overlap_score(query: str, text: str) -> float:
    """Compute lexical term overlap score between query and text as a lightweight heuristic."""
    query_tokens = set(re.findall(r"\w+", query.lower()))
    text_tokens = set(re.findall(r"\w+", text.lower()))
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(text_tokens))
    return float(overlap) / float(len(query_tokens))


class CrossEncoderReranker(BaseReranker):
    """Reranker adapter scoring candidate chunks with second-stage cross-encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self._model = None

        if not use_fallback:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(model_name)
                logger.info("Loaded CrossEncoder reranker model: %s", model_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CrossEncoder could not be loaded (%s). Using lexical overlap reranker.",
                    exc,
                )

    def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """Rerank candidates based on cross-encoder or lexical relevance."""
        if not chunks:
            return []

        chunks_list = list(chunks)

        if self._model is not None:
            pairs = [[query, chunk.content] for chunk in chunks_list]
            scores = self._model.predict(pairs)
            for chunk, score in zip(chunks_list, scores, strict=True):
                chunk.rerank_score = float(score)
        else:
            for chunk in chunks_list:
                # Combine initial vector score with lexical overlap
                lex_score = compute_lexical_overlap_score(query, chunk.content)
                chunk.rerank_score = round(0.5 * chunk.score + 0.5 * lex_score, 4)

        # Sort by rerank_score descending
        chunks_list.sort(
            key=lambda c: c.rerank_score if c.rerank_score is not None else 0.0,
            reverse=True,
        )
        return chunks_list[:top_k]
