"""SentenceTransformer embedding adapter with deterministic fallback support."""

import hashlib
import logging
import math
from collections.abc import Sequence

from llmops.domain.interfaces.embeddings import BaseEmbeddingModel

logger = logging.getLogger(__name__)


def generate_deterministic_vector(text: str, dimensions: int = 384) -> list[float]:
    """Generate a deterministic unit-norm float vector from text hash."""
    vector = []
    base_hash = hashlib.sha256(text.encode("utf-8")).digest()

    for i in range(dimensions):
        # Generate pseudorandom float in range [-1.0, 1.0]
        byte_val = base_hash[(i * 7) % len(base_hash)]
        offset = (i * 13) % 255
        val = ((byte_val ^ offset) / 127.5) - 1.0
        vector.append(val)

    # Normalize vector to unit length (L2 norm)
    norm = math.sqrt(sum(x * x for x in vector))
    if norm > 0:
        vector = [x / norm for x in vector]

    return vector


class SentenceTransformerEmbeddings(BaseEmbeddingModel):
    """Embedding adapter implementing BaseEmbeddingModel."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimensions: int = 384,
        normalize: bool = True,
        use_fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self._dimensions = dimensions
        self.normalize = normalize
        self._model = None

        if not use_fallback:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name)
                logger.info("Loaded SentenceTransformer model: %s", model_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SentenceTransformer could not be loaded (%s). Using fallback hash embedding.",
                    exc,
                )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        if self._model is not None:
            raw_embeddings = self._model.encode(
                list(texts),
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            return [list(map(float, emb)) for emb in raw_embeddings]

        return [generate_deterministic_vector(t, self._dimensions) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]
