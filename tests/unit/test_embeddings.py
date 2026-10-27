"""Unit tests for the embedding adapter."""

import math

from llmops.infrastructure.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
    generate_deterministic_vector,
)


def test_deterministic_vector_generation() -> None:
    """Verify deterministic vector generation has correct dimensionality and unit norm."""
    vec = generate_deterministic_vector("FastAPI web framework", dimensions=384)
    assert len(vec) == 384
    # Check L2 unit norm: sum of squares should be approx 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-5


def test_deterministic_vector_consistency() -> None:
    """Verify same text produces identical embeddings."""
    vec1 = generate_deterministic_vector("Pydantic models", dimensions=384)
    vec2 = generate_deterministic_vector("Pydantic models", dimensions=384)
    assert vec1 == vec2


def test_sentence_transformer_fallback() -> None:
    """Verify SentenceTransformerEmbeddings works with deterministic fallback."""
    embedder = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        use_fallback=True,
    )
    assert embedder.dimensions == 384

    texts = ["FastAPI route definition", "SQLAlchemy ORM session"]
    embeddings = embedder.embed_texts(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384

    query_emb = embedder.embed_query("route definition")
    assert len(query_emb) == 384
