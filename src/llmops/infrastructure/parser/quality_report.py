"""Dataset quality reporting and statistical summary generator."""

import statistics
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from llmops.domain.models.document import Document, DocumentChunk


class TokenStatistics(BaseModel):
    """Token count summary statistics."""

    min_tokens: int = 0
    max_tokens: int = 0
    mean_tokens: float = 0.0
    median_tokens: float = 0.0
    total_tokens: int = 0


class QualityReport(BaseModel):
    """Machine-readable dataset quality report as required by Section 57 of the specification."""

    dataset_version: str = "v1"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_documents: int = 0
    total_chunks: int = 0
    exact_duplicates_removed: int = 0
    near_duplicates_removed: int = 0
    rejected_records: int = 0
    source_distribution: dict[str, int] = Field(default_factory=dict)
    chunk_token_stats: TokenStatistics = Field(default_factory=TokenStatistics)
    metadata: dict[str, Any] = Field(default_factory=dict)


def generate_quality_report(
    documents: Sequence[Document],
    chunks: Sequence[DocumentChunk],
    exact_dups: int = 0,
    near_dups: int = 0,
    rejected: int = 0,
    dataset_version: str = "v1",
) -> QualityReport:
    """Calculate and return a QualityReport given documents and chunks."""
    # Source distribution
    source_counts = Counter(doc.source_name for doc in documents)

    # Token statistics on chunks
    token_counts = [chunk.token_count for chunk in chunks]
    if token_counts:
        token_stats = TokenStatistics(
            min_tokens=min(token_counts),
            max_tokens=max(token_counts),
            mean_tokens=round(statistics.mean(token_counts), 2),
            median_tokens=round(statistics.median(token_counts), 2),
            total_tokens=sum(token_counts),
        )
    else:
        token_stats = TokenStatistics()

    return QualityReport(
        dataset_version=dataset_version,
        total_documents=len(documents),
        total_chunks=len(chunks),
        exact_duplicates_removed=exact_dups,
        near_duplicates_removed=near_dups,
        rejected_records=rejected,
        source_distribution=dict(source_counts),
        chunk_token_stats=token_stats,
    )
