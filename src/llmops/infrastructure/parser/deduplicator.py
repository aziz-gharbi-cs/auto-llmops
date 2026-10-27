"""Deduplication engine supporting exact hash and near-duplicate Jaccard matching."""

import re
from collections.abc import Sequence
from typing import TypeVar

from llmops.domain.models.document import Document, DocumentChunk

T = TypeVar("T", Document, DocumentChunk)


def compute_shingles(text: str, k: int = 3) -> set[str]:
    """Compute character k-shingles for near-duplicate Jaccard comparison."""
    cleaned = re.sub(r"\s+", " ", text.lower().strip())
    if len(cleaned) < k:
        return {cleaned}
    return {cleaned[i : i + k] for i in range(len(cleaned) - k + 1)}


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity coefficient between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return float(intersection) / float(union) if union > 0 else 0.0


class Deduplicator:
    """Detects and filters exact duplicates (SHA-256) and near-duplicates (Jaccard similarity)."""

    def __init__(self, near_duplicate_threshold: float = 0.85, shingle_size: int = 4) -> None:
        self.near_duplicate_threshold = near_duplicate_threshold
        self.shingle_size = shingle_size

    def deduplicate(
        self,
        items: Sequence[T],
        check_near_duplicates: bool = True,
    ) -> tuple[list[T], int, int]:
        """Filter exact and near duplicates.

        Returns:
            (unique_items, exact_duplicates_removed, near_duplicates_removed)
        """
        seen_hashes: set[str] = set()
        exact_filtered: list[T] = []
        exact_duplicates = 0

        # Phase 1: Exact duplicate filtering via content_hash
        for item in items:
            c_hash = item.content_hash
            if c_hash in seen_hashes:
                exact_duplicates += 1
                continue
            seen_hashes.add(c_hash)
            exact_filtered.append(item)

        if not check_near_duplicates or len(exact_filtered) <= 1:
            return exact_filtered, exact_duplicates, 0

        # Phase 2: Near-duplicate filtering via Jaccard shingling
        unique_items: list[T] = []
        shingle_index: list[set[str]] = []
        near_duplicates = 0

        for item in exact_filtered:
            content = item.content
            shingles = compute_shingles(content, k=self.shingle_size)

            is_near_dup = False
            for existing_shingles in shingle_index:
                sim = jaccard_similarity(shingles, existing_shingles)
                if sim >= self.near_duplicate_threshold:
                    is_near_dup = True
                    near_duplicates += 1
                    break

            if not is_near_dup:
                unique_items.append(item)
                shingle_index.append(shingles)

        return unique_items, exact_duplicates, near_duplicates
