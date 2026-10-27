"""Retrieval evaluation metrics: Recall@K, Precision@K, MRR, and nDCG."""

import math
from collections.abc import Sequence

from llmops.domain.models.retrieval import RetrievalMetrics


def calculate_retrieval_metrics(
    retrieved_chunk_ids: Sequence[str],
    ground_truth_chunk_ids: set[str],
    k: int = 3,
) -> RetrievalMetrics:
    """Calculate Recall@K, Precision@K, MRR, and nDCG for a single query."""
    top_k = list(retrieved_chunk_ids)[:k]
    if not top_k or not ground_truth_chunk_ids:
        return RetrievalMetrics(k=k, recall_at_k=0.0, precision_at_k=0.0, mrr=0.0, ndcg=0.0)

    # 1. Hits in top K
    hits = [1 if cid in ground_truth_chunk_ids else 0 for cid in top_k]
    num_hits = sum(hits)

    # 2. Recall@K & Precision@K
    recall = float(num_hits) / float(len(ground_truth_chunk_ids))
    precision = float(num_hits) / float(len(top_k))

    # 3. MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit == 1:
            mrr = 1.0 / float(rank)
            break

    # 4. nDCG@K
    dcg = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit == 1:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG
    ideal_hits = min(len(ground_truth_chunk_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = (dcg / idcg) if idcg > 0.0 else 0.0

    return RetrievalMetrics(
        k=k,
        recall_at_k=round(recall, 4),
        precision_at_k=round(precision, 4),
        mrr=round(mrr, 4),
        ndcg=round(ndcg, 4),
    )
