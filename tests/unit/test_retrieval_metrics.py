"""Unit tests for information retrieval metrics calculations."""

from llmops.domain.services.retrieval_metrics import calculate_retrieval_metrics


def test_calculate_retrieval_metrics_perfect_hit() -> None:
    """Verify Recall@K, Precision@K, MRR, and nDCG when relevant item is top rank."""
    retrieved = ["doc_1", "doc_2", "doc_3"]
    ground_truth = {"doc_1"}

    metrics = calculate_retrieval_metrics(retrieved, ground_truth, k=3)
    assert metrics.recall_at_k == 1.0
    assert metrics.precision_at_k == round(1.0 / 3.0, 4)
    assert metrics.mrr == 1.0
    assert metrics.ndcg == 1.0


def test_calculate_retrieval_metrics_second_rank_hit() -> None:
    """Verify metrics when relevant item is at rank 2."""
    retrieved = ["doc_other", "doc_target", "doc_third"]
    ground_truth = {"doc_target"}

    metrics = calculate_retrieval_metrics(retrieved, ground_truth, k=3)
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 0.5  # 1 / 2


def test_calculate_retrieval_metrics_zero_hit() -> None:
    """Verify zero metrics when no retrieved items match ground truth."""
    retrieved = ["doc_x", "doc_y"]
    ground_truth = {"doc_target"}

    metrics = calculate_retrieval_metrics(retrieved, ground_truth, k=2)
    assert metrics.recall_at_k == 0.0
    assert metrics.precision_at_k == 0.0
    assert metrics.mrr == 0.0
    assert metrics.ndcg == 0.0
