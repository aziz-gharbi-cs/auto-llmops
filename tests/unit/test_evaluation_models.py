"""Unit tests for multi-layer evaluation domain models conforming to Section 19/20."""

from llmops.domain.models.evaluation import (
    EvaluationReport,
    LayerAResults,
    LayerBResults,
    LayerCResults,
    LayerDResults,
    LayerEResults,
    QualityGateThresholds,
)


def test_layer_a_results_model() -> None:
    res = LayerAResults(
        total_records=50,
        valid_records=50,
        schema_errors=0,
        duplicate_count=0,
        leakage_detected=False,
        passed=True,
    )
    assert res.passed is True
    assert res.total_records == 50


def test_layer_b_results_model() -> None:
    res = LayerBResults(
        recall_at_k=0.92,
        precision_at_k=0.85,
        mrr=0.88,
        ndcg=0.91,
        context_relevance=4.6,
        passed=True,
    )
    assert res.mrr == 0.88
    assert res.passed is True


def test_layer_c_results_model() -> None:
    res = LayerCResults(
        correctness=4.5,
        relevance=4.8,
        groundedness=4.7,
        instruction_following=4.6,
        clarity=4.5,
        overall=4.62,
        passed=True,
    )
    assert res.groundedness == 4.7
    assert res.passed is True


def test_layer_d_results_model() -> None:
    res = LayerDResults(
        base_model_name="gemma-3-1b-base",
        candidate_model_name="gemma-3-1b-lora",
        base_overall_score=3.7,
        candidate_overall_score=4.4,
        delta=0.7,
        candidate_won=True,
        passed=True,
    )
    assert res.delta == 0.7
    assert res.candidate_won is True


def test_layer_e_results_model() -> None:
    res = LayerEResults(
        base_score=3.7,
        fine_tuned_score=4.4,
        rag_augmented_score=4.8,
        rag_grounding_rate=0.95,
        passed=True,
    )
    assert res.rag_augmented_score == 4.8
    assert res.rag_grounding_rate == 0.95


def test_quality_gate_thresholds_defaults() -> None:
    qg = QualityGateThresholds()
    assert qg.min_retrieval_mrr == 0.5
    assert qg.min_generation_groundedness == 3.5
    assert qg.min_candidate_win_margin == 0.0


def test_evaluation_report_serialization() -> None:
    report = EvaluationReport(
        evaluation_id="eval-001",
        model_version="gemma-3-lora-v1",
        dataset_version="v1",
        layer_a=LayerAResults(total_records=10, valid_records=10, passed=True),
        layer_b=LayerBResults(
            recall_at_k=0.9, precision_at_k=0.8, mrr=0.85, ndcg=0.88, context_relevance=4.5
        ),
        layer_c=LayerCResults(
            correctness=4.5,
            relevance=4.5,
            groundedness=4.5,
            instruction_following=4.5,
            clarity=4.5,
            overall=4.5,
        ),
        layer_d=LayerDResults(
            base_model_name="base",
            candidate_model_name="cand",
            base_overall_score=3.5,
            candidate_overall_score=4.5,
            delta=1.0,
            candidate_won=True,
            passed=True,
        ),
        layer_e=LayerEResults(
            base_score=3.5, fine_tuned_score=4.5, rag_augmented_score=4.8, rag_grounding_rate=0.9
        ),
        quality_gate_passed=True,
        recommendation="APPROVE_FOR_STAGING",
    )
    data = report.model_dump()
    assert data["evaluation_id"] == "eval-001"
    assert data["quality_gate_passed"] is True
    assert data["recommendation"] == "APPROVE_FOR_STAGING"
