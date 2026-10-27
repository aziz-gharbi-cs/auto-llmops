"""Unit tests for EvaluateModelUseCase and quality gate decisions."""

import json
from pathlib import Path

from llmops.application.use_cases.evaluate_model import EvaluateModelUseCase
from llmops.domain.models.evaluation import QualityGateThresholds
from llmops.infrastructure.curation.judge import DataCurationJudge
from llmops.infrastructure.llm.mock_llm import MockLLMClient


def _make_test_dataset(path: Path, num_records: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(num_records):
            entry = {
                "messages": [
                    {"role": "system", "content": "Assistant"},
                    {"role": "user", "content": f"How do I use FastAPI dependency {i}?"},
                    {"role": "assistant", "content": f"Use Depends(func_{i}) in your route."},
                ]
            }
            f.write(json.dumps(entry) + "\n")


def test_evaluate_model_quality_gate_passed(tmp_path: Path) -> None:
    test_file = tmp_path / "test.jsonl"
    _make_test_dataset(test_file)
    out_dir = tmp_path / "eval_out"

    candidate_client = MockLLMClient(
        model_name="gemma-3-lora-v1",
        default_response="Use Depends(func) in FastAPI route handlers.",
    )
    base_client = MockLLMClient(
        model_name="gemma-3-base",
        default_response="Dependencies can be used with Depends in FastAPI.",
    )
    judge_client = MockLLMClient(
        model_name="eval-judge",
        structured_payload={
            "relevance": 5,
            "correctness": 5,
            "groundedness": 5,
            "instruction_following": 5,
            "clarity": 5,
            "overall": 5.0,
            "pass": True,
            "reason": "Exemplary answer",
        },
    )
    judge = DataCurationJudge(llm_client=judge_client)

    eval_use_case = EvaluateModelUseCase(
        candidate_client=candidate_client,
        base_client=base_client,
        judge=judge,
        thresholds=QualityGateThresholds(
            min_retrieval_mrr=0.5,
            min_generation_groundedness=3.5,
            min_candidate_win_margin=0.0,
        ),
    )

    report = eval_use_case.execute(
        test_dataset_path=test_file,
        output_dir=out_dir,
        model_version="gemma-3-lora-v1",
    )

    assert report.quality_gate_passed is True
    assert report.recommendation == "APPROVE_FOR_STAGING"
    assert len(report.rejection_reasons) == 0
    assert report.layer_a.passed is True
    assert report.layer_d.candidate_won is True
    assert (out_dir / "evaluation_report.json").exists()


def test_evaluate_model_quality_gate_failed_thresholds(tmp_path: Path) -> None:
    """Verify Quality Gate fails when candidate metrics fail strict thresholds."""
    test_file = tmp_path / "test.jsonl"
    _make_test_dataset(test_file)

    candidate_client = MockLLMClient(model_name="cand")
    eval_use_case = EvaluateModelUseCase(
        candidate_client=candidate_client,
        thresholds=QualityGateThresholds(
            min_generation_groundedness=4.95,  # Unattainably high threshold triggers gate failure
        ),
    )

    report = eval_use_case.execute(
        test_dataset_path=test_file,
        output_dir=tmp_path / "eval_fail",
    )

    assert report.quality_gate_passed is False
    assert report.recommendation == "REJECT"
    assert len(report.rejection_reasons) > 0
    assert any("groundedness" in r for r in report.rejection_reasons)
