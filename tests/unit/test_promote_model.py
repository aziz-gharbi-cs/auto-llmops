"""Unit tests for PromoteModelUseCase enforcing Quality Gate criteria (Section 22)."""

from pathlib import Path

import pytest
import yaml

from llmops.application.use_cases.promote_model import PromoteModelUseCase
from llmops.domain.exceptions import QualityGateError
from llmops.domain.models.evaluation import (
    EvaluationReport,
    LayerAResults,
    LayerBResults,
    LayerCResults,
    LayerDResults,
    LayerEResults,
)
from llmops.domain.models.training import ModelManifest


def _setup_artifacts(tmp_path: Path, quality_gate_passed: bool) -> tuple[Path, Path]:
    report_file = tmp_path / "evaluation_report.json"
    manifest_file = tmp_path / "model_manifest.yaml"

    report = EvaluationReport(
        evaluation_id="eval-123",
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
        quality_gate_passed=quality_gate_passed,
        rejection_reasons=[] if quality_gate_passed else ["MRR below threshold"],
        recommendation="APPROVE_FOR_STAGING" if quality_gate_passed else "REJECT",
    )
    report_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    manifest = ModelManifest(
        model_version="gemma-3-lora-v1",
        git_commit="sha123",
        dvc_revision="dvc-v1",
        mlflow_run_id="run-1",
        base_model="google/gemma-3-1b-it",
        base_model_revision="main",
        adapter_version="lora-v1",
        training_config_hash="cfghash",
        approved=False,
    )
    manifest_dict = manifest.model_dump()
    manifest_dict["created_at"] = manifest.created_at.isoformat()
    manifest_file.write_text(yaml.safe_dump(manifest_dict), encoding="utf-8")

    return report_file, manifest_file


def test_promote_model_success(tmp_path: Path) -> None:
    report_file, manifest_file = _setup_artifacts(tmp_path, quality_gate_passed=True)

    use_case = PromoteModelUseCase()
    promoted = use_case.execute(
        model_version="gemma-3-lora-v1",
        target_stage="staging",
        manifest_path=manifest_file,
        evaluation_report_path=report_file,
    )

    assert promoted.approved is True
    assert promoted.evaluation_report is not None

    # Check updated file on disk
    updated_dict = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    assert updated_dict["approved"] is True
    assert updated_dict["stage"] == "staging"


def test_promote_model_rejected_by_quality_gate(tmp_path: Path) -> None:
    report_file, manifest_file = _setup_artifacts(tmp_path, quality_gate_passed=False)

    use_case = PromoteModelUseCase()
    with pytest.raises(QualityGateError, match="Quality Gate rejected candidate"):
        use_case.execute(
            model_version="gemma-3-lora-v1",
            target_stage="production",
            manifest_path=manifest_file,
            evaluation_report_path=report_file,
        )


def test_promote_model_missing_report(tmp_path: Path) -> None:
    use_case = PromoteModelUseCase()
    with pytest.raises(QualityGateError, match="evaluation report not found"):
        use_case.execute(
            model_version="gemma-3-lora-v1",
            evaluation_report_path=tmp_path / "non_existent_report.json",
        )
