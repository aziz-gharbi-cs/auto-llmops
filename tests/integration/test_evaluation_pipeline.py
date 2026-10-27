"""Integration tests for the complete evaluation and model promotion pipeline (Milestone 8)."""

import json
from pathlib import Path

import yaml

from llmops.application.use_cases.evaluate_model import EvaluateModelUseCase
from llmops.application.use_cases.promote_model import PromoteModelUseCase
from llmops.domain.models.training import ModelManifest
from llmops.infrastructure.llm.mock_llm import MockLLMClient


def test_evaluation_and_promotion_pipeline_end_to_end(tmp_path: Path) -> None:
    # 1. Setup test dataset
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    test_file = data_dir / "test.jsonl"
    eval_dir = tmp_path / "eval"
    models_dir = tmp_path / "models" / "adapter"
    models_dir.mkdir(parents=True)

    samples = [
        {
            "messages": [
                {"role": "system", "content": "Assistant"},
                {"role": "user", "content": f"How do I use FastAPI dependency {i}?"},
                {"role": "assistant", "content": f"Use Depends(func_{i}) in your route."},
            ]
        }
        for i in range(3)
    ]
    with test_file.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # 2. Setup unapproved model manifest
    manifest_yaml = models_dir / "model_manifest.yaml"
    initial_manifest = ModelManifest(
        model_version="gemma-3-lora-v1",
        git_commit="git-commit-hash",
        dvc_revision="dvc-v1",
        mlflow_run_id="run-test-001",
        base_model="google/gemma-3-1b-it",
        base_model_revision="main",
        adapter_version="lora-r16-a32",
        training_config_hash="confighash123",
        approved=False,
    )
    man_dict = initial_manifest.model_dump()
    man_dict["created_at"] = initial_manifest.created_at.isoformat()
    manifest_yaml.write_text(yaml.safe_dump(man_dict), encoding="utf-8")

    # 3. Execute EvaluateModelUseCase
    eval_use_case = EvaluateModelUseCase(
        candidate_client=MockLLMClient(
            model_name="gemma-3-lora-v1",
            default_response="Use Depends(func) in FastAPI route handlers.",
        ),
        base_client=MockLLMClient(
            model_name="gemma-3-base",
            default_response="Dependencies can be used with Depends in FastAPI.",
        ),
    )
    report = eval_use_case.execute(
        test_dataset_path=test_file,
        output_dir=eval_dir,
        model_version="gemma-3-lora-v1",
    )

    assert report.quality_gate_passed is True
    report_file = eval_dir / "evaluation_report.json"
    assert report_file.exists()

    # 4. Execute PromoteModelUseCase
    promote_use_case = PromoteModelUseCase()
    promoted_manifest = promote_use_case.execute(
        model_version="gemma-3-lora-v1",
        target_stage="production",
        manifest_path=manifest_yaml,
        evaluation_report_path=report_file,
    )

    assert promoted_manifest.approved is True
    assert promoted_manifest.evaluation_report is not None

    # 5. Verify persisted manifest on disk has approved=True and stage=production
    updated_yaml = yaml.safe_load(manifest_yaml.read_text(encoding="utf-8"))
    assert updated_yaml["approved"] is True
    assert updated_yaml["stage"] == "production"
