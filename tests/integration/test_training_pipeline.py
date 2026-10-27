"""Integration tests for the complete model training pipeline (Milestone 7)."""

import json
from pathlib import Path

import yaml

from llmops.application.use_cases.train_model import TrainModelUseCase
from llmops.domain.models.training import LoRAConfig, TrainingConfig
from llmops.infrastructure.training.local_tracker import LocalExperimentTracker
from llmops.infrastructure.training.lora_trainer import LoRATrainer


def test_training_pipeline_end_to_end(tmp_path: Path) -> None:
    # 1. Setup sample train and val datasets
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"

    train_data = [
        {
            "messages": [
                {"role": "system", "content": "Assistant"},
                {"role": "user", "content": f"How do I define schema {i} in FastAPI?"},
                {"role": "assistant", "content": f"Use Pydantic BaseModel for schema {i}."},
            ]
        }
        for i in range(5)
    ]
    with train_file.open("w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")

    val_data = [
        {
            "messages": [
                {"role": "system", "content": "Assistant"},
                {"role": "user", "content": "How do I define dependency in FastAPI?"},
                {"role": "assistant", "content": "Use Depends() in FastAPI path operations."},
            ]
        }
    ]
    with val_file.open("w", encoding="utf-8") as f:
        for entry in val_data:
            f.write(json.dumps(entry) + "\n")

    # 2. Setup tracker, trainer, and use case
    tracker = LocalExperimentTracker(base_dir=tmp_path / "experiments")
    trainer = LoRATrainer(tracker=tracker)
    use_case = TrainModelUseCase(trainer=trainer, tracker=tracker)

    config = TrainingConfig(
        base_model="google/gemma-3-1b-it",
        profile="smoke",
        lora=LoRAConfig(r=16, lora_alpha=32),
        output_dir=str(tmp_path / "models"),
    )

    # 3. Execute training
    result, manifest = use_case.execute(
        config=config,
        train_dataset_path=train_file,
        val_dataset_path=val_file,
    )

    # 4. Verify training result
    assert result.run_id is not None
    assert result.final_train_loss > 0.0
    assert result.final_val_loss is not None
    assert "gemma-3-lora" in result.model_version
    assert Path(result.adapter_dir).exists()

    # 5. Verify Section 23 ModelManifest contents
    assert manifest.model_version == result.model_version
    assert manifest.base_model == "google/gemma-3-1b-it"
    assert manifest.adapter_version == "lora-r16-a32"
    assert manifest.approved is False
    assert len(manifest.training_config_hash) == 64

    # 6. Verify manifest file exists on disk
    manifest_yaml = Path(result.adapter_dir) / "model_manifest.yaml"
    assert manifest_yaml.exists()
    yaml_content = yaml.safe_load(manifest_yaml.read_text(encoding="utf-8"))
    assert yaml_content["model_version"] == manifest.model_version
    assert yaml_content["base_model"] == manifest.base_model

    # 7. Verify experiment tracking artifacts
    run_dir = tmp_path / "experiments" / manifest.mlflow_run_id
    assert run_dir.exists()
    info_json = run_dir / "run_info.json"
    assert info_json.exists()
    info_data = json.loads(info_json.read_text(encoding="utf-8"))
    assert info_data["status"] == "FINISHED"
    assert info_data["params"]["base_model"] == "google/gemma-3-1b-it"
    assert (run_dir / "artifacts" / "adapter").exists()
