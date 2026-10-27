"""Unit tests for LoRATrainer supporting smoke profile and CUDA hardware detection."""

import json
from pathlib import Path

import pytest

from llmops.domain.exceptions import HardwareNotSupportedError, TrainingExecutionError
from llmops.domain.models.training import LoRAConfig, TrainingConfig
from llmops.infrastructure.training.local_tracker import LocalExperimentTracker
from llmops.infrastructure.training.lora_trainer import LoRATrainer


def _create_sample_dataset(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    sample = {
        "messages": [
            {"role": "system", "content": "Assistant"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
    }
    file_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")


def test_lora_trainer_smoke_profile(tmp_path: Path) -> None:
    train_file = tmp_path / "train.jsonl"
    _create_sample_dataset(train_file)

    tracker = LocalExperimentTracker(base_dir=tmp_path / "experiments")
    trainer = LoRATrainer(tracker=tracker)

    config = TrainingConfig(
        base_model="google/gemma-3-1b-it",
        profile="smoke",
        lora=LoRAConfig(r=8, lora_alpha=16),
        output_dir=str(tmp_path / "models"),
    )

    tracker.start_run("smoke-unit-test")
    result = trainer.train(config=config, train_dataset_path=train_file)
    tracker.end_run()

    assert result.final_train_loss > 0.0
    assert len(result.loss_history) > 0
    assert result.hardware_info["profile"] == "smoke"
    assert result.hardware_info["simulated"] is True

    # Verify adapter files
    adapter_path = Path(result.adapter_dir)
    assert (adapter_path / "adapter_config.json").exists()
    assert (adapter_path / "adapter_model.safetensors").exists()
    assert (adapter_path / "tokenizer_config.json").exists()

    adapter_config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert adapter_config["r"] == 8
    assert adapter_config["lora_alpha"] == 16
    assert adapter_config["peft_type"] == "LORA"


def test_lora_trainer_missing_dataset(tmp_path: Path) -> None:
    trainer = LoRATrainer()
    config = TrainingConfig()
    non_existent = tmp_path / "non_existent.jsonl"

    with pytest.raises(TrainingExecutionError):
        trainer.train(config=config, train_dataset_path=non_existent)


def test_lora_trainer_full_profile_hardware_detection(tmp_path: Path) -> None:
    """Verifies that running 'full' profile on non-CUDA machines raises exception."""
    train_file = tmp_path / "train.jsonl"
    _create_sample_dataset(train_file)

    trainer = LoRATrainer()
    config = TrainingConfig(profile="full")

    # In our local test environment (without CUDA GPU configured for torch),
    # it must raise HardwareNotSupportedError or NotImplementedError
    with pytest.raises((HardwareNotSupportedError, NotImplementedError)):
        trainer.train(config=config, train_dataset_path=train_file)
