"""Unit tests for training domain models and Section 23 ModelManifest."""

from llmops.domain.models.training import (
    LoRAConfig,
    ModelManifest,
    TrainingConfig,
    TrainingHyperparameters,
    TrainingResult,
)


def test_lora_config_defaults() -> None:
    lora = LoRAConfig()
    assert lora.r == 16
    assert lora.lora_alpha == 32
    assert "q_proj" in lora.target_modules
    assert lora.task_type == "CAUSAL_LM"


def test_training_hyperparameters_defaults() -> None:
    hp = TrainingHyperparameters()
    assert hp.learning_rate == 2e-4
    assert hp.batch_size == 2
    assert hp.num_epochs == 3
    assert hp.optimizer == "adamw_8bit"


def test_training_config_hash_determinism() -> None:
    cfg1 = TrainingConfig(
        base_model="google/gemma-3-1b-it",
        profile="smoke",
        lora=LoRAConfig(r=16, lora_alpha=32),
    )
    cfg2 = TrainingConfig(
        base_model="google/gemma-3-1b-it",
        profile="smoke",
        lora=LoRAConfig(r=16, lora_alpha=32),
    )
    cfg3 = TrainingConfig(
        base_model="google/gemma-3-1b-it",
        profile="smoke",
        lora=LoRAConfig(r=32, lora_alpha=64),  # Changed LoRA rank
    )

    h1 = cfg1.compute_config_hash()
    h2 = cfg2.compute_config_hash()
    h3 = cfg3.compute_config_hash()

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # SHA-256


def test_model_manifest_traceability() -> None:
    """Verify Section 23 fields are all present in ModelManifest."""
    manifest = ModelManifest(
        model_version="gemma-3-lora-v1",
        git_commit="abcdef1234567890",
        dvc_revision="dvc-v1",
        mlflow_run_id="run-mlflow-001",
        base_model="google/gemma-3-1b-it",
        base_model_revision="main",
        adapter_version="lora-r16-a32",
        training_config_hash="sha256hash123",
        approved=False,
    )

    data = manifest.model_dump()
    assert data["model_version"] == "gemma-3-lora-v1"
    assert data["git_commit"] == "abcdef1234567890"
    assert data["dvc_revision"] == "dvc-v1"
    assert data["mlflow_run_id"] == "run-mlflow-001"
    assert data["base_model"] == "google/gemma-3-1b-it"
    assert data["approved"] is False
    assert "created_at" in data


def test_training_result_model() -> None:
    res = TrainingResult(
        run_id="run-1",
        model_version="v1",
        loss_history=[{"step": 1.0, "train_loss": 2.0}],
        final_train_loss=1.2,
        final_val_loss=1.3,
        adapter_dir="artifacts/adapter",
        duration_seconds=12.5,
        hardware_info={"profile": "smoke"},
    )
    assert res.run_id == "run-1"
    assert res.final_train_loss == 1.2
