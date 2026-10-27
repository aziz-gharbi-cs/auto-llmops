"""Domain models for Gemma 3 LoRA fine-tuning, MLflow tracking, and model manifests."""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class LoRAConfig(BaseModel):
    """Low-Rank Adaptation (LoRA) parameter-efficient configuration."""

    r: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1, le=512)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: str = Field(default="none")
    task_type: str = Field(default="CAUSAL_LM")


class TrainingHyperparameters(BaseModel):
    """Optimization and schedule hyperparameters for training."""

    learning_rate: float = Field(default=2e-4, gt=0.0)
    batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    num_epochs: int = Field(default=3, ge=1)
    max_seq_length: int = Field(default=1024, ge=32)
    warmup_ratio: float = Field(default=0.05, ge=0.0, le=0.5)
    weight_decay: float = Field(default=0.01, ge=0.0)
    lr_scheduler_type: str = Field(default="cosine")
    optimizer: str = Field(default="adamw_8bit")
    seed: int = Field(default=42)


class TrainingConfig(BaseModel):
    """Complete specification for a model training job."""

    base_model: str = Field(default="google/gemma-3-1b-it")
    base_model_revision: str = Field(default="main")
    profile: str = Field(default="smoke")  # "smoke" or "full"
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    hyperparameters: TrainingHyperparameters = Field(default_factory=TrainingHyperparameters)
    dataset_version: str = Field(default="v1")
    output_dir: str = Field(default="artifacts/models/gemma-3-lora")

    def compute_config_hash(self) -> str:
        """Compute deterministic SHA-256 digest of configuration parameters."""
        raw_repr = json.dumps(
            {
                "base_model": self.base_model,
                "base_model_revision": self.base_model_revision,
                "profile": self.profile,
                "lora": self.lora.model_dump(),
                "hyperparameters": self.hyperparameters.model_dump(),
                "dataset_version": self.dataset_version,
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw_repr.encode("utf-8")).hexdigest()


class TrainingResult(BaseModel):
    """Execution metrics and artifacts produced by a training run."""

    run_id: str
    model_version: str
    loss_history: list[dict[str, float]] = Field(default_factory=list)
    final_train_loss: float
    final_val_loss: float | None = None
    adapter_dir: str
    duration_seconds: float
    hardware_info: dict[str, Any] = Field(default_factory=dict)
    manifest_path: str | None = None
    success: bool = True

    @property
    def training_loss(self) -> float:
        """Alias for final_train_loss."""
        return self.final_train_loss


class ModelManifest(BaseModel):
    """Traceability manifest conforming to Section 23 of the Master Build Specification."""

    model_version: str
    git_commit: str
    dvc_revision: str
    mlflow_run_id: str
    base_model: str
    base_model_revision: str
    adapter_version: str
    training_config_hash: str
    evaluation_report: dict[str, Any] | None = None
    approved: bool = False
    stage: str = "candidate"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def promoted(self) -> bool:
        """Indicates if model has been promoted through quality gate."""
        return self.approved

    @property
    def manifest(self) -> "ModelManifest":
        """Self reference for promotion result compatibility."""
        return self
