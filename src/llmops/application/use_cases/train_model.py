"""Use case orchestrating model fine-tuning, MLflow tracking, and manifest creation."""

import logging
import subprocess
from pathlib import Path
from typing import Any

import yaml

from llmops.domain.interfaces.experiment_tracker import BaseExperimentTracker
from llmops.domain.interfaces.trainer import BaseTrainer
from llmops.domain.models.training import (
    LoRAConfig,
    ModelManifest,
    TrainingConfig,
    TrainingHyperparameters,
    TrainingResult,
)
from llmops.infrastructure.training.local_tracker import LocalExperimentTracker
from llmops.infrastructure.training.lora_trainer import LoRATrainer

logger = logging.getLogger(__name__)


class TrainingExecutionResult(tuple[Any, ...]):
    """Tuple subclass returning (TrainingResult, ModelManifest) with convenience properties."""

    def __new__(cls, result: TrainingResult, manifest: ModelManifest) -> "TrainingExecutionResult":
        return super().__new__(cls, (result, manifest))

    def __init__(self, result: TrainingResult, manifest: ModelManifest) -> None:
        self.result = result
        self.manifest = manifest

    @property
    def success(self) -> bool:
        return self.result.success

    @property
    def training_loss(self) -> float:
        return self.result.final_train_loss

    @property
    def run_id(self) -> str:
        return self.result.run_id

    @property
    def model_version(self) -> str:
        return self.result.model_version

    @property
    def manifest_path(self) -> str | None:
        return self.result.manifest_path

    @property
    def adapter_dir(self) -> str:
        return self.result.adapter_dir


class TrainModelUseCase:
    """Orchestrates end-to-end model fine-tuning with experiment tracking and traceability."""

    def __init__(
        self,
        trainer: BaseTrainer | None = None,
        tracker: BaseExperimentTracker | None = None,
    ) -> None:
        self.tracker = tracker or LocalExperimentTracker()
        self.trainer = trainer or LoRATrainer(tracker=self.tracker)

    def load_config(
        self, config_path: str | Path, profile_override: str | None = None
    ) -> TrainingConfig:
        """Load and parse TrainingConfig from a YAML configuration file."""
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            logger.warning("Config file '%s' not found. Using default TrainingConfig.", cfg_path)
            config = TrainingConfig()
        else:
            raw_data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            lora_data = raw_data.get("lora", {})
            hp_data = raw_data.get("hyperparameters", {})

            # Apply profile specific overrides if defined in config
            profile = profile_override or raw_data.get("profile", "smoke")
            profile_cfg = raw_data.get("profiles", {}).get(profile, {})
            if "num_epochs" in profile_cfg:
                hp_data["num_epochs"] = profile_cfg["num_epochs"]
            if "batch_size" in profile_cfg:
                hp_data["batch_size"] = profile_cfg["batch_size"]
            if "max_seq_length" in profile_cfg:
                hp_data["max_seq_length"] = profile_cfg["max_seq_length"]

            config = TrainingConfig(
                base_model=raw_data.get("base_model", "google/gemma-3-1b-it"),
                base_model_revision=raw_data.get("base_model_revision", "main"),
                profile=profile,
                lora=LoRAConfig(**lora_data),
                hyperparameters=TrainingHyperparameters(**hp_data),
                dataset_version=raw_data.get("dataset_version", "v1"),
                output_dir=raw_data.get("output_dir", "artifacts/models/gemma-3-lora"),
            )

        if profile_override:
            config.profile = profile_override
        return config

    def _get_git_commit(self) -> str:
        """Retrieve current Git commit SHA."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return res.stdout.strip()
        except Exception:
            return "git-head-untracked"

    def execute(
        self,
        config: TrainingConfig | None = None,
        train_dataset_path: str | Path | None = None,
        val_dataset_path: str | Path | None = None,
        dataset_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        profile: str = "smoke",
    ) -> tuple[TrainingResult, ModelManifest]:
        """Execute training, track experiment with MLflow, and produce ModelManifest."""
        if config is None:
            config = TrainingConfig(profile=profile)
            if output_dir:
                config.output_dir = str(output_dir)

        if dataset_dir is not None:
            p_data = Path(dataset_dir)
            train_dataset_path = p_data / "train.jsonl"
            val_candidate = p_data / "val.jsonl"
            if val_candidate.exists():
                val_dataset_path = val_candidate

        if train_dataset_path is None:
            train_dataset_path = "data/processed/train.jsonl"
        train_path = Path(train_dataset_path)
        val_path = Path(val_dataset_path) if val_dataset_path else None

        # Step 1: Start tracking run
        run_name = f"gemma-3-{config.profile}-{config.dataset_version}"
        run_id = self.tracker.start_run(
            run_name=run_name,
            tags={
                "base_model": config.base_model,
                "profile": config.profile,
                "dataset_version": config.dataset_version,
            },
        )

        # Step 2: Log parameters
        params: dict[str, Any] = {
            "base_model": config.base_model,
            "base_model_revision": config.base_model_revision,
            "profile": config.profile,
            "lora_r": config.lora.r,
            "lora_alpha": config.lora.lora_alpha,
            "lora_dropout": config.lora.lora_dropout,
            "learning_rate": config.hyperparameters.learning_rate,
            "batch_size": config.hyperparameters.batch_size,
            "num_epochs": config.hyperparameters.num_epochs,
            "max_seq_length": config.hyperparameters.max_seq_length,
            "optimizer": config.hyperparameters.optimizer,
            "config_hash": config.compute_config_hash(),
        }
        self.tracker.log_params(params)

        try:
            # Step 3: Execute model fine-tuning
            result = self.trainer.train(
                config=config,
                train_dataset_path=train_path,
                val_dataset_path=val_path,
            )

            # Step 4: Construct Section 23 ModelManifest
            git_commit = self._get_git_commit()
            manifest = ModelManifest(
                model_version=result.model_version,
                git_commit=git_commit,
                dvc_revision=f"dvc-{config.dataset_version}",
                mlflow_run_id=run_id,
                base_model=config.base_model,
                base_model_revision=config.base_model_revision,
                adapter_version=f"lora-r{config.lora.r}-a{config.lora.lora_alpha}",
                training_config_hash=config.compute_config_hash(),
                approved=False,  # Awaiting Quality Gate (Section 22)
            )

            # Save manifest file
            adapter_dir = Path(result.adapter_dir)
            manifest_yaml_path = adapter_dir / "model_manifest.yaml"
            manifest_dict = manifest.model_dump()
            manifest_dict["created_at"] = manifest.created_at.isoformat()
            manifest_yaml_path.write_text(yaml.safe_dump(manifest_dict), encoding="utf-8")
            result.manifest_path = str(manifest_yaml_path)

            # Step 5: Log artifacts & manifest to tracker
            self.tracker.log_artifact(adapter_dir, artifact_path="adapter")
            self.tracker.log_model_manifest(manifest)

            return TrainingExecutionResult(result, manifest)

        finally:
            self.tracker.end_run()
