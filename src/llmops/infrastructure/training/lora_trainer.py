"""LoRA fine-tuning engine supporting CPU smoke profile and CUDA GPU training (Section 10/79)."""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from llmops.domain.exceptions import HardwareNotSupportedError, TrainingExecutionError
from llmops.domain.interfaces.experiment_tracker import BaseExperimentTracker
from llmops.domain.interfaces.trainer import BaseTrainer
from llmops.domain.models.training import TrainingConfig, TrainingResult

logger = logging.getLogger(__name__)


class LoRATrainer(BaseTrainer):
    """Fine-tunes Gemma 3 models using LoRA / PEFT with profile-driven hardware paths."""

    def __init__(self, tracker: BaseExperimentTracker | None = None) -> None:
        self.tracker = tracker

    def train(
        self,
        config: TrainingConfig,
        train_dataset_path: Path,
        val_dataset_path: Path | None = None,
    ) -> TrainingResult:
        """Execute training according to the configured profile (smoke or full)."""
        if not train_dataset_path.exists():
            raise TrainingExecutionError(
                f"Training dataset does not exist at '{train_dataset_path}'"
            )

        if config.profile == "smoke":
            return self._train_smoke(config, train_dataset_path, val_dataset_path)

        if config.profile == "full":
            return self._train_gpu_full(config, train_dataset_path, val_dataset_path)

        raise TrainingExecutionError(f"Unsupported training profile: '{config.profile}'")

    def _train_smoke(
        self,
        config: TrainingConfig,
        train_dataset_path: Path,
        val_dataset_path: Path | None = None,
    ) -> TrainingResult:
        """Lightweight, deterministic CPU smoke test simulating training loop and adapter export."""
        start_time = time.time()
        run_id = self.tracker.active_run_id if self.tracker else f"smoke-{uuid.uuid4().hex[:8]}"

        # Validate dataset format
        lines = [
            line.strip()
            for line in train_dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            raise TrainingExecutionError(f"Training dataset '{train_dataset_path}' is empty.")

        # Simulate training steps and decreasing loss curve
        num_steps = 5
        base_loss = 2.45
        loss_history: list[dict[str, float]] = []

        for step in range(1, num_steps + 1):
            simulated_train_loss = round(base_loss * (0.85**step), 4)
            simulated_val_loss = round(simulated_train_loss * 1.05, 4)
            record = {
                "step": float(step),
                "train_loss": simulated_train_loss,
                "val_loss": simulated_val_loss,
            }
            loss_history.append(record)

            if self.tracker:
                self.tracker.log_metrics(
                    {"train_loss": simulated_train_loss, "val_loss": simulated_val_loss},
                    step=step,
                )

        final_train_loss = loss_history[-1]["train_loss"]
        final_val_loss = loss_history[-1]["val_loss"]

        # Export mock PEFT LoRA adapter artifact
        adapter_dir = Path(config.output_dir) / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        peft_config = {
            "base_model_name_or_path": config.base_model,
            "bias": config.lora.bias,
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "lora_alpha": config.lora.lora_alpha,
            "lora_dropout": config.lora.lora_dropout,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": config.lora.r,
            "target_modules": config.lora.target_modules,
            "task_type": config.lora.task_type,
        }
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(peft_config, indent=2), encoding="utf-8"
        )
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"MOCK_LORA_WEIGHTS_SMOKE_PROFILE")
        (adapter_dir / "tokenizer_config.json").write_text(
            json.dumps({"model_max_length": config.hyperparameters.max_seq_length}, indent=2),
            encoding="utf-8",
        )
        (adapter_dir / "training_args.json").write_text(
            config.hyperparameters.model_dump_json(indent=2), encoding="utf-8"
        )

        duration = round(time.time() - start_time, 2)
        model_version = f"gemma-3-lora-{config.dataset_version}-smoke"

        hardware_info: dict[str, Any] = {
            "profile": "smoke",
            "hardware": "CPU",
            "accelerator": "none",
            "simulated": True,
            "steps": num_steps,
        }

        return TrainingResult(
            run_id=run_id or "smoke-run",
            model_version=model_version,
            loss_history=loss_history,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            adapter_dir=str(adapter_dir),
            duration_seconds=duration,
            hardware_info=hardware_info,
        )

    def _train_gpu_full(
        self,
        config: TrainingConfig,
        train_dataset_path: Path,
        val_dataset_path: Path | None = None,
    ) -> TrainingResult:
        """Production GPU training path utilizing CUDA and PyTorch/Unsloth (Section 10/79)."""
        # Hardware pre-flight check
        try:
            import torch

            if not torch.cuda.is_available():
                raise HardwareNotSupportedError(
                    "Profile 'full' requires a CUDA-capable GPU, but no CUDA device was detected. "
                    "Use '--profile smoke' on development laptops (Section 79)."
                )
        except ImportError as exc:
            raise HardwareNotSupportedError(
                "Profile 'full' requires PyTorch with CUDA acceleration. "
                "PyTorch is not installed or lacks CUDA support."
            ) from exc

        # Execute training with Transformers/PEFT
        logger.info(
            "Starting GPU training for '%s' with LoRA (r=%d, alpha=%d)",
            config.base_model,
            config.lora.r,
            config.lora.lora_alpha,
        )
        # Placeholder for full training loop on GPU clusters
        raise NotImplementedError(
            "GPU training cluster execution requires distributed worker environment."
        )
