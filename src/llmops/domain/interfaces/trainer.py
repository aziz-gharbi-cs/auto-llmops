"""Abstract port for model fine-tuning engines (Section 10)."""

from abc import ABC, abstractmethod
from pathlib import Path

from llmops.domain.models.training import TrainingConfig, TrainingResult


class BaseTrainer(ABC):
    """Abstract port for parameter-efficient fine-tuning engines."""

    @abstractmethod
    def train(
        self,
        config: TrainingConfig,
        train_dataset_path: Path,
        val_dataset_path: Path | None = None,
    ) -> TrainingResult:
        """Execute fine-tuning on the given dataset and return execution results and artifacts."""
        ...
