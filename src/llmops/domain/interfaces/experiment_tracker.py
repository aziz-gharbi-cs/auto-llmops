"""Abstract port for experiment tracking and model registry operations (Section 21/22)."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from llmops.domain.models.training import ModelManifest


class BaseExperimentTracker(ABC):
    """Abstract port for experiment tracking (MLflow, local, or cloud)."""

    @property
    @abstractmethod
    def active_run_id(self) -> str | None:
        """Return the active run ID if a run is currently open."""
        ...

    @abstractmethod
    def start_run(self, run_name: str, tags: dict[str, str] | None = None) -> str:
        """Begin a new experiment run and return the assigned run ID."""
        ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log training hyperparameters and configuration parameters."""
        ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log scalar training or evaluation metrics at an optional step/epoch."""
        ...

    @abstractmethod
    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log a local file or directory as a run artifact."""
        ...

    @abstractmethod
    def log_model_manifest(self, manifest: ModelManifest) -> None:
        """Log the complete traceability model manifest as a run artifact."""
        ...

    @abstractmethod
    def end_run(self) -> None:
        """Conclude the active experiment run."""
        ...
