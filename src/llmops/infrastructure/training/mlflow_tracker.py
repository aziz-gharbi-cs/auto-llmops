"""MLflow experiment tracker adapter (Section 21/22)."""

import logging
from pathlib import Path
from typing import Any

from llmops.domain.interfaces.experiment_tracker import BaseExperimentTracker
from llmops.domain.models.training import ModelManifest
from llmops.infrastructure.training.local_tracker import LocalExperimentTracker

logger = logging.getLogger(__name__)


class MLflowTracker(BaseExperimentTracker):
    """MLflow experiment tracker with seamless fallback to LocalExperimentTracker."""

    def __init__(
        self,
        tracking_uri: str = "sqlite:///artifacts/mlflow.db",
        experiment_name: str = "gemma-3-finetuning",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._active_run_id: str | None = None
        self._mlflow_available = False
        self._local_fallback: LocalExperimentTracker | None = None

        try:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow_available = True
            self._mlflow = mlflow
            logger.info(
                "MLflow tracker initialized (uri='%s', exp='%s')",
                self.tracking_uri,
                self.experiment_name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MLflow not available (%s). Using local file-based tracker fallback.",
                exc,
            )
            self._local_fallback = LocalExperimentTracker()

    @property
    def active_run_id(self) -> str | None:
        if self._mlflow_available:
            return self._active_run_id
        return self._local_fallback.active_run_id if self._local_fallback else None

    def start_run(self, run_name: str, tags: dict[str, str] | None = None) -> str:
        if self._mlflow_available:
            run = self._mlflow.start_run(run_name=run_name, tags=tags)
            self._active_run_id = str(run.info.run_id)
            return self._active_run_id
        assert self._local_fallback is not None
        return self._local_fallback.start_run(run_name=run_name, tags=tags)

    def log_params(self, params: dict[str, Any]) -> None:
        if self._mlflow_available:
            # Flatten or stringify complex params for MLflow
            flat_params: dict[str, Any] = {}
            for k, v in params.items():
                if isinstance(v, (str, int, float, bool)):
                    flat_params[k] = v
                else:
                    flat_params[k] = str(v)
            self._mlflow.log_params(flat_params)
            return
        assert self._local_fallback is not None
        self._local_fallback.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._mlflow_available:
            self._mlflow.log_metrics(metrics, step=step)
            return
        assert self._local_fallback is not None
        self._local_fallback.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        if self._mlflow_available:
            path_str = str(local_path)
            self._mlflow.log_artifact(path_str, artifact_path=artifact_path)
            return
        assert self._local_fallback is not None
        self._local_fallback.log_artifact(local_path, artifact_path=artifact_path)

    def log_model_manifest(self, manifest: ModelManifest) -> None:
        if self._mlflow_available:
            import tempfile

            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                f.write(manifest.model_dump_json(indent=2))
                tmp_path = f.name
            self._mlflow.log_artifact(tmp_path, artifact_path="model_manifest")
            return
        assert self._local_fallback is not None
        self._local_fallback.log_model_manifest(manifest)

    def end_run(self) -> None:
        if self._mlflow_available:
            self._mlflow.end_run()
            self._active_run_id = None
            return
        if self._local_fallback:
            self._local_fallback.end_run()
