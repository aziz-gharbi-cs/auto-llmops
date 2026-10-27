"""Local file-based experiment tracker implementation."""

import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llmops.domain.interfaces.experiment_tracker import BaseExperimentTracker
from llmops.domain.models.training import ModelManifest


class LocalExperimentTracker(BaseExperimentTracker):
    """File-based experiment tracker that records runs, metrics, and artifacts locally."""

    def __init__(self, base_dir: str | Path = "artifacts/experiments") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._active_run_id: str | None = None
        self._current_run_dir: Path | None = None
        self._run_data: dict[str, Any] = {}

    @property
    def active_run_id(self) -> str | None:
        return self._active_run_id

    def start_run(self, run_name: str, tags: dict[str, str] | None = None) -> str:
        """Start a new local experiment run."""
        self._active_run_id = f"run-{uuid.uuid4().hex[:12]}"
        self._current_run_dir = self.base_dir / self._active_run_id
        self._current_run_dir.mkdir(parents=True, exist_ok=True)
        (self._current_run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

        self._run_data = {
            "run_id": self._active_run_id,
            "run_name": run_name,
            "status": "RUNNING",
            "start_time": datetime.now(UTC).isoformat(),
            "end_time": None,
            "tags": tags or {},
            "params": {},
            "metrics": {},
            "metrics_history": [],
        }
        self._flush()
        return self._active_run_id

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._active_run_id or not self._run_data:
            raise RuntimeError("Cannot log params: no active run.")
        self._run_data["params"].update(params)
        self._flush()

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self._active_run_id or not self._run_data:
            raise RuntimeError("Cannot log metrics: no active run.")
        self._run_data["metrics"].update(metrics)
        record = {
            "step": step,
            "timestamp": datetime.now(UTC).isoformat(),
            "metrics": metrics,
        }
        self._run_data["metrics_history"].append(record)
        self._flush()

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        if not self._active_run_id or not self._current_run_dir:
            raise RuntimeError("Cannot log artifact: no active run.")
        src = Path(local_path)
        if not src.exists():
            return

        dest_dir = self._current_run_dir / "artifacts"
        if artifact_path:
            dest_dir = dest_dir / artifact_path
            dest_dir.mkdir(parents=True, exist_ok=True)

        if src.is_file():
            shutil.copy2(src, dest_dir / src.name)
        elif src.is_dir():
            target = dest_dir / src.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)

    def log_model_manifest(self, manifest: ModelManifest) -> None:
        if not self._active_run_id or not self._current_run_dir:
            raise RuntimeError("Cannot log manifest: no active run.")
        manifest_file = self._current_run_dir / "artifacts" / "model_manifest.json"
        manifest_file.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        self._run_data["model_manifest"] = manifest.model_dump(mode="json")
        self._flush()

    def end_run(self) -> None:
        if not self._active_run_id or not self._run_data:
            return
        self._run_data["status"] = "FINISHED"
        self._run_data["end_time"] = datetime.now(UTC).isoformat()
        self._flush()
        self._active_run_id = None
        self._current_run_dir = None

    def _flush(self) -> None:
        if self._current_run_dir and self._run_data:
            info_file = self._current_run_dir / "run_info.json"
            info_file.write_text(
                json.dumps(self._run_data, indent=2, default=str), encoding="utf-8"
            )
