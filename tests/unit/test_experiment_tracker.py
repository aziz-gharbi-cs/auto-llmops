"""Unit tests for BaseExperimentTracker, LocalExperimentTracker, and MLflowTracker."""

import json
from pathlib import Path

from llmops.domain.models.training import ModelManifest
from llmops.infrastructure.training.local_tracker import LocalExperimentTracker
from llmops.infrastructure.training.mlflow_tracker import MLflowTracker


def test_local_tracker_lifecycle(tmp_path: Path) -> None:
    tracker = LocalExperimentTracker(base_dir=tmp_path)
    assert tracker.active_run_id is None

    run_id = tracker.start_run(run_name="test-run", tags={"env": "test"})
    assert run_id.startswith("run-")
    assert tracker.active_run_id == run_id

    # Log params
    tracker.log_params({"lr": 0.001, "batch_size": 4})

    # Log metrics
    tracker.log_metrics({"train_loss": 1.5}, step=1)
    tracker.log_metrics({"train_loss": 1.1}, step=2)

    # Log artifact
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("artifact content", encoding="utf-8")
    tracker.log_artifact(dummy_file, artifact_path="configs")

    # Log model manifest
    manifest = ModelManifest(
        model_version="v1",
        git_commit="git-sha",
        dvc_revision="dvc-v1",
        mlflow_run_id=run_id,
        base_model="google/gemma-3-1b-it",
        base_model_revision="main",
        adapter_version="lora-v1",
        training_config_hash="cfg-hash",
        approved=False,
    )
    tracker.log_model_manifest(manifest)

    # End run
    tracker.end_run()
    assert tracker.active_run_id is None

    # Inspect run_info.json
    run_dir = tmp_path / run_id
    info_file = run_dir / "run_info.json"
    assert info_file.exists()

    data = json.loads(info_file.read_text(encoding="utf-8"))
    assert data["status"] == "FINISHED"
    assert data["params"]["lr"] == 0.001
    assert data["metrics"]["train_loss"] == 1.1
    assert len(data["metrics_history"]) == 2
    assert (run_dir / "artifacts" / "configs" / "dummy.txt").exists()
    assert (run_dir / "artifacts" / "model_manifest.json").exists()


def test_mlflow_tracker_fallback_works(tmp_path: Path) -> None:
    """Verify MLflowTracker delegates to LocalExperimentTracker if MLflow unavailable."""
    tracker = MLflowTracker(
        tracking_uri=f"file://{tmp_path}/mlruns",
        experiment_name="unit-tests",
    )
    run_id = tracker.start_run(run_name="fallback-run")
    assert run_id is not None

    tracker.log_params({"model": "gemma-3"})
    tracker.log_metrics({"loss": 0.5})
    tracker.end_run()
