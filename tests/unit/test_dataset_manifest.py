"""Unit tests for dataset version manifests."""

import json
from datetime import UTC, datetime

from llmops.domain.models.dataset_manifest import DatasetFileRecord, DatasetVersionManifest


def test_dataset_version_manifest_schema() -> None:
    """Verify DatasetVersionManifest serialization and validation."""
    file_record = DatasetFileRecord(
        filename="chunks.jsonl",
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        byte_size=2048,
    )

    manifest = DatasetVersionManifest(
        dataset_version="v1.0.0",
        git_commit="a1b2c3d4",
        dvc_revision="rev-001",
        content_hash="c0ffee1234",
        storage_uri="s3://llmops-artifacts/datasets/v1.0.0/",
        created_at=datetime.now(UTC),
        num_documents=10,
        num_chunks=45,
        files=[file_record],
        quality_report_uri="s3://llmops-artifacts/datasets/v1.0.0/quality_report.json",
        metadata={"source": "fastapi_pydantic_sqlalchemy"},
    )

    raw_json = manifest.model_dump_json()
    data = json.loads(raw_json)

    assert data["dataset_version"] == "v1.0.0"
    assert data["git_commit"] == "a1b2c3d4"
    assert data["num_documents"] == 10
    assert data["num_chunks"] == 45
    assert len(data["files"]) == 1
    assert data["files"][0]["filename"] == "chunks.jsonl"

    reloaded = DatasetVersionManifest.model_validate_json(raw_json)
    assert reloaded.dataset_version == "v1.0.0"
    assert reloaded.files[0].sha256_hash == file_record.sha256_hash
