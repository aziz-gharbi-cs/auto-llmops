"""Factory creating the appropriate BaseStorage implementation based on environment or config."""

import os
from pathlib import Path

from llmops.domain.interfaces.storage import BaseStorage
from llmops.infrastructure.storage.local_storage import LocalStorageAdapter
from llmops.infrastructure.storage.s3_storage import S3StorageAdapter


def create_storage_adapter(
    provider: str | None = None,
    base_dir: str | Path = "data/storage",
    bucket_name: str | None = None,
    endpoint_url: str | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
) -> BaseStorage:
    """Create and return a configured BaseStorage adapter."""
    selected_provider = (
        provider or os.getenv("STORAGE_PROVIDER") or os.getenv("STORAGE_BACKEND") or "local"
    ).lower()

    if selected_provider in {"s3", "minio"}:
        target_bucket = bucket_name or os.getenv("S3_BUCKET_NAME") or "llmops-artifacts"
        target_endpoint = endpoint_url or os.getenv("S3_ENDPOINT_URL")
        target_key = aws_access_key_id or os.getenv("S3_ACCESS_KEY_ID")
        target_secret = aws_secret_access_key or os.getenv("S3_SECRET_ACCESS_KEY")
        target_region = os.getenv("S3_REGION", "us-east-1")

        return S3StorageAdapter(
            bucket_name=target_bucket,
            endpoint_url=target_endpoint,
            aws_access_key_id=target_key,
            aws_secret_access_key=target_secret,
            region_name=target_region,
        )

    return LocalStorageAdapter(base_dir=base_dir)


create_storage = create_storage_adapter
