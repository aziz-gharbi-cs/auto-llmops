"""S3 / MinIO implementation of the BaseStorage port using boto3."""

import io
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from llmops.domain.exceptions import StorageError
from llmops.domain.interfaces.storage import BaseStorage

logger = logging.getLogger(__name__)


class S3StorageAdapter(BaseStorage):
    """Storage adapter connecting to AWS S3 or self-hosted MinIO object storage."""

    def __init__(
        self,
        bucket_name: str,
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        region_name: str = "us-east-1",
        s3_client: Any | None = None,
    ) -> None:
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url
        self.region_name = region_name

        if s3_client is not None:
            self._client = s3_client
        else:
            import boto3

            self._client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name,
            )

    def upload_file(self, local_path: str | Path, remote_key: str) -> str:
        source = Path(local_path)
        if not source.exists():
            raise StorageError(f"Local source file does not exist: '{source}'")

        try:
            self._client.upload_file(str(source), self.bucket_name, remote_key)
            return f"s3://{self.bucket_name}/{remote_key}"
        except ClientError as err:
            raise StorageError(
                f"Failed to upload '{source}' to S3 '{self.bucket_name}/{remote_key}': {err}"
            ) from err

    def download_file(self, remote_key: str, local_path: str | Path) -> Path:
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._client.download_file(self.bucket_name, remote_key, str(target))
            return target
        except ClientError as err:
            raise StorageError(
                f"Failed to download from S3 '{self.bucket_name}/{remote_key}': {err}"
            ) from err

    def upload_bytes(
        self, data: bytes, remote_key: str, content_type: str = "application/octet-stream"
    ) -> str:
        try:
            self._client.put_object(
                Bucket=self.bucket_name,
                Key=remote_key,
                Body=data,
                ContentType=content_type,
            )
            return f"s3://{self.bucket_name}/{remote_key}"
        except ClientError as err:
            raise StorageError(
                f"Failed to upload bytes to S3 '{self.bucket_name}/{remote_key}': {err}"
            ) from err

    def download_bytes(self, remote_key: str) -> bytes:
        try:
            resp = self._client.get_object(Bucket=self.bucket_name, Key=remote_key)
            body: io.BytesIO = resp["Body"]
            return body.read()
        except ClientError as err:
            raise StorageError(
                f"Failed to download bytes from S3 '{self.bucket_name}/{remote_key}': {err}"
            ) from err

    def list_objects(self, prefix: str = "") -> Sequence[str]:
        keys: list[str] = []
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for item in page.get("Contents", []):
                    keys.append(item["Key"])
            return sorted(keys)
        except ClientError as err:
            raise StorageError(
                f"Failed to list S3 objects in '{self.bucket_name}' prefix '{prefix}': {err}"
            ) from err

    def exists(self, remote_key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket_name, Key=remote_key)
            return True
        except ClientError as err:
            error_code = err.response.get("Error", {}).get("Code")
            if error_code in {"404", "NoSuchKey"}:
                return False
            raise StorageError(
                f"Error checking existence for S3 '{self.bucket_name}/{remote_key}': {err}"
            ) from err

    def delete_object(self, remote_key: str) -> bool:
        try:
            self._client.delete_object(Bucket=self.bucket_name, Key=remote_key)
            return True
        except ClientError as err:
            logger.warning("Failed to delete S3 object '%s': %s", remote_key, err)
            return False
