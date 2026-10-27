"""Unit tests for storage adapters and storage factory."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llmops.domain.exceptions import StorageError
from llmops.infrastructure.storage.factory import create_storage_adapter
from llmops.infrastructure.storage.local_storage import LocalStorageAdapter
from llmops.infrastructure.storage.s3_storage import S3StorageAdapter


def test_local_storage_crud(tmp_path: Path) -> None:
    """Verify LocalStorageAdapter upload, download, list, exists, and delete."""
    storage = LocalStorageAdapter(base_dir=tmp_path)

    # 1. upload_bytes and download_bytes
    data = b"Hello Local Storage"
    uri = storage.upload_bytes(data, "test/data.bin")
    assert uri.startswith("file://")
    assert storage.exists("test/data.bin") is True
    assert storage.download_bytes("test/data.bin") == data

    # 2. upload_file and download_file
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("sample content", encoding="utf-8")
    storage.upload_file(sample_file, "nested/sample.txt")
    assert storage.exists("nested/sample.txt") is True

    downloaded_dest = tmp_path / "downloaded.txt"
    storage.download_file("nested/sample.txt", downloaded_dest)
    assert downloaded_dest.read_text(encoding="utf-8") == "sample content"

    # 3. list_objects
    objects = storage.list_objects()
    assert "nested/sample.txt" in objects
    assert "test/data.bin" in objects

    # 4. delete_object
    deleted = storage.delete_object("test/data.bin")
    assert deleted is True
    assert storage.exists("test/data.bin") is False


def test_local_storage_prevents_path_traversal(tmp_path: Path) -> None:
    """Verify path traversal attempts raise StorageError."""
    storage = LocalStorageAdapter(base_dir=tmp_path / "safe")
    with pytest.raises(StorageError, match="Path traversal detected"):
        storage.upload_bytes(b"exploit", "../../etc/passwd")


def test_s3_storage_with_mock_client(tmp_path: Path) -> None:
    """Verify S3StorageAdapter operations using a mocked boto3 client."""
    mock_boto = MagicMock()
    mock_boto.get_object.return_value = {"Body": MagicMock(read=lambda: b"mocked s3 data")}
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {"Contents": [{"Key": "data/chunk_1.json"}, {"Key": "data/chunk_2.json"}]}
    ]
    mock_boto.get_paginator.return_value = mock_paginator

    s3 = S3StorageAdapter(
        bucket_name="test-bucket",
        endpoint_url="http://localhost:9000",
        s3_client=mock_boto,
    )

    # upload_bytes
    uri = s3.upload_bytes(b"hello", "test/hello.txt")
    assert uri == "s3://test-bucket/test/hello.txt"
    mock_boto.put_object.assert_called_once_with(
        Bucket="test-bucket",
        Key="test/hello.txt",
        Body=b"hello",
        ContentType="application/octet-stream",
    )

    # download_bytes
    data = s3.download_bytes("test/hello.txt")
    assert data == b"mocked s3 data"

    # list_objects
    items = s3.list_objects(prefix="data/")
    assert items == ["data/chunk_1.json", "data/chunk_2.json"]

    # exists
    assert s3.exists("test/hello.txt") is True

    # delete
    assert s3.delete_object("test/hello.txt") is True
    mock_boto.delete_object.assert_called_once_with(
        Bucket="test-bucket",
        Key="test/hello.txt",
    )


def test_storage_factory(tmp_path: Path) -> None:
    """Verify factory returns appropriate adapter for local and s3."""
    local_adapter = create_storage_adapter(provider="local", base_dir=tmp_path)
    assert isinstance(local_adapter, LocalStorageAdapter)

    s3_adapter = create_storage_adapter(
        provider="s3",
        bucket_name="my-bucket",
        endpoint_url="http://localhost:9000",
    )
    assert isinstance(s3_adapter, S3StorageAdapter)
