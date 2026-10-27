"""Contract test for BaseStorage implementations."""

from pathlib import Path

import pytest

from llmops.domain.interfaces.storage import BaseStorage
from llmops.infrastructure.storage.local_storage import LocalStorageAdapter


@pytest.fixture
def local_storage(tmp_path: Path) -> BaseStorage:
    """Fixture providing a LocalStorageAdapter instance."""
    return LocalStorageAdapter(base_dir=tmp_path / "contract_storage")


def verify_storage_contract(storage: BaseStorage, tmp_path: Path) -> None:
    """Generic contract verification asserting expected BaseStorage behavior."""
    key = "contract/test.txt"
    content = b"Storage contract verification content."

    # 1. Initially object should not exist
    assert storage.exists(key) is False

    # 2. Upload bytes
    uri = storage.upload_bytes(content, key)
    assert uri is not None
    assert storage.exists(key) is True

    # 3. Download bytes
    downloaded = storage.download_bytes(key)
    assert downloaded == content

    # 4. List objects
    keys = storage.list_objects(prefix="contract")
    assert key in keys

    # 5. File upload & download
    local_src = tmp_path / "source.txt"
    local_src.write_bytes(b"File payload")
    file_key = "contract/file.txt"
    storage.upload_file(local_src, file_key)
    assert storage.exists(file_key) is True

    local_dst = tmp_path / "destination.txt"
    storage.download_file(file_key, local_dst)
    assert local_dst.read_bytes() == b"File payload"

    # 6. Delete
    assert storage.delete_object(key) is True
    assert storage.exists(key) is False


def test_local_storage_fulfills_contract(local_storage: BaseStorage, tmp_path: Path) -> None:
    """Verify LocalStorageAdapter satisfies the BaseStorage contract."""
    verify_storage_contract(local_storage, tmp_path)
