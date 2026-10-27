"""Abstract storage port for interchangeable object storage adapters."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path


class BaseStorage(ABC):
    """Abstract interface defining the object storage contract."""

    @abstractmethod
    def upload_file(self, local_path: str | Path, remote_key: str) -> str:
        """Upload a local file to remote storage key. Returns URI."""

    @abstractmethod
    def download_file(self, remote_key: str, local_path: str | Path) -> Path:
        """Download remote storage key to a local destination file path."""

    @abstractmethod
    def upload_bytes(
        self, data: bytes, remote_key: str, content_type: str = "application/octet-stream"
    ) -> str:
        """Upload raw bytes to remote storage key. Returns URI."""

    @abstractmethod
    def download_bytes(self, remote_key: str) -> bytes:
        """Download raw bytes from remote storage key."""

    @abstractmethod
    def list_objects(self, prefix: str = "") -> Sequence[str]:
        """List object keys under the specified prefix."""

    @abstractmethod
    def exists(self, remote_key: str) -> bool:
        """Check if an object exists at the specified key."""

    @abstractmethod
    def delete_object(self, remote_key: str) -> bool:
        """Delete an object at the specified key. Returns True if deleted."""
