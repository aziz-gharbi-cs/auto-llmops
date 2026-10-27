"""Local filesystem implementation of the BaseStorage port."""

from collections.abc import Sequence
from pathlib import Path

from llmops.domain.exceptions import StorageError
from llmops.domain.interfaces.storage import BaseStorage


class LocalStorageAdapter(BaseStorage):
    """Storage adapter using the local filesystem as object storage."""

    def __init__(self, base_dir: str | Path = "data/storage") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, remote_key: str) -> Path:
        """Resolve a remote key safely within the base directory."""
        clean_key = remote_key.lstrip("/\\")
        target_path = (self.base_dir / clean_key).resolve()
        # Security guard: prevent path traversal attacks
        if not str(target_path).startswith(str(self.base_dir)):
            raise StorageError(f"Path traversal detected for key '{remote_key}'")
        return target_path

    def upload_file(self, local_path: str | Path, remote_key: str) -> str:
        source = Path(local_path)
        if not source.exists():
            raise StorageError(f"Local source file does not exist: '{source}'")

        target = self._resolve_path(remote_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        return f"file://{target}"

    def download_file(self, remote_key: str, local_path: str | Path) -> Path:
        source = self._resolve_path(remote_key)
        if not source.exists():
            raise StorageError(f"Remote object not found: '{remote_key}'")

        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        return destination

    def upload_bytes(
        self, data: bytes, remote_key: str, content_type: str = "application/octet-stream"
    ) -> str:
        target = self._resolve_path(remote_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return f"file://{target}"

    def download_bytes(self, remote_key: str) -> bytes:
        target = self._resolve_path(remote_key)
        if not target.exists():
            raise StorageError(f"Remote object not found: '{remote_key}'")
        return target.read_bytes()

    def list_objects(self, prefix: str = "") -> Sequence[str]:
        target_dir = self._resolve_path(prefix) if prefix else self.base_dir
        if not target_dir.exists():
            return []

        keys: list[str] = []
        if target_dir.is_dir():
            for p in target_dir.rglob("*"):
                if p.is_file():
                    rel_path = p.relative_to(self.base_dir)
                    keys.append(str(rel_path).replace("\\", "/"))
        elif target_dir.is_file():
            rel_path = target_dir.relative_to(self.base_dir)
            keys.append(str(rel_path).replace("\\", "/"))
        return sorted(keys)

    def exists(self, remote_key: str) -> bool:
        target = self._resolve_path(remote_key)
        return target.exists() and target.is_file()

    def delete_object(self, remote_key: str) -> bool:
        target = self._resolve_path(remote_key)
        if target.exists() and target.is_file():
            target.unlink()
            return True
        return False
