"""Domain-level exceptions for LLMOps platform."""


class LLMOpsError(Exception):
    """Base exception for all domain errors within LLMOps platform."""

    def __init__(self, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(LLMOpsError):
    """Raised when configuration is invalid, missing, or malformed."""


class ValidationError(LLMOpsError):
    """Raised when data validation fails against domain rules."""


class SourceAcquisitionError(LLMOpsError):
    """Raised when data acquisition or crawling encounters an unrecoverable failure."""


class StorageError(LLMOpsError):
    """Raised when an object storage or filesystem operation fails."""


class VectorStoreError(LLMOpsError):
    """Raised when vector indexing or retrieval operations fail."""


class ModelInferenceError(LLMOpsError):
    """Raised when model serving or inference fails."""


class QualityGateError(LLMOpsError):
    """Raised when a candidate model or dataset fails to pass quality thresholds."""


class HardwareNotSupportedError(LLMOpsError):
    """Raised when the execution profile is unsupported on current hardware (Section 79)."""


class TrainingExecutionError(LLMOpsError):
    """Raised when a model training or adapter export process fails."""
