"""Smoke tests verifying package structure and basic domain contracts."""

import llmops
from llmops.domain.exceptions import (
    ConfigurationError,
    LLMOpsError,
    ModelInferenceError,
    QualityGateError,
    SourceAcquisitionError,
    StorageError,
    ValidationError,
    VectorStoreError,
)


def test_package_version() -> None:
    """Verify package version is defined and follows semver."""
    assert llmops.__version__ == "0.1.0"


def test_domain_exceptions_hierarchy() -> None:
    """Verify that domain exceptions correctly inherit from LLMOpsError."""
    exceptions = [
        ConfigurationError("test config error"),
        ValidationError("test validation error"),
        SourceAcquisitionError("test crawler error"),
        StorageError("test storage error"),
        VectorStoreError("test vector error"),
        ModelInferenceError("test model error"),
        QualityGateError("test quality error"),
    ]
    for exc in exceptions:
        assert isinstance(exc, LLMOpsError)
        assert exc.message.startswith("test")
        assert exc.details == {}


def test_domain_exception_with_details() -> None:
    """Verify domain exceptions preserve custom structured details."""
    details = {"source": "fastapi", "status_code": 404}
    err = SourceAcquisitionError("Failed to fetch", details=details)
    assert err.message == "Failed to fetch"
    assert err.details == details
