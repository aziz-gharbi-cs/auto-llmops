"""Unit tests for crawler safety, SSRF protection, and URL normalization."""

import pytest

from llmops.domain.exceptions import ValidationError
from llmops.infrastructure.crawler.safety import (
    SSRFValidationError,
    is_ip_private_or_restricted,
    normalize_url,
    validate_url_safety,
)


def test_normalize_url_strips_fragments_and_tracking_params() -> None:
    """Verify normalize_url strips fragments, utm params, and lowercases host."""
    raw = "HTTPS://FastAPI.tiangolo.COM/tutorial/first-steps/?utm_source=twitter&ref=abc&id=123#summary"
    normalized = normalize_url(raw)
    assert normalized == "https://fastapi.tiangolo.com/tutorial/first-steps/?id=123"


def test_normalize_url_handles_duplicate_slashes() -> None:
    """Verify duplicate slashes in path are normalized."""
    raw = "https://docs.pydantic.dev//concepts///models/"
    normalized = normalize_url(raw)
    assert normalized == "https://docs.pydantic.dev/concepts/models/"


def test_normalize_url_invalid_structure() -> None:
    """Verify invalid URL structures raise ValidationError."""
    with pytest.raises(ValidationError):
        normalize_url("not-a-valid-url")


def test_is_ip_private_or_restricted() -> None:
    """Verify detection of private, loopback, and link-local IP addresses."""
    # Private / loopback IPv4
    assert is_ip_private_or_restricted("127.0.0.1") is True
    assert is_ip_private_or_restricted("10.0.0.5") is True
    assert is_ip_private_or_restricted("192.168.1.100") is True
    assert is_ip_private_or_restricted("172.16.0.1") is True
    assert is_ip_private_or_restricted("169.254.169.254") is True  # link-local
    assert is_ip_private_or_restricted("0.0.0.0") is True

    # IPv6
    assert is_ip_private_or_restricted("::1") is True
    assert is_ip_private_or_restricted("fe80::1") is True

    # Public IPv4
    assert is_ip_private_or_restricted("8.8.8.8") is False
    assert is_ip_private_or_restricted("1.1.1.1") is False


def test_validate_url_safety_allows_valid_domain() -> None:
    """Verify safe URL matching allowed domains succeeds."""
    allowed = ["fastapi.tiangolo.com", "docs.pydantic.dev"]
    url = "https://fastapi.tiangolo.com/tutorial/first-steps/"
    validated = validate_url_safety(url, allowed, check_dns=False)
    assert validated == url


def test_validate_url_safety_rejects_disallowed_domain() -> None:
    """Verify URL outside allowed domains is rejected."""
    allowed = ["fastapi.tiangolo.com"]
    with pytest.raises(SSRFValidationError, match="not in allowed domains"):
        validate_url_safety("https://malicious-site.com/doc", allowed, check_dns=False)


def test_validate_url_safety_rejects_forbidden_schemes() -> None:
    """Verify non-http/https schemes raise SSRFValidationError."""
    allowed = ["fastapi.tiangolo.com"]
    with pytest.raises(SSRFValidationError, match="Forbidden URL scheme"):
        validate_url_safety("ftp://fastapi.tiangolo.com/doc", allowed, check_dns=False)

    with pytest.raises(SSRFValidationError, match="Forbidden URL scheme"):
        validate_url_safety("file:///etc/passwd", allowed, check_dns=False)


def test_validate_url_safety_rejects_cloud_metadata() -> None:
    """Verify cloud metadata endpoints are explicitly blocked."""
    allowed = ["169.254.169.254", "metadata.google.internal"]
    with pytest.raises(SSRFValidationError, match="matches cloud metadata"):
        validate_url_safety("http://169.254.169.254/latest/meta-data/", allowed, check_dns=False)

    with pytest.raises(SSRFValidationError, match="matches cloud metadata"):
        validate_url_safety(
            "http://metadata.google.internal/computeMetadata/v1/", allowed, check_dns=False
        )
