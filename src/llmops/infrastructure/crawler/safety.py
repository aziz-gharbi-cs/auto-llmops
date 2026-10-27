"""SSRF protection, URL normalization, and allowlist validation for crawler safety."""

import ipaddress
import re
import socket
from collections.abc import Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from llmops.domain.exceptions import ValidationError

# Known cloud metadata hosts and IP addresses
CLOUD_METADATA_HOSTS = {
    "169.254.169.254",
    "metadata.google.internal",
    "100.100.100.200",  # Alibaba cloud metadata
    "instance-data",
}

TRACKING_PARAM_PATTERNS = re.compile(r"^(utm_.*|ref|fbclid|gclid|mc_cid|mc_eid)$", re.IGNORECASE)


class SSRFValidationError(ValidationError):
    """Raised when a URL violates SSRF safety constraints."""


def normalize_url(raw_url: str) -> str:
    """Normalize URL: strip fragments, sort and filter tracking params, lowercase host."""
    parsed = urlparse(raw_url.strip())
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise SSRFValidationError(
            f"Forbidden URL scheme '{scheme}'. Only http and https are permitted."
        )

    netloc = parsed.netloc.lower()
    if not netloc:
        raise ValidationError(f"Invalid URL structure (missing host): '{raw_url}'")

    path = parsed.path or "/"
    # Clean redundant duplicate slashes in path
    path = re.sub(r"/{2,}", "/", path)

    # Filter out tracking query params while preserving meaningful ones
    if parsed.query:
        query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        filtered_pairs = [(k, v) for k, v in query_pairs if not TRACKING_PARAM_PATTERNS.match(k)]
        filtered_pairs.sort(key=lambda x: x[0])
        new_query = urlencode(filtered_pairs)
    else:
        new_query = ""

    # Strip fragments entirely
    return urlunparse((scheme, netloc, path, parsed.params, new_query, ""))


def is_ip_private_or_restricted(ip_str: str) -> bool:
    """Return True if an IP address is private, loopback, link-local, multicast, or reserved."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        return True


def validate_url_safety(
    url: str,
    allowed_domains: Sequence[str],
    check_dns: bool = True,
) -> str:
    """Validate that a URL is safe against SSRF and belongs to the allowed domains.

    Returns the normalized URL on success, or raises SSRFValidationError / ValidationError.
    """
    normalized = normalize_url(url)
    parsed = urlparse(normalized)

    # 1. Scheme check
    if parsed.scheme not in {"http", "https"}:
        raise SSRFValidationError(
            f"Forbidden URL scheme '{parsed.scheme}'. Only http and https are permitted."
        )

    # 2. Host presence and metadata check
    hostname = parsed.hostname
    if not hostname:
        raise SSRFValidationError("URL contains no hostname.")

    hostname_lower = hostname.lower()
    if hostname_lower in CLOUD_METADATA_HOSTS:
        raise SSRFValidationError(f"Target host '{hostname}' matches cloud metadata service.")

    # 3. Domain allowlist check
    domain_allowed = False
    for allowed in allowed_domains:
        allowed_lower = allowed.lower()
        if hostname_lower == allowed_lower or hostname_lower.endswith("." + allowed_lower):
            domain_allowed = True
            break

    if not domain_allowed:
        raise SSRFValidationError(
            f"Host '{hostname}' is not in allowed domains: {list(allowed_domains)}"
        )

    # 4. SSRF IP resolution check
    if check_dns:
        try:
            # Check if hostname is an explicit IP
            if is_ip_private_or_restricted(hostname_lower):
                raise SSRFValidationError(
                    f"Host IP '{hostname}' resolves to a restricted/private network address."
                )
        except ValueError:
            # Hostname is a domain name, not an IP literal
            pass

        try:
            addr_info = socket.getaddrinfo(hostname_lower, None)
            for item in addr_info:
                sockaddr = item[4]
                ip_addr = str(sockaddr[0])
                if is_ip_private_or_restricted(ip_addr):
                    raise SSRFValidationError(
                        f"Resolved address '{ip_addr}' for '{hostname}' is private or restricted."
                    )
        except socket.gaierror as err:
            raise SSRFValidationError(
                f"Failed to resolve DNS for hostname '{hostname}': {err}"
            ) from err

    return normalized
