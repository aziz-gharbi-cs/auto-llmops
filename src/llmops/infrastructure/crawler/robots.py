"""Robots.txt parsing and compliance checker with SSRF protection."""

import logging
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from llmops.infrastructure.crawler.safety import validate_url_safety

logger = logging.getLogger(__name__)


class SafeRobotChecker:
    """SSRF-safe robots.txt checker for a given source host."""

    def __init__(
        self,
        base_url: str,
        user_agent: str = "AntigravityLLMOpsBot",
        timeout: float = 5.0,
        allowed_domains: list[str] | None = None,
    ) -> None:
        self.base_url = base_url
        self.user_agent = user_agent
        self.timeout = timeout
        self.allowed_domains = allowed_domains or []
        self._parser = RobotFileParser()
        self._loaded = False
        self._allow_all = False
        self._disallow_all = False

    def load(self, client: httpx.Client | None = None) -> None:
        """Fetch and parse robots.txt safely."""
        parsed = urlparse(self.base_url)
        robots_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", "/robots.txt")

        try:
            if self.allowed_domains:
                validate_url_safety(robots_url, self.allowed_domains, check_dns=False)

            if client is not None:
                response = client.get(robots_url, timeout=self.timeout)
            else:
                with httpx.Client(timeout=self.timeout) as c:
                    response = c.get(robots_url)

            if response.status_code == 200:
                self._parser.parse(response.text.splitlines())
                self._loaded = True
            elif response.status_code in {401, 403}:
                # Disallow all if unauthorized
                self._disallow_all = True
                self._loaded = True
            elif response.status_code == 404:
                # Allow all if not found
                self._allow_all = True
                self._loaded = True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not fetch robots.txt for %s: %s. Defaulting to allow.",
                self.base_url,
                exc,
            )
            self._allow_all = True
            self._loaded = True

    def can_fetch(self, url: str) -> bool:
        """Check if the given URL is allowed by robots.txt."""
        if not self._loaded:
            self.load()
        if self._disallow_all:
            return False
        if self._allow_all:
            return True
        return bool(self._parser.can_fetch(self.user_agent, url))
