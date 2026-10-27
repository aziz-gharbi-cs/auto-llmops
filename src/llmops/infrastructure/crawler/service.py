"""Documentation crawler service implementing safe, rate-limited, raw snapshot acquisition."""

import hashlib
import logging
import time
import uuid
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from llmops.domain.exceptions import SourceAcquisitionError
from llmops.domain.models.crawler import (
    CrawlerGlobalConfig,
    CrawlerSourceConfig,
    CrawlManifest,
    PageSnapshot,
)
from llmops.infrastructure.crawler.robots import SafeRobotChecker
from llmops.infrastructure.crawler.safety import (
    SSRFValidationError,
    normalize_url,
    validate_url_safety,
)

logger = logging.getLogger(__name__)


class DocumentationCrawler:
    """Service to crawl technical documentation and store raw snapshots with manifests."""

    def __init__(
        self,
        source_config: CrawlerSourceConfig,
        global_config: CrawlerGlobalConfig | None = None,
        storage_dir: str | Path = "data/raw",
        client: httpx.Client | None = None,
        check_dns: bool = True,
    ) -> None:
        self.source = source_config
        self.config = global_config or CrawlerGlobalConfig()
        self.storage_dir = Path(storage_dir) / self.source.name
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._client = client
        self._check_dns = check_dns

        self.robot_checker: SafeRobotChecker | None = None
        if self.config.respect_robots_txt:
            self.robot_checker = SafeRobotChecker(
                base_url=self.source.base_url,
                user_agent=self.config.user_agent,
                timeout=self.config.timeout_seconds,
                allowed_domains=self.source.allowed_domains,
            )

    def _get_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(
            headers={"User-Agent": self.config.user_agent},
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
        )

    def extract_links(self, html_content: str, current_url: str) -> list[str]:
        """Extract and normalize safe downstream links within the allowed domains."""
        soup = BeautifulSoup(html_content, "html.parser")
        discovered: list[str] = []

        for anchor in soup.find_all("a", href=True):
            raw_href = anchor.get("href")
            if not isinstance(raw_href, str):
                continue
            href = raw_href.strip()
            if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
                continue

            full_url = urljoin(current_url, href)
            try:
                safe_url = validate_url_safety(
                    full_url,
                    self.source.allowed_domains,
                    check_dns=self._check_dns,
                )
                discovered.append(safe_url)
            except (SSRFValidationError, Exception):  # noqa: BLE001
                # Skip invalid or unsafe links
                continue

        return discovered

    def fetch_page_with_retries(
        self, client: httpx.Client, url: str
    ) -> tuple[int, str, bytes, dict[str, str]]:
        """Fetch page content with exponential backoff retries and size guard."""
        retries = 0
        backoff = self.config.backoff_factor

        while retries <= self.config.max_retries:
            try:
                resp = client.get(url)
                status_code = resp.status_code
                content_type = resp.headers.get("content-type", "")

                # Check content length
                raw_bytes = resp.content
                if len(raw_bytes) > self.config.max_page_size_bytes:
                    raise SourceAcquisitionError(
                        f"Page at '{url}' exceeds max allowed size of "
                        f"{self.config.max_page_size_bytes} bytes (got {len(raw_bytes)})"
                    )

                headers = {k.lower(): v for k, v in resp.headers.items()}
                return status_code, content_type, raw_bytes, headers

            except httpx.HTTPError as err:
                retries += 1
                if retries > self.config.max_retries:
                    raise SourceAcquisitionError(
                        f"Max retries exceeded fetching '{url}': {err}"
                    ) from err
                time.sleep(backoff)
                backoff *= 2

        raise SourceAcquisitionError(f"Failed to fetch '{url}' after {retries} retries.")

    def run(self) -> CrawlManifest:
        """Execute the crawl according to configuration and write raw snapshot files."""
        run_id = f"crawl_{self.source.name}_{uuid.uuid4().hex[:8]}"
        started_at = datetime.now(UTC)
        logger.info("Starting documentation crawl for %s (Run ID: %s)", self.source.name, run_id)

        # Queue setup
        queue: deque[str] = deque()
        visited: set[str] = set()
        pages: list[PageSnapshot] = []
        total_bytes = 0

        # Initialize seeds
        seeds = self.source.seed_urls or [self.source.base_url]
        for seed in seeds:
            try:
                norm_seed = normalize_url(seed)
                queue.append(norm_seed)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Invalid seed URL '%s': %s", seed, exc)

        client = self._get_client()

        # Initialize robots if enabled
        if self.robot_checker is not None:
            self.robot_checker.load(client)

        try:
            while queue and len(visited) < self.source.max_pages:
                current_url = queue.popleft()
                if current_url in visited:
                    continue

                # Safety check
                try:
                    validate_url_safety(
                        current_url,
                        self.source.allowed_domains,
                        check_dns=self._check_dns,
                    )
                except SSRFValidationError as err:
                    logger.warning("Skipping unsafe URL '%s': %s", current_url, err)
                    continue

                # Robots check
                if self.robot_checker and not self.robot_checker.can_fetch(current_url):
                    logger.info("Blocked by robots.txt: %s", current_url)
                    continue

                visited.add(current_url)

                try:
                    status_code, content_type, raw_bytes, headers = self.fetch_page_with_retries(
                        client, current_url
                    )
                except SourceAcquisitionError as err:
                    logger.warning("Failed to fetch '%s': %s", current_url, err)
                    continue

                if status_code != 200:
                    logger.debug("Non-200 status code %s for '%s'", status_code, current_url)
                    continue

                # Validate content type is allowed
                is_allowed_type = any(
                    allowed_t in content_type for allowed_t in self.config.allowed_content_types
                )
                if not is_allowed_type:
                    logger.debug(
                        "Skipping unallowed content type '%s' for '%s'", content_type, current_url
                    )
                    continue

                # Compute content hash
                content_hash = hashlib.sha256(raw_bytes).hexdigest()

                # Save raw snapshot
                parsed_path = urlparse(current_url).path
                ext = ".html" if "html" in content_type else ".txt"
                if parsed_path.endswith((".md", ".rst")):
                    ext = Path(parsed_path).suffix

                snapshot_filename = f"{content_hash}{ext}"
                snapshot_path = self.storage_dir / snapshot_filename
                snapshot_path.write_bytes(raw_bytes)

                snapshot = PageSnapshot(
                    url=current_url,
                    status_code=status_code,
                    content_type=content_type,
                    content_hash=content_hash,
                    raw_filepath=str(snapshot_path),
                    byte_size=len(raw_bytes),
                    headers=headers,
                )
                pages.append(snapshot)
                total_bytes += len(raw_bytes)

                # Extract further links if HTML
                if "html" in content_type:
                    discovered = self.extract_links(
                        raw_bytes.decode("utf-8", errors="replace"), current_url
                    )
                    for new_url in discovered:
                        if new_url not in visited and new_url not in queue:
                            queue.append(new_url)

                # Respect rate limiting
                if self.config.rate_limit_delay_seconds > 0:
                    time.sleep(self.config.rate_limit_delay_seconds)

        finally:
            if self._client is None:
                client.close()

        completed_at = datetime.now(UTC)
        manifest = CrawlManifest(
            ingestion_run_id=run_id,
            source_name=self.source.name,
            base_url=self.source.base_url,
            started_at=started_at,
            completed_at=completed_at,
            total_pages_crawled=len(pages),
            total_bytes=total_bytes,
            pages=pages,
        )

        # Write manifest.json
        manifest_path = self.storage_dir / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "Crawl completed for %s: %d pages (%d bytes) saved to %s",
            self.source.name,
            len(pages),
            total_bytes,
            self.storage_dir,
        )
        return manifest
