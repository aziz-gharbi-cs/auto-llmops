"""Domain models for crawler configuration, raw snapshots, and crawl manifests."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class CrawlerSourceConfig(BaseModel):
    """Configuration for a specific documentation source."""

    name: str
    type: str = Field(default="documentation")
    base_url: str
    allowed_domains: list[str] = Field(default_factory=list)
    seed_urls: list[str] = Field(default_factory=list)
    max_pages: int = Field(default=50, ge=1, le=1000)


class CrawlerGlobalConfig(BaseModel):
    """Global defaults and safety limits for the crawler."""

    user_agent: str = Field(default="AntigravityLLMOpsBot/1.0")
    timeout_seconds: float = Field(default=10.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    backoff_factor: float = Field(default=1.5, ge=1.0)
    rate_limit_delay_seconds: float = Field(default=0.5, ge=0.0)
    max_page_size_bytes: int = Field(default=5242880, gt=0)  # 5 MB
    respect_robots_txt: bool = Field(default=True)
    allowed_content_types: list[str] = Field(
        default_factory=lambda: ["text/html", "text/plain", "text/markdown"]
    )


class IngestionConfig(BaseModel):
    """Full configuration model loaded from configs/ingestion.yaml."""

    version: str = Field(default="1.0.0")
    storage_dir: str = Field(default="data/raw")
    crawler_defaults: CrawlerGlobalConfig = Field(default_factory=CrawlerGlobalConfig)
    sources: list[CrawlerSourceConfig] = Field(default_factory=list)


class PageSnapshot(BaseModel):
    """Metadata record for a single acquired raw page."""

    url: str
    status_code: int
    content_type: str
    content_hash: str
    raw_filepath: str
    byte_size: int
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    headers: dict[str, str] = Field(default_factory=dict)


class CrawlManifest(BaseModel):
    """Provenance manifest recording all snapshots acquired during a crawl run."""

    ingestion_run_id: str
    source_name: str
    base_url: str
    started_at: datetime
    completed_at: datetime
    total_pages_crawled: int = Field(default=0)
    total_bytes: int = Field(default=0)
    pages: list[PageSnapshot] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
