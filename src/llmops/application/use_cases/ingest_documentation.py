"""Use case coordinating documentation ingestion across configured sources."""

import logging
from pathlib import Path

import yaml

from llmops.domain.exceptions import ConfigurationError
from llmops.domain.models.crawler import CrawlManifest, IngestionConfig
from llmops.infrastructure.crawler.service import DocumentationCrawler

logger = logging.getLogger(__name__)


class IngestDocumentationUseCase:
    """Orchestrates source acquisition according to ingestion configuration."""

    def __init__(self, config_path: str | Path = "configs/ingestion.yaml") -> None:
        self.config_path = Path(config_path)

    def load_config(self) -> IngestionConfig:
        """Load and validate the ingestion configuration from YAML."""
        if not self.config_path.exists():
            raise ConfigurationError(f"Ingestion config not found at '{self.config_path}'")

        try:
            raw_data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
            return IngestionConfig.model_validate(raw_data)
        except Exception as exc:  # noqa: BLE001
            raise ConfigurationError(
                f"Failed to parse ingestion config at '{self.config_path}': {exc}"
            ) from exc

    def execute(self, target_source: str | None = None) -> list[CrawlManifest]:
        """Execute crawler for all configured sources or a specific source."""
        config = self.load_config()
        manifests: list[CrawlManifest] = []

        sources_to_crawl = config.sources
        if target_source:
            sources_to_crawl = [
                s for s in config.sources if s.name.lower() == target_source.lower()
            ]
            if not sources_to_crawl:
                raise ConfigurationError(
                    f"Target source '{target_source}' not found in configuration. "
                    f"Available sources: {[s.name for s in config.sources]}"
                )

        for source in sources_to_crawl:
            logger.info("Executing ingestion for source: %s", source.name)
            crawler = DocumentationCrawler(
                source_config=source,
                global_config=config.crawler_defaults,
                storage_dir=config.storage_dir,
            )
            manifest = crawler.run()
            manifests.append(manifest)

        return manifests
