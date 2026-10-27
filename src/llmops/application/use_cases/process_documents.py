"""Use case orchestrating document parsing, cleaning, chunking, deduplication, and reporting."""

import json
import logging
from pathlib import Path

from llmops.domain.interfaces.storage import BaseStorage
from llmops.domain.models.crawler import CrawlManifest
from llmops.domain.models.document import Document, DocumentChunk
from llmops.infrastructure.chunker.semantic_chunker import SemanticChunker
from llmops.infrastructure.parser.deduplicator import Deduplicator
from llmops.infrastructure.parser.html_parser import HTMLDocumentParser
from llmops.infrastructure.parser.quality_report import QualityReport, generate_quality_report

logger = logging.getLogger(__name__)


class ProcessDocumentsUseCase:
    """Orchestrates parsing raw snapshots into canonical documents and semantic chunks."""

    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        interim_dir: str | Path = "data/interim",
        processed_dir: str | Path = "data/processed",
        dataset_version: str = "v1",
        target_chunk_size: int = 400,
        max_chunk_size: int = 600,
        storage: BaseStorage | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.interim_dir = Path(interim_dir)
        self.processed_dir = Path(processed_dir)
        self.dataset_version = dataset_version
        self.storage = storage

        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.parser = HTMLDocumentParser()
        self.chunker = SemanticChunker(
            target_chunk_size=target_chunk_size,
            max_chunk_size=max_chunk_size,
        )
        self.deduplicator = Deduplicator()

    def load_manifests(self) -> list[CrawlManifest]:
        """Find and load all manifest.json files in the raw storage directory."""
        manifests: list[CrawlManifest] = []
        for manifest_file in self.raw_dir.glob("*/manifest.json"):
            try:
                manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
                manifests.append(CrawlManifest.model_validate(manifest_data))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load manifest at '%s': %s", manifest_file, exc)
        return manifests

    def execute(
        self,
        source_name: str | None = None,
        raw_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        run_id: str | None = None,
    ) -> QualityReport:
        """Execute full parsing, chunking, deduplication, and quality report generation."""
        if raw_dir:
            self.raw_dir = Path(raw_dir)
        if output_dir:
            self.processed_dir = Path(output_dir)
            self.processed_dir.mkdir(parents=True, exist_ok=True)

        manifests = self.load_manifests()
        if source_name:
            manifests = [m for m in manifests if m.source_name.lower() == source_name.lower()]

        if not manifests:
            logger.info("No crawl manifests found in '%s'.", self.raw_dir)
            return QualityReport(dataset_version=self.dataset_version)

        raw_documents: list[Document] = []
        rejected_count = 0

        # Step 1: Parse raw files into canonical Documents
        for manifest in manifests:
            for page in manifest.pages:
                raw_path = Path(page.raw_filepath)
                if not raw_path.exists():
                    logger.warning("Raw snapshot file missing at '%s'", raw_path)
                    rejected_count += 1
                    continue

                try:
                    content_text = raw_path.read_text(encoding="utf-8", errors="replace")
                    doc = self.parser.parse(
                        html_content=content_text,
                        source_name=manifest.source_name,
                        source_url=page.url,
                    )
                    if len(doc.content.strip()) < 10:
                        # Skip empty or trivial documents
                        rejected_count += 1
                        continue
                    raw_documents.append(doc)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to parse '%s': %s", raw_path, exc)
                    rejected_count += 1

        # Step 2: Deduplicate documents
        unique_docs, doc_exact_dups, doc_near_dups = self.deduplicator.deduplicate(
            raw_documents, check_near_duplicates=True
        )

        # Write canonical documents to data/interim/documents.jsonl
        interim_docs_path = self.interim_dir / "documents.jsonl"
        with interim_docs_path.open("w", encoding="utf-8") as f:
            for doc in unique_docs:
                f.write(doc.model_dump_json() + "\n")

        # Step 3: Semantic Chunking
        raw_chunks: list[DocumentChunk] = []
        for doc in unique_docs:
            chunks = self.chunker.chunk_document(
                document=doc,
                dataset_version=self.dataset_version,
            )
            raw_chunks.extend(chunks)

        # Step 4: Deduplicate chunks
        unique_chunks, chunk_exact_dups, chunk_near_dups = self.deduplicator.deduplicate(
            raw_chunks, check_near_duplicates=True
        )

        # Write processed chunks to data/processed/chunks.jsonl
        processed_chunks_path = self.processed_dir / "chunks.jsonl"
        with processed_chunks_path.open("w", encoding="utf-8") as f:
            for chunk in unique_chunks:
                f.write(chunk.model_dump_json() + "\n")

        # Step 5: Generate quality report
        report = generate_quality_report(
            documents=unique_docs,
            chunks=unique_chunks,
            exact_dups=doc_exact_dups + chunk_exact_dups,
            near_dups=doc_near_dups + chunk_near_dups,
            rejected=rejected_count,
            dataset_version=self.dataset_version,
        )

        # Write quality report artifact to data/processed/quality_report.json
        report_path = self.processed_dir / "quality_report.json"
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

        logger.info(
            "Processing complete: %d unique documents, %d chunks saved to '%s'",
            len(unique_docs),
            len(unique_chunks),
            self.processed_dir,
        )
        return report
