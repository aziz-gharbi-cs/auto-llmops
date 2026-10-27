"""Integration tests for ContinuousLLMOpsPipeline (Milestone 14)."""

from pathlib import Path

from llmops.application.use_cases.continuous_pipeline import (
    ContinuousLLMOpsPipeline,
    ContinuousPipelineResult,
)
from llmops.domain.models.crawler import CrawlManifest, PageSnapshot
from llmops.domain.models.evaluation import QualityGateThresholds
from llmops.infrastructure.storage.local_storage import LocalStorageAdapter


def test_continuous_llmops_pipeline_run(tmp_path: Path) -> None:
    # 1. Setup raw crawled snapshots
    raw_dir = tmp_path / "raw"
    fastapi_raw = raw_dir / "fastapi"
    fastapi_raw.mkdir(parents=True)

    html_file1 = fastapi_raw / "page1.html"
    html_file1.write_text(
        "<html><body><h1>FastAPI Concepts</h1><p>FastAPI OpenAPI docs.</p></body></html>",
        encoding="utf-8",
    )
    html_file2 = fastapi_raw / "page2.html"
    html_file2.write_text(
        "<html><body><h1>Pydantic Validation</h1><p>Pydantic models.</p></body></html>",
        encoding="utf-8",
    )
    html_file3 = fastapi_raw / "page3.html"
    html_file3.write_text(
        "<html><body><h1>SQLAlchemy ORM</h1><p>SQLAlchemy database sessions.</p></body></html>",
        encoding="utf-8",
    )

    from datetime import UTC, datetime

    now = datetime.now(UTC)
    manifest = CrawlManifest(
        ingestion_run_id="crawl-123",
        source_name="fastapi",
        base_url="https://fastapi.tiangolo.com",
        started_at=now,
        completed_at=now,
        total_pages_crawled=3,
        total_bytes=1000,
        pages=[
            PageSnapshot(
                url="https://fastapi.tiangolo.com/page1",
                status_code=200,
                content_type="text/html",
                content_hash="h1" * 32,
                raw_filepath=str(html_file1),
                byte_size=100,
            ),
            PageSnapshot(
                url="https://fastapi.tiangolo.com/page2",
                status_code=200,
                content_type="text/html",
                content_hash="h2" * 32,
                raw_filepath=str(html_file2),
                byte_size=100,
            ),
            PageSnapshot(
                url="https://fastapi.tiangolo.com/page3",
                status_code=200,
                content_type="text/html",
                content_hash="h3" * 32,
                raw_filepath=str(html_file3),
                byte_size=100,
            ),
        ],
    )
    (fastapi_raw / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    # 2. Setup pipeline
    storage = LocalStorageAdapter(base_dir=tmp_path / "storage")
    pipeline = ContinuousLLMOpsPipeline(
        storage=storage,
        output_dir=tmp_path / "output",
    )

    # 3. Execute pipeline with permissive thresholds for mock test
    thresholds = QualityGateThresholds(
        min_retrieval_mrr=0.1,
        min_generation_groundedness=1.0,
        min_generation_correctness=1.0,
        min_rag_grounding_rate=0.5,
    )
    result: ContinuousPipelineResult = pipeline.run_lifecycle(
        raw_documents_dir=raw_dir,
        force_retrain=True,
        training_profile="smoke",
        thresholds=thresholds,
    )

    # 4. Verify results
    assert result.pipeline_run_id.startswith("pipe-")
    assert result.crawled_docs_count == 3
    assert result.chunks_count >= 3
    assert result.training_executed is True
    assert result.evaluation_passed is True
    assert result.promoted_stage == "production"
    assert result.manifest_path is not None
    assert Path(result.manifest_path).exists()
    assert "processing" in result.stage_durations_sec
    assert "curation" in result.stage_durations_sec
    assert "training" in result.stage_durations_sec
    assert "evaluation" in result.stage_durations_sec
