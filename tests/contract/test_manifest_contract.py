"""Contract tests for crawl manifests."""

import json
from datetime import UTC, datetime

from llmops.domain.models.crawler import CrawlManifest, PageSnapshot


def test_crawl_manifest_contract_schema() -> None:
    """Verify that CrawlManifest satisfies the required fields and serialization contract."""
    snapshot = PageSnapshot(
        url="https://fastapi.tiangolo.com/tutorial/",
        status_code=200,
        content_type="text/html",
        content_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        raw_filepath="data/raw/fastapi/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.html",
        byte_size=1024,
        headers={"content-type": "text/html"},
    )

    manifest = CrawlManifest(
        ingestion_run_id="crawl_fastapi_001",
        source_name="fastapi",
        base_url="https://fastapi.tiangolo.com/",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        total_pages_crawled=1,
        total_bytes=1024,
        pages=[snapshot],
        metadata={"crawler_version": "1.0.0"},
    )

    raw_json = manifest.model_dump_json()
    parsed = json.loads(raw_json)

    # Required contract keys
    expected_top_keys = {
        "ingestion_run_id",
        "source_name",
        "base_url",
        "started_at",
        "completed_at",
        "total_pages_crawled",
        "total_bytes",
        "pages",
        "metadata",
    }
    assert expected_top_keys.issubset(parsed.keys())

    # Snapshot contract keys
    snapshot_data = parsed["pages"][0]
    expected_snapshot_keys = {
        "url",
        "status_code",
        "content_type",
        "content_hash",
        "raw_filepath",
        "byte_size",
        "fetched_at",
        "headers",
    }
    assert expected_snapshot_keys.issubset(snapshot_data.keys())

    # Ensure round-trip validation works
    reloaded = CrawlManifest.model_validate_json(raw_json)
    assert reloaded.ingestion_run_id == manifest.ingestion_run_id
    assert len(reloaded.pages) == 1
    assert reloaded.pages[0].content_hash == snapshot.content_hash
