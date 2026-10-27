"""Integration test for the document processing pipeline."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from llmops.application.use_cases.process_documents import ProcessDocumentsUseCase
from llmops.domain.models.crawler import CrawlManifest, PageSnapshot


def test_process_documents_pipeline(tmp_path: Path) -> None:
    """Verify end-to-end processing pipeline from raw snapshots to canonical chunks and reports."""
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"

    source_dir = raw_dir / "fastapi"
    source_dir.mkdir(parents=True)

    # 1. Create sample raw snapshot files
    html1 = """
    <html>
        <body>
            <h1>FastAPI Tutorial</h1>
            <p>FastAPI is a modern web framework for Python.</p>
            <h2>Path Parameters</h2>
            <p>You can declare path parameters using standard Python format strings.</p>
            <pre><code class="language-python">@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}</code></pre>
        </body>
    </html>
    """
    html2 = """
    <html>
        <body>
            <h1>Query Parameters</h1>
            <p>
                When you declare other function parameters that are not part of the path,
                they are query parameters.
            </p>
        </body>
    </html>
    """

    hash1 = hashlib.sha256(html1.encode("utf-8")).hexdigest()
    hash2 = hashlib.sha256(html2.encode("utf-8")).hexdigest()

    file1 = source_dir / f"{hash1}.html"
    file2 = source_dir / f"{hash2}.html"
    file1.write_text(html1, encoding="utf-8")
    file2.write_text(html2, encoding="utf-8")

    # 2. Create manifest.json
    manifest = CrawlManifest(
        ingestion_run_id="run-fastapi-test",
        source_name="fastapi",
        base_url="https://fastapi.tiangolo.com/",
        started_at=datetime(2026, 10, 27, 10, 0, 0, tzinfo=UTC),
        completed_at=datetime(2026, 10, 27, 10, 1, 0, tzinfo=UTC),
        total_pages_crawled=2,
        total_bytes=len(html1) + len(html2),
        pages=[
            PageSnapshot(
                url="https://fastapi.tiangolo.com/tutorial/",
                status_code=200,
                content_type="text/html",
                content_hash=hash1,
                raw_filepath=str(file1),
                byte_size=len(html1),
            ),
            PageSnapshot(
                url="https://fastapi.tiangolo.com/tutorial/query-params/",
                status_code=200,
                content_type="text/html",
                content_hash=hash2,
                raw_filepath=str(file2),
                byte_size=len(html2),
            ),
        ],
    )
    (source_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    # 3. Execute ProcessDocumentsUseCase
    use_case = ProcessDocumentsUseCase(
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        dataset_version="v1",
        target_chunk_size=100,
        max_chunk_size=200,
    )
    report = use_case.execute()

    # 4. Verify outputs
    assert report.total_documents == 2
    assert report.total_chunks >= 2
    assert report.exact_duplicates_removed == 0
    assert report.source_distribution["fastapi"] == 2

    # Check interim documents file
    docs_file = interim_dir / "documents.jsonl"
    assert docs_file.exists()
    doc_lines = [json.loads(line) for line in docs_file.read_text(encoding="utf-8").splitlines()]
    assert len(doc_lines) == 2
    assert "FastAPI Tutorial" in doc_lines[0]["title"]
    assert "```python" in doc_lines[0]["content"]

    # Check processed chunks file
    chunks_file = processed_dir / "chunks.jsonl"
    assert chunks_file.exists()
    chunk_lines = [
        json.loads(line) for line in chunks_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(chunk_lines) >= 2
    assert chunk_lines[0]["provenance"]["source_name"] == "fastapi"
    assert chunk_lines[0]["provenance"]["dataset_version"] == "v1"

    # Check quality report artifact
    report_file = processed_dir / "quality_report.json"
    assert report_file.exists()
    report_data = json.loads(report_file.read_text(encoding="utf-8"))
    assert report_data["total_documents"] == 2
    assert report_data["chunk_token_stats"]["total_tokens"] > 0
