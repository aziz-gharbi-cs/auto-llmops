"""Unit tests for the documentation crawler service using mock HTTP transport."""

import hashlib
import json
from pathlib import Path

import httpx

from llmops.domain.models.crawler import CrawlerGlobalConfig, CrawlerSourceConfig
from llmops.infrastructure.crawler.service import DocumentationCrawler


def make_mock_client(responses: dict[str, tuple[int, dict[str, str], bytes]]) -> httpx.Client:
    """Create an httpx.Client backed by a mock handler dictionary."""

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if url_str in responses:
            status_code, headers, content = responses[url_str]
            return httpx.Response(status_code=status_code, headers=headers, content=content)
        return httpx.Response(status_code=404, text="Not Found")

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


def test_crawler_crawls_seed_and_saves_snapshots(tmp_path: Path) -> None:
    """Verify crawler crawls seeds, extracts links, saves raw html and writes manifest."""
    source = CrawlerSourceConfig(
        name="fastapi_test",
        base_url="https://fastapi.tiangolo.com/",
        allowed_domains=["fastapi.tiangolo.com"],
        seed_urls=["https://fastapi.tiangolo.com/"],
        max_pages=5,
    )
    global_cfg = CrawlerGlobalConfig(
        respect_robots_txt=False,
        rate_limit_delay_seconds=0.0,
    )

    page1_html = b"<html><body><h1>FastAPI</h1><a href='/tutorial/'>Tutorial</a></body></html>"
    page2_html = b"<html><body><h1>Tutorial</h1><p>Welcome to FastAPI tutorial</p></body></html>"

    mock_responses = {
        "https://fastapi.tiangolo.com/": (
            200,
            {"content-type": "text/html"},
            page1_html,
        ),
        "https://fastapi.tiangolo.com/tutorial/": (
            200,
            {"content-type": "text/html"},
            page2_html,
        ),
    }

    client = make_mock_client(mock_responses)
    crawler = DocumentationCrawler(
        source_config=source,
        global_config=global_cfg,
        storage_dir=tmp_path,
        client=client,
        check_dns=False,
    )

    manifest = crawler.run()

    # Manifest checks
    assert manifest.source_name == "fastapi_test"
    assert manifest.total_pages_crawled == 2
    assert len(manifest.pages) == 2

    # Verify snapshot files written
    p1_hash = hashlib.sha256(page1_html).hexdigest()
    p2_hash = hashlib.sha256(page2_html).hexdigest()

    p1_file = tmp_path / "fastapi_test" / f"{p1_hash}.html"
    p2_file = tmp_path / "fastapi_test" / f"{p2_hash}.html"
    manifest_file = tmp_path / "fastapi_test" / "manifest.json"

    assert p1_file.exists()
    assert p1_file.read_bytes() == page1_html
    assert p2_file.exists()
    assert p2_file.read_bytes() == page2_html
    assert manifest_file.exists()

    manifest_json = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest_json["total_pages_crawled"] == 2


def test_crawler_respects_robots_txt(tmp_path: Path) -> None:
    """Verify crawler respects robots.txt disallow rules."""
    source = CrawlerSourceConfig(
        name="pydantic_test",
        base_url="https://docs.pydantic.dev/",
        allowed_domains=["docs.pydantic.dev"],
        seed_urls=["https://docs.pydantic.dev/"],
        max_pages=5,
    )
    global_cfg = CrawlerGlobalConfig(
        respect_robots_txt=True,
        rate_limit_delay_seconds=0.0,
    )

    robots_txt = b"User-agent: *\nDisallow: /private/\n"
    index_html = (
        b"<html><body><a href='/public/'>Public</a><a href='/private/'>Secret</a></body></html>"
    )
    public_html = b"<html><body><h1>Public Page</h1></body></html>"
    private_html = b"<html><body><h1>Should not crawl</h1></body></html>"

    mock_responses = {
        "https://docs.pydantic.dev/robots.txt": (
            200,
            {"content-type": "text/plain"},
            robots_txt,
        ),
        "https://docs.pydantic.dev/": (
            200,
            {"content-type": "text/html"},
            index_html,
        ),
        "https://docs.pydantic.dev/public/": (
            200,
            {"content-type": "text/html"},
            public_html,
        ),
        "https://docs.pydantic.dev/private/": (
            200,
            {"content-type": "text/html"},
            private_html,
        ),
    }

    client = make_mock_client(mock_responses)
    crawler = DocumentationCrawler(
        source_config=source,
        global_config=global_cfg,
        storage_dir=tmp_path,
        client=client,
        check_dns=False,
    )

    manifest = crawler.run()

    crawled_urls = [p.url for p in manifest.pages]
    assert "https://docs.pydantic.dev/" in crawled_urls
    assert "https://docs.pydantic.dev/public/" in crawled_urls
    assert "https://docs.pydantic.dev/private/" not in crawled_urls


def test_crawler_page_size_limit_guard(tmp_path: Path) -> None:
    """Verify page exceeding max_page_size_bytes is skipped."""
    source = CrawlerSourceConfig(
        name="size_test",
        base_url="https://docs.sqlalchemy.org/",
        allowed_domains=["docs.sqlalchemy.org"],
        seed_urls=["https://docs.sqlalchemy.org/large.html"],
        max_pages=2,
    )
    global_cfg = CrawlerGlobalConfig(
        max_page_size_bytes=50,  # 50 bytes limit
        respect_robots_txt=False,
        rate_limit_delay_seconds=0.0,
    )

    large_content = b"A" * 100  # 100 bytes > 50 bytes

    mock_responses = {
        "https://docs.sqlalchemy.org/large.html": (
            200,
            {"content-type": "text/html"},
            large_content,
        ),
    }

    client = make_mock_client(mock_responses)
    crawler = DocumentationCrawler(
        source_config=source,
        global_config=global_cfg,
        storage_dir=tmp_path,
        client=client,
        check_dns=False,
    )

    manifest = crawler.run()
    # Large page should have been skipped safely without crashing the crawler
    assert manifest.total_pages_crawled == 0
