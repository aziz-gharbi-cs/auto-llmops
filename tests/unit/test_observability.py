"""Unit tests for Observability: metrics, tracing, structured logging (Milestone 10)."""

import json
import logging

import pytest

from llmops.infrastructure.observability.logging import (
    StructuredJsonFormatter,
    redact_sensitive_text,
)
from llmops.infrastructure.observability.metrics import (
    Counter,
    Histogram,
    MetricsRegistry,
    registry,
)
from llmops.infrastructure.observability.tracer import (
    LLMOpsTracer,
    Span,
)
from llmops.interfaces.api.routes.health import prometheus_metrics


def test_counter_metric_increments() -> None:
    counter = Counter(name="test_counter", description="Test counter description")
    counter.inc(1.0, labels={"env": "test"})
    counter.inc(2.5, labels={"env": "test"})

    samples = counter.get_samples()
    assert len(samples) == 1
    assert samples[0].labels == {"env": "test"}
    assert samples[0].value == 3.5

    with pytest.raises(ValueError, match="Counters can only be incremented"):
        counter.inc(-1.0)


def test_histogram_metric_observations() -> None:
    hist = Histogram(name="test_hist", description="Test histogram", buckets=[0.1, 0.5, 1.0])
    hist.observe(0.05, labels={"route": "/v1/chat"})
    hist.observe(0.25, labels={"route": "/v1/chat"})
    hist.observe(1.50, labels={"route": "/v1/chat"})

    stats = hist.get_stats()
    assert len(stats) == 1
    labels, count, total_sum, bucket_counts = stats[0]
    assert labels == {"route": "/v1/chat"}
    assert count == 3
    assert total_sum == pytest.approx(1.80)
    assert bucket_counts[0.1] == 1  # 0.05 <= 0.1
    assert bucket_counts[0.5] == 2  # 0.05, 0.25 <= 0.5
    assert bucket_counts[1.0] == 2  # 0.05, 0.25 <= 1.0


def test_metrics_registry_section42_metrics_and_exposition() -> None:
    reg = MetricsRegistry()

    # Verify all Section 42 metrics exist
    for metric_name in [
        "request_count",
        "request_latency",
        "error_count",
        "retrieval_latency",
        "embedding_latency",
        "generation_latency",
        "tokens_in",
        "tokens_out",
        "model_requests",
        "cache_hits",
    ]:
        if "latency" in metric_name:
            assert metric_name in reg._histograms
        else:
            assert metric_name in reg._counters

    # Record some values
    reg.counter("request_count").inc(1, labels={"method": "GET", "path": "/health"})
    reg.counter("tokens_in").inc(42, labels={"model": "gemma-3-lora-v1"})
    reg.histogram("generation_latency").observe(0.35, labels={"model": "gemma-3-lora-v1"})

    exposition = reg.generate_exposition()
    assert "# TYPE request_count counter" in exposition
    assert 'request_count{method="GET",path="/health"} 1.0' in exposition
    assert "# TYPE tokens_in counter" in exposition
    assert 'tokens_in{model="gemma-3-lora-v1"} 42.0' in exposition
    assert "# TYPE generation_latency histogram" in exposition
    assert "generation_latency_count" in exposition
    assert "generation_latency_sum" in exposition


def test_tracer_span_lifecycle() -> None:
    local_tracer = LLMOpsTracer(service_name="test-service")

    with local_tracer.start_span("retrieval.search", attributes={"top_k": 5}) as span:
        assert isinstance(span, Span)
        assert span.name == "retrieval.search"
        assert span.attributes["top_k"] == 5
        assert span.status == "ok"
        assert span.span_id is not None
        assert span.trace_id is not None

    spans = local_tracer.get_completed_spans()
    assert len(spans) == 1
    assert spans[0].name == "retrieval.search"
    assert spans[0].end_time is not None


def test_tracer_span_error_handling() -> None:
    local_tracer = LLMOpsTracer(service_name="test-service")

    with pytest.raises(RuntimeError), local_tracer.start_span("failing.step"):
        raise RuntimeError("Database connection timed out")

    spans = local_tracer.get_completed_spans()
    assert len(spans) == 1
    assert spans[0].status == "error"
    assert "Database connection timed out" in (spans[0].error_message or "")


def test_sensitive_text_redaction() -> None:
    raw_text = (
        "Received Authorization: Bearer eyJhbGciOiJIUzI1NiJ9 and "
        "api_key = 'sk-12345678901234567890123' with password='secretpassword'"
    )
    redacted = redact_sensitive_text(raw_text)
    assert "Bearer [REDACTED]" in redacted or "[REDACTED]" in redacted
    assert "secretpassword" not in redacted
    assert "sk-12345678901234567890123" not in redacted


def test_structured_json_formatter_section43() -> None:
    formatter = StructuredJsonFormatter(service_name="llmops-api", environment="test")
    record = logging.LogRecord(
        name="llmops.api",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Processed user request successfully with api_key=sk-secretsecretsecret123456",
        args=(),
        exc_info=None,
    )
    # Attach mandatory Section 43 attributes
    record.request_id = "req-12345"
    record.trace_id = "trace-67890"
    record.model_version = "gemma-3-lora-v1"
    record.latency = 0.045
    record.status = 200

    formatted_str = formatter.format(record)
    log_data = json.loads(formatted_str)

    assert log_data["level"] == "INFO"
    assert log_data["service"] == "llmops-api"
    assert log_data["environment"] == "test"
    assert log_data["request_id"] == "req-12345"
    assert log_data["trace_id"] == "trace-67890"
    assert log_data["model_version"] == "gemma-3-lora-v1"
    assert log_data["latency"] == 0.045
    assert log_data["status"] == 200
    assert "sk-secretsecretsecret123456" not in log_data["message"]
    assert "[REDACTED]" in log_data["message"]


def test_prometheus_metrics_endpoint() -> None:
    # Trigger metric update
    registry.counter("request_count").inc(
        1, labels={"method": "GET", "path": "/health", "status": "200"}
    )

    resp = prometheus_metrics()
    assert resp.status_code == 200
    body = bytes(resp.body).decode("utf-8")
    assert "# TYPE request_count counter" in body
    assert "request_count" in body
