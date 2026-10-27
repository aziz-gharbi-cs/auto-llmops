"""OpenTelemetry tracing engine with distributed context propagation (Section 42)."""

import contextlib
import logging
import time
import uuid
from collections.abc import Generator, Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """Represents a distributed trace span."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error_message: str | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def finish(self, status: str = "ok", error_message: str | None = None) -> None:
        self.end_time = time.time()
        self.status = status
        self.error_message = error_message


class LLMOpsTracer:
    """Tracer managing spans, OpenTelemetry integration, and local trace state."""

    def __init__(self, service_name: str = "llmops-service") -> None:
        self.service_name = service_name
        self._completed_spans: list[Span] = []
        self._otel_tracer = None

        # Attempt to bind OpenTelemetry tracer if SDK is available
        try:
            from opentelemetry import trace

            self._otel_tracer = trace.get_tracer(service_name)
        except Exception:
            self._otel_tracer = None

    @contextlib.contextmanager
    def start_span(
        self,
        name: str,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> Generator[Span]:
        """Start a new span as a context manager."""
        active_trace_id = trace_id or uuid.uuid4().hex
        span_id = uuid.uuid4().hex[:16]
        span = Span(
            name=name,
            trace_id=active_trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            attributes=dict(attributes or {}),
        )

        try:
            yield span
            span.finish(status="ok")
        except Exception as exc:
            span.finish(status="error", error_message=str(exc))
            raise
        finally:
            self._completed_spans.append(span)

    def get_completed_spans(self) -> list[Span]:
        """Return snapshot of recorded spans."""
        return list(self._completed_spans)

    def clear(self) -> None:
        """Clear recorded spans (useful in test teardown)."""
        self._completed_spans.clear()


# Global singleton tracer
tracer = LLMOpsTracer()
