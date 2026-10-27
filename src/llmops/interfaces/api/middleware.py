"""API middleware enforcing security headers, request size limits, and tracing (Section 28)."""

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from llmops.infrastructure.observability.metrics import registry
from llmops.infrastructure.observability.tracer import tracer


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Enforces production HTTP security headers (Section 28)."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        path = request.url.path.rstrip("/")
        if path in ("", "/docs", "/redoc") or request.url.path == "/openapi.json":
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "img-src 'self' data: https://fastapi.tiangolo.com; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "connect-src 'self'"
            )
        else:
            response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Guards against denial-of-service by rejecting oversized payloads (Section 28)."""

    def __init__(self, app: Any, max_size_bytes: int = 2 * 1024 * 1024) -> None:
        super().__init__(app)
        self.max_size_bytes = max_size_bytes

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                length = int(content_length)
                if length > self.max_size_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "PayloadTooLarge",
                            "message": (
                                f"Request body exceeds maximum allowed size "
                                f"({self.max_size_bytes} bytes)"
                            ),
                        },
                    )
            except ValueError:
                pass
        return await call_next(request)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Injects unique request ID, trace ID, and records Section 42 metrics and spans."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:12]}"
        trace_id = request.headers.get("X-Trace-ID") or uuid.uuid4().hex
        start_time = time.perf_counter()

        with tracer.start_span(
            "api.request",
            trace_id=trace_id,
            attributes={"http.method": request.method, "http.target": request.url.path},
        ):
            response = await call_next(request)

        duration_sec = time.perf_counter() - start_time
        duration_ms = duration_sec * 1000.0

        # Record Section 42 Prometheus metrics
        method = request.method
        path = request.url.path
        status_code = str(response.status_code)

        registry.counter("request_count").inc(
            labels={"method": method, "path": path, "status": status_code}
        )
        registry.histogram("request_latency").observe(
            duration_sec, labels={"method": method, "path": path}
        )
        if response.status_code >= 400:
            registry.counter("error_count").inc(
                labels={"method": method, "path": path, "status": status_code}
            )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
        return response
