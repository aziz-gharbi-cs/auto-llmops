"""Structured JSON logging with configurable redaction and contextual metadata (Section 43)."""

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

# Sensitive patterns to redact: API keys, tokens, passwords, bearer headers
SENSITIVE_PATTERNS = [
    re.compile(
        r"(?i)(api[_-]?key|secret|token|password|authorization)\s*[:=]\s*['\"]?([^'\"\s,]+)"
    ),
    re.compile(r"Bearer\s+([A-Za-z0-9_\-\.+=/]+)"),
    re.compile(r"(sk-[A-Za-z0-9]{20,})"),
]


def redact_sensitive_text(text: str) -> str:
    """Mask credentials and sensitive tokens with '[REDACTED]'."""
    redacted = text
    for pattern in SENSITIVE_PATTERNS:
        redacted = pattern.sub(r"\1: [REDACTED]", redacted)
    return redacted


class StructuredJsonFormatter(logging.Formatter):
    """Outputs standardized structured JSON log events conforming to Section 43."""

    def __init__(
        self,
        service_name: str = "llmops-api",
        environment: str = "production",
        redact: bool = True,
        dump_payloads: bool = False,
    ) -> None:
        super().__init__()
        self.service_name = service_name
        self.environment = environment
        self.redact = redact
        self.dump_payloads = dump_payloads

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a structured JSON object."""
        # 1. Base log record fields
        timestamp = datetime.fromtimestamp(record.created, tz=UTC).isoformat()
        raw_message = record.getMessage()
        message = redact_sensitive_text(raw_message) if self.redact else raw_message

        # 2. Extract contextual attributes per Section 43
        data: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "service": self.service_name,
            "environment": getattr(record, "environment", self.environment),
            "message": message,
            # Mandatory Section 43 request attributes
            "request_id": getattr(record, "request_id", None),
            "trace_id": getattr(record, "trace_id", None),
            "model_version": getattr(record, "model_version", None),
            "latency": getattr(record, "latency", None),
            "status": getattr(record, "status", None),
        }

        # 3. Extra structured fields attached to record.__dict__
        standard_keys = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "environment",
            "request_id",
            "trace_id",
            "model_version",
            "latency",
            "status",
        }

        extras = {k: v for k, v in record.__dict__.items() if k not in standard_keys}
        if extras:
            if not self.dump_payloads:
                # Do not dump full prompts or completions
                extras = {
                    k: v for k, v in extras.items() if k not in ("prompt", "completion", "body")
                }
            data["extra"] = extras

        # 4. Exception details if present
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)

        return json.dumps(data)


def configure_structured_logging(
    level: int = logging.INFO,
    service_name: str = "llmops-api",
    environment: str = "production",
    redact: bool = True,
    dump_payloads: bool = False,
) -> logging.Handler:
    """Configure root logger to use StructuredJsonFormatter."""
    handler = logging.StreamHandler()
    formatter = StructuredJsonFormatter(
        service_name=service_name,
        environment=environment,
        redact=redact,
        dump_payloads=dump_payloads,
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    return handler
