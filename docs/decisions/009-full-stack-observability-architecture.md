# ADR 009: Full-Stack Observability Architecture (OpenTelemetry, Prometheus, Grafana, Loki)

## Status
Accepted

## Context
Production LLMOps systems require comprehensive visibility across distributed API gateways, semantic vector retrieval engines, and LLM generative models (Section 42-44). Traditional APM tooling often misses domain-specific LLM telemetry (e.g. token velocity, embedding latency, judge curation pass rates, vector search top-K distributions). Furthermore, Section 43 mandates strict privacy guardrails: logs must be structured in JSON format with contextual trace and request IDs, while actively redacting API credentials, authentication tokens, and preventing bulk raw prompt/response dumping.

## Decision
1. **OpenTelemetry for Distributed Tracing**:
   - Implement `LLMOpsTracer` creating contextual trace spans across:
     - `api.request`: HTTP gateway invocations and security checks.
     - `retrieval.embed`: Query embedding vector calculations.
     - `retrieval.search`: Vector database similarity searches.
     - `retrieval.rerank`: Cross-encoder second-stage reranking.
     - `llm.generate`: Foundation model token generation.
2. **Prometheus Metrics Registry**:
   - Track all Section 42 metrics via zero-overhead, thread-safe metric primitives (`Counter`, `Histogram`):
     - `request_count`, `request_latency`, `error_count`
     - `retrieval_latency`, `embedding_latency`, `generation_latency`
     - `tokens_in`, `tokens_out`, `model_requests`, `cache_hits`
   - Expose native Prometheus 0.0.4 exposition text format at `/metrics`.
3. **Structured JSON Logging with Configurable Redaction**:
   - Implement `StructuredJsonFormatter` appending `request_id`, `trace_id`, `model_version`, `environment`, `latency`, and `status`.
   - Apply automated regex redaction to strip API keys, Bearer tokens, and secrets.
   - Omit raw generation payloads from default application log records.
4. **Grafana Dashboards**:
   - Provide pre-configured production JSON dashboards for:
     - API Gateway throughput, latency percentiles (p50/p95/p99), and error rates.
     - Retrieval & Vector store latency, top-K distributions, and empty retrieval rates.
     - Model inference latencies, token consumption velocity, and active version status.
     - Continuous pipeline duration, dataset split metrics, and judge evaluation rates.

## Consequences
- Full compliance with Sections 42, 43, and 44.
- Production-grade observability stack integrated with Docker Compose, Prometheus scraping, and Grafana provisioning.
- Privacy compliance with zero sensitive credential leaks into logs or metric label sets.
