# Antigravity LLMOps Platform --- Implementation Plan

**Master Specification Source of Truth:** [`antigravity_llmops_master_build_spec.md`](antigravity_llmops_master_build_spec.md)  
**Status:** In Progress  
**Current Milestone:** Milestone 6 (Data Curation & LLM-as-a-Judge)

---

## 1. Project Baseline & Gap Analysis

### Current State
- Repository initialized on `main`.
- Master specification in [`antigravity_llmops_master_build_spec.md`](antigravity_llmops_master_build_spec.md).
- Python 3.13 environment configured with uv, ruff, mypy, pytest, pydantic v2, httpx, beautifulsoup4, boto3.
- Milestone 0 completed: repository skeleton, `pyproject.toml`, quality gates (`make check`, `scripts/check.ps1`), CI workflow, ADR 001 and ADR 002.
- Milestone 1 completed: configuration-driven crawler with SSRF safety, robots.txt parsing, SHA-256 raw snapshots, crawl manifests, canonical document models, unit and contract tests.
- Milestone 2 completed: text normalization, HTML/Markdown parsing with code-block preservation, heading-aware semantic chunking with provenance, exact (SHA-256) and near-duplicate (Jaccard) deduplication, dataset quality reporting.
- Milestone 3 completed: `BaseStorage` port, `LocalStorageAdapter`, `S3StorageAdapter` with MinIO/S3 support, storage factory, `DatasetVersionManifest`, DVC pipeline (`dvc.yaml`, `params.yaml`), ADR 003, storage contract tests.
- Milestone 4 completed: `BaseVectorStore`, `BaseEmbeddingModel`, `BaseReranker` ports, `QdrantVectorStore` with Qdrant REST API + Section 6 schema, `InMemoryVectorStore`, `SentenceTransformerEmbeddings`, `CrossEncoderReranker`, two-stage `RetrievalService`, IR metrics (Recall@K, Precision@K, MRR, nDCG), contract and unit tests.
- Milestone 5 completed: `BaseLLMClient` port, `OllamaClient`, `VLLMClient`, `MockLLMClient`, versioned prompt management (`developer_assistant`, `rag_grounded_answer`), `RAGService` with citations and insufficient context detection, ADR 004, unit and contract tests (66 tests passing, 81% coverage).

---

## 2. Milestone Roadmap

### Milestone 0: Repository Foundation [COMPLETED]
- **Goal:** Establish production-grade Python package structure, development tools, linting/typing/testing gates, and baseline CI.
- **Status:** Done (26 files, tests passing, 0 lint/type issues).

### Milestone 1: Source Acquisition [COMPLETED]
- **Goal:** Configuration-driven documentation acquisition for FastAPI, Pydantic, and SQLAlchemy with domain allowlists, robots respect, SSRF safety, rate-limiting, and raw snapshot manifests.
- **Status:** Done (Crawler service, SSRF guard, manifest contract, canonical document model, 25 tests passing).

### Milestone 2: Processing (Parsing, Normalization, Chunking, Deduplication) [COMPLETED]
- **Goal:** Transform raw snapshots into canonical cleaned documents and semantic chunks with deduplication and quality reporting.
- **Status:** Done (HTML parser with code fences, semantic chunker with provenance, deduplicator, quality reporter, integration pipeline, 40 tests passing).

### Milestone 3: Dataset Versioning & Object Storage (DVC & S3/MinIO) [COMPLETED]
- **Goal:** Abstract object storage interface, local/S3 adapters, and DVC pipeline for immutable versioned datasets.
- **Status:** Done (BaseStorage port, Local & S3 adapters, DatasetVersionManifest, dvc.yaml, params.yaml, ADR 003, contract tests, 46 tests passing).

### Milestone 4: Vector Retrieval Subsystem (Embeddings, Qdrant, Reranker) [COMPLETED]
- **Goal:** Production-grade semantic search with Qdrant vector store, configurable embedding models, cross-encoder reranking, and retrieval metrics.
- **Status:** Done (Qdrant & in-memory vector stores, SentenceTransformer embeddings, CrossEncoder reranker, RetrievalService, IR metrics, 61 tests passing).

### Milestone 5: Local Generation & RAG Subsystem [COMPLETED]
- **Goal:** Integrate local LLM runtime (Ollama/vLLM) for grounded question-answering with citations and prompt versioning.
- **Status:** Done (BaseLLMClient, Ollama & vLLM clients, PromptManager, versioned prompts, RAGService with citations and fallback, ADR 004, 66 tests passing).

### Milestone 6: Data Curation & LLM-as-a-Judge [CURRENT]
- **Goal:** Transform raw snapshots into canonical cleaned documents and semantic chunks with deduplication and quality reporting.
- **Tasks:**
  - HTML/Markdown parser converting raw crawl into canonical `Document` instances.
  - Text normalizers (stripping boilerplate, code block preservation, anchor cleanup).
  - Document chunker supporting semantic/heading-aware chunking with token counting and chunk metadata provenance.
  - Exact deduplication (SHA-256 content hashes) and MinHash/LSH near-duplicate detection.
  - Data quality reporter outputting token statistics, section counts, duplicate counts into `data/interim/` and `data/processed/`.
  - Comprehensive unit tests.
- **Acceptance Criteria:** Deterministic processing pipeline converts raw sources into deduplicated chunks with full provenance and generates a data quality report artifact.

### Milestone 3: Dataset Versioning & Object Storage (DVC & S3/MinIO)
- **Goal:** Abstract object storage interface and DVC pipeline for immutable versioned datasets.
- **Tasks:**
  - Storage adapter (`src/llmops/infrastructure/storage/`) supporting local filesystem and S3/MinIO via unified contract.
  - Configure `dvc.yaml` and `params.yaml` tracking stages (`acquire`, `parse`, `chunk`, `dedup`).
  - Implement metadata tracking tying dataset versions to commit hash, config hash, and storage URIs.
- **Acceptance Criteria:** DVC pipeline reproduces artifacts deterministically; data is decoupled from git; dataset manifests record version provenance.

### Milestone 4: Vector Retrieval Subsystem (Embeddings, Qdrant, Reranker)
- **Goal:** Production-grade semantic search with Qdrant vector store, configurable embedding models, cross-encoder reranking, and retrieval metrics.
- **Tasks:**
  - Embedding adapter abstraction (`src/llmops/infrastructure/embeddings/`) with sentence-transformers implementation and mock/test provider.
  - Qdrant adapter (`src/llmops/infrastructure/vector/`) supporting payload schema (document_id, chunk_id, provenance, hash).
  - Cross-encoder reranker adapter (`src/llmops/infrastructure/reranker/`).
  - Retrieval service implementing Top-N Qdrant search -> metadata filtering -> Top-K cross-encoder reranking.
  - Retrieval evaluation suite calculating Recall@K, MRR, NDCG.
- **Acceptance Criteria:** Chunk indexing into Qdrant, top-N retrieval with metadata filtering, reranking verified by integration and contract tests.

### Milestone 5: Local Generation & RAG Subsystem
- **Goal:** Integrate local LLM runtime (Ollama/OpenAI-compatible) for grounded question-answering with citations.
- **Tasks:**
  - LLM client abstraction (`src/llmops/infrastructure/llm/`) with Ollama and OpenAI-compatible adapters.
  - Prompt management (`prompts/rag/`, `prompts/system/`) with prompt versioning.
  - RAG service chaining retrieval -> context assembly -> prompt template -> LLM generation -> citation extraction.
  - Grounding validation: return structured answer with citations and unanswerable fallback if retrieval evidence is insufficient.
- **Acceptance Criteria:** End-to-end RAG query returns accurate answer citing retrieved source chunks, with fallback on insufficient context.

### Milestone 6: Data Curation & LLM-as-a-Judge
- **Goal:** Synthetic instruction/Q&A generation and automated quality curation with structured LLM judging.
- **Tasks:**
  - Synthetic Q&A generator using canonical chunks and generation prompts (`prompts/generation/`).
  - LLM-as-a-Judge evaluator (`prompts/judge/`) scoring relevance, correctness, groundedness, clarity on 1-5 scale.
  - Pydantic models for structured judge decisions (`pass`/`fail`, scores, reason).
  - Train/Val/Test splitter with strict leakage protection (deterministic exact/near duplicate detection and source boundary splits).
  - Output curated dataset in standard messages format (`{"messages": [...]}`).
- **Acceptance Criteria:** Synthetic candidate generation, strict quality threshold gating, leakage validation, and export of versioned JSONL datasets.

### Milestone 7: Gemma 3 Parameter-Efficient Fine-Tuning (LoRA / PEFT)
- **Goal:** Reproducible fine-tuning pipeline for Gemma 3 using Hugging Face PEFT/LoRA, Unsloth optimization, and MLflow experiment tracking.
- **Tasks:**
  - Training configuration schema (`configs/training.yaml`, `params.yaml`).
  - Training module (`src/llmops/domain/training/`, `src/llmops/infrastructure/training/`) supporting CPU smoke profile and CUDA GPU profile.
  - Adapter checkpoint export with metadata manifest (`training_config_hash`, base model revision, LoRA alpha/rank/dropout).
  - MLflow logging for parameters, train/val loss curves, and artifact logging.
- **Acceptance Criteria:** Training runs end-to-end under smoke mode (CPU/synthetic) and full mode (GPU/Gemma 3), producing adapter artifacts logged to MLflow.

### Milestone 8: Multi-Layer Evaluation & Quality Gate
- **Goal:** Comprehensive evaluation comparing Base Gemma 3 vs Fine-Tuned Gemma 3 vs Fine-Tuned + RAG, with deterministic quality gates.
- **Tasks:**
  - Evaluation harness (`src/llmops/domain/evaluation/`) running held-out benchmark dataset.
  - Layer A (deterministic syntax/schema checks), Layer B (retrieval metrics), Layer C (generation metrics), Layer D (model comparison), Layer E (RAG comparison).
  - Automated quality gate evaluating candidate model against baseline thresholds; outputs machine-readable evaluation report.
- **Acceptance Criteria:** Machine-readable evaluation report generated; regression detection triggers quality gate pass/fail.

### Milestone 9: Production Serving & Inference API
- **Goal:** High-throughput FastAPI application with vLLM/Ollama model routing, Nginx reverse proxy, and health/readiness endpoints.
- **Tasks:**
  - FastAPI application (`src/llmops/interfaces/api/`) implementing `/health`, `/ready`, `/version`, `/v1/chat`, `/v1/generate`, `/v1/retrieve`, `/v1/models`, `/v1/feedback`.
  - Model registry & promotion status integration.
  - Nginx configuration with TLS termination, rate-limiting, and security headers.
  - Dockerfiles for API, Worker, Inference, Training.
- **Acceptance Criteria:** FastAPI serves chat and retrieval endpoints; health and readiness reflect dependency states; Nginx routes requests properly.

### Milestone 10: Full-Stack Observability
- **Goal:** Complete monitoring with OpenTelemetry distributed tracing, Prometheus metrics, structured JSON logging, and Grafana dashboards.
- **Tasks:**
  - OpenTelemetry instrumentation for API requests, retrieval spans, and LLM calls.
  - Prometheus metrics exporter (request counts, latencies, tokens in/out, retrieval stats).
  - Structured JSON logging with request_id, trace_id, redaction of sensitive tokens.
  - Grafana dashboard definitions (`observability/grafana/dashboards/`) for API, Model, Retrieval, and Pipelines.
- **Acceptance Criteria:** Metrics endpoint `/metrics` exposes Prometheus counters/histograms; traces emitted to OTel collector; logs formatted as structured JSON.

### Milestone 11: Kubernetes & Helm Packaging
- **Goal:** Cloud-portable Kubernetes manifests and production-ready Helm charts.
- **Tasks:**
  - Helm charts (`deployment/helm/llmops/`) with modular subcharts for API, inference, and pipeline.
  - Environment values files (`values-dev.yaml`, `values-staging.yaml`, `values-prod.yaml`).
  - HPA, PodDisruptionBudgets, resource requests/limits, liveness/readiness probes, and ingress rules.
- **Acceptance Criteria:** `helm lint` and dry-run template render valid Kubernetes manifests conforming to best practices.

### Milestone 12: GitOps with Argo CD
- **Goal:** Automated, declarative deployment workflows and promotion using GitOps.
- **Tasks:**
  - Argo CD Application and ApplicationSet manifests (`deployment/argocd/`).
  - Release promotion pipeline updating environment-specific image tags and model versions.
  - Documented automated rollback procedures.
- **Acceptance Criteria:** Argo CD manifests correctly reference environment Helm values; automated deployment and rollback workflows defined.

### Milestone 13: Infrastructure as Code (Terraform)
- **Goal:** Cloud-portable infrastructure definitions provisioning compute, storage, networking, and security.
- **Tasks:**
  - Terraform modules (`deployment/terraform/modules/`) for object storage, Kubernetes cluster, VPC/networking, and database.
  - Environment root modules (`environments/dev`, `environments/prod`).
  - Localstack / MinIO dev compatibility path.
- **Acceptance Criteria:** `terraform validate` and `terraform plan` execute without syntax or structural errors.

### Milestone 14: Continuous LLMOps & Workflow Orchestration
- **Goal:** Prefect orchestration for automated data acquisition, evaluation, quality gating, model promotion, and monitoring feedback loop.
- **Tasks:**
  - Prefect flows (`pipelines/`) orchestrating: `ingest_flow`, `curate_flow`, `train_flow`, `eval_flow`, `promote_flow`.
  - Promotion policy: automated retraining triggered only when data drift or quality thresholds warrant it.
  - End-to-end acceptance test covering the entire lifecycle from crawl to promoted model serving.
- **Acceptance Criteria:** End-to-end integration test passes validating all 21 verification criteria from Section 78 of the master spec.

---

## 3. Immediate Focus: Milestone 0 Work Plan
1. Project layout directory creation:
   - `src/llmops/{domain,application,infrastructure,interfaces}`
   - `configs/`, `prompts/`, `data/{raw,interim,processed,evaluation,external}`
   - `tests/{unit,integration,contract,evaluation}`
   - `docs/{architecture,decisions,runbooks,api,development}`
   - `docker/`, `deployment/`, `observability/`, `nginx/`, `.github/workflows/`
2. `pyproject.toml` with:
   - Package metadata (`llmops`)
   - Dependencies: `fastapi`, `pydantic`, `uvicorn`, `sqlalchemy`, `psycopg`, `pydantic-settings`, `pyyaml`, `requests`, `httpx`
   - Dependency groups: `dev` (ruff, mypy, pytest, pytest-cov, pre-commit), `ml` (torch, transformers, sentence-transformers, qdrant-client), `observability` (prometheus-client, opentelemetry-api)
   - Tool settings for Ruff (line-length 100, target-version py313, lint rules), mypy (strict mode, py313), pytest (testpaths, cov config)
3. Configuration files:
   - `.gitignore`, `.pre-commit-config.yaml`, `.env.example`, `Makefile`, `scripts/check.ps1`
   - Initial application configs in `configs/application.yaml`
4. Core ADRs:
   - `docs/decisions/001-python-runtime.md`
   - `docs/decisions/002-clean-architecture.md`
5. Foundation code:
   - `src/llmops/__init__.py` (version declaration)
   - `src/llmops/domain/` entities and base exceptions
   - Basic CLI entrypoint (`src/llmops/interfaces/cli.py`)
6. Quality Gate Verification:
   - Unit tests under `tests/unit/`
   - Run `ruff check`, `ruff format --check`, `mypy`, `pytest`
