# Antigravity LLMOps Platform --- Implementation Plan

**Master Specification Source of Truth:** [`antigravity_llmops_master_build_spec.md`](antigravity_llmops_master_build_spec.md)  
**Status:** In Progress  
**Current Milestone:** Milestone 8 (Multi-Layer Evaluation & Quality Gate)

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
- Milestone 5 completed: `BaseLLMClient` port, `OllamaClient`, `VLLMClient`, `MockLLMClient`, versioned prompt management (`developer_assistant`, `rag_grounded_answer`), `RAGService` with citations and insufficient context detection, ADR 004, unit and contract tests.
- Milestone 6 completed: synthetic developer Q&A generator, structured `DataCurationJudge` with multi-dimensional rubric, `DatasetSplitter` ensuring zero document-level data leakage, train/val/test export with external audit metadata isolation (`*_metadata.jsonl`), statistical dataset summary, ADR 005.
- Milestone 7 completed: Gemma 3 parameter-efficient fine-tuning (LoRA / PEFT), `BaseTrainer` port with profile-driven execution (`smoke` for CPU/CI, `full` for CUDA GPU), `BaseExperimentTracker` port with `LocalExperimentTracker` and `MLflowTracker`, Section 23 machine-readable `ModelManifest` traceability contract, ADR 006 (98 tests passing, 81% overall coverage).

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

### Milestone 6: Data Curation & LLM-as-a-Judge [COMPLETED]
- **Goal:** Synthetic instruction/Q&A generation and automated quality curation with structured LLM judging.
- **Status:** Done (Synthetic generator, DataCurationJudge, DatasetSplitter with zero document leakage, external audit metadata files, ADR 005, 87 tests passing).

### Milestone 7: Gemma 3 Parameter-Efficient Fine-Tuning (LoRA / PEFT) [COMPLETED]
- **Goal:** Reproducible fine-tuning pipeline for Gemma 3 using Hugging Face PEFT/LoRA, profile-driven CPU/GPU paths, MLflow tracking, and Section 23 ModelManifest traceability contract.
- **Status:** Done (TrainingConfig, LoRATrainer, LocalExperimentTracker, MLflowTracker, TrainModelUseCase, ModelManifest, ADR 006, 98 tests passing).

### Milestone 8: Multi-Layer Evaluation & Quality Gate [CURRENT]
- **Goal:** Comprehensive evaluation comparing Base Gemma 3 vs Fine-Tuned Gemma 3 vs Fine-Tuned + RAG, with deterministic quality gates.
- **Tasks:**
  - Evaluation harness (`src/llmops/domain/evaluation/`) running held-out benchmark dataset.
  - Multi-layer evaluation:
    - Layer A: deterministic syntax/schema checks.
    - Layer B: retrieval metrics (Recall@K, MRR, nDCG).
    - Layer C: generation metrics (accuracy, formatting, latency).
    - Layer D: model comparison (Base Gemma 3 vs Fine-Tuned Gemma 3).
    - Layer E: system comparison (Fine-Tuned model vs Fine-Tuned + RAG).
  - Automated quality gate evaluating candidate model against baseline thresholds; outputs machine-readable `evaluation_report.json`.
  - Promotion logic (`llmops promote`) setting `approved: true` in `ModelManifest`.
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
