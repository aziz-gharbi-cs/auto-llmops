# Antigravity LLMOps Platform

> Production-Grade, Documentation-Grounded Developer Assistance System built with Gemma 3, LoRA/PEFT, RAG, and End-to-End LLMOps Engineering.

[![CI](https://github.com/example/llmops/actions/workflows/ci.yml/badge.svg)](https://github.com/example/llmops/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![uv](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-155%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-84%25-brightgreen.svg)]()

---

## 1. Overview & Objective

This platform delivers high-accuracy, grounded developer assistance for modern Python frameworks (**FastAPI**, **Pydantic**, and **SQLAlchemy**) using official documentation as the authoritative knowledge source. 

Rather than a basic fine-tuning script, it represents a complete, reproducible **LLMOps platform** covering the entire lifecycle:

```text
Official Docs → Crawl → Canonical Docs → DVC Versioning
     ↓
Embedding & Qdrant Index → Reranker → Grounded RAG
     ↓
Synthetic Generation → LLM-as-a-Judge Curated Dataset
     ↓
Gemma 3 LoRA Fine-Tuning → MLflow Tracking → Multi-Layer Evaluation (Layers A–E)
     ↓
Quality Gate → Model Registry → vLLM / Ollama Serving → FastAPI + Nginx
     ↓
OpenTelemetry + Prometheus + Grafana Observability
     ↓
Helm Packaging → Argo CD GitOps → Multi-Cloud Terraform IaC
```

---

## 2. Architecture & Design Principles

The system adheres to **Clean Architecture** with explicit boundaries:
- **`src/llmops/domain/`**: Pure entities, domain contracts, and business rules. Zero external library dependencies.
- **`src/llmops/application/`**: Use cases coordinating workflows and domain ports.
- **`src/llmops/infrastructure/`**: Adapters for vector search (Qdrant), storage (MinIO/S3), inference (Ollama/vLLM), embeddings, training (LoRA/PEFT), tracking (MLflow), and observability (Prometheus/OTel).
- **`src/llmops/interfaces/`**: Delivery interfaces (FastAPI REST API, Click CLI, continuous pipeline runner).

All infrastructure adapters satisfy abstract interfaces, ensuring interchangeable providers without touching business logic.

---

## 3. Tech Stack

| Subsystem | Technology |
|---|---|
| **Language & Tooling** | Python 3.13, `uv`, Ruff, mypy (strict), pytest |
| **API & Gateway** | FastAPI, Pydantic v2, Uvicorn, Nginx |
| **Metadata Database** | PostgreSQL, SQLAlchemy 2.x, psycopg |
| **Vector Search & Reranking** | Qdrant, Sentence-Transformers, Cross-Encoder Reranker |
| **Model Runtime** | Ollama (Local Dev) / vLLM (Production GPU) |
| **Base Model & Fine-Tuning** | Gemma 3, Unsloth, Hugging Face PEFT / LoRA |
| **Experiment Tracking** | MLflow |
| **Workflow Orchestration** | Continuous LLMOps Pipeline Engine (`continuous_pipeline.py`) |
| **Data Versioning** | DVC + MinIO / S3 |
| **Observability** | OpenTelemetry, Prometheus, Grafana, Loki |
| **Deployment & GitOps** | Docker, Helm, Kubernetes, Argo CD, Terraform |

---

## 4. Quickstart

### Prerequisites
- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) installed
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/example/llmops.git
cd llmops

# Install dependencies using uv
uv sync --all-groups

# Copy sample configuration
cp .env.example .env
```

### Running Quality Checks
```bash
# Full local quality gate (lint + typecheck + tests)
powershell -ExecutionPolicy Bypass -File .\scripts\check.ps1

# Or with Ruff and Mypy directly:
python -m ruff check src tests
python -m mypy src tests
python -m pytest tests/
```

### Section 78 Master Acceptance Verification
```bash
python -m pytest tests/acceptance/test_section78_lifecycle.py -v
```

---

## 5. CLI Reference

```bash
llmops --help
llmops --version
llmops ingest       # Crawl documentation sources into raw snapshots
llmops validate     # Run deterministic dataset validation
llmops embed        # Generate embeddings and populate Qdrant index
llmops curate       # Generate synthetic Q&A and filter via LLM judge
llmops train        # Execute Gemma 3 LoRA fine-tuning
llmops evaluate     # Run multi-layer regression evaluation (Layers A-E)
llmops promote      # Promote candidate model through quality gate
llmops serve        # Start the FastAPI application server
llmops pipeline     # Run continuous end-to-end LLMOps lifecycle
```

---

## 6. Architecture Decision Records (ADRs)

Key architectural decisions are documented in [`docs/decisions/`](docs/decisions/):
- **ADR 001**: Python Runtime and Toolchain Strategy (`001-python-runtime.md`)
- **ADR 002**: Clean Architecture Layering and Domain Boundaries (`002-clean-architecture.md`)
- **ADR 003**: Object Storage Port and DVC Dataset Versioning (`003-storage-and-dvc.md`)
- **ADR 004**: Interchangeable LLM Clients and Prompt Versioning (`004-ollama-and-vllm.md`)
- **ADR 005**: Data Curation with LLM-as-a-Judge and Leakage-Safe Splitting (`005-data-curation-and-judge.md`)
- **ADR 006**: Gemma 3 LoRA Fine-Tuning and Model Manifest Traceability (`006-lora-training-and-traceability.md`)
- **ADR 007**: Multi-Layer Evaluation and Deterministic Quality Gate (`007-evaluation-and-quality-gates.md`)
- **ADR 008**: API Gateway, Container Architecture, and Reverse Proxy (`008-api-gateway-and-container-architecture.md`)
- **ADR 009**: Full-Stack Observability Architecture (`009-full-stack-observability-architecture.md`)
- **ADR 010**: Kubernetes and Helm Packaging Architecture (`010-kubernetes-and-helm-packaging.md`)
- **ADR 011**: GitOps and Continuous Delivery with Argo CD (`011-gitops-and-continuous-delivery.md`)
- **ADR 012**: Infrastructure as Code Architecture (Terraform) (`012-infrastructure-as-code-terraform.md`)
- **ADR 013**: Continuous LLMOps Lifecycle and Acceptance Verification (`013-continuous-llmops-and-acceptance-verification.md`)

---

## 7. Disaster Recovery & Rollback

Operational procedures for declarative rollbacks and disaster recovery are detailed in [`docs/runbooks/rollback.md`](docs/runbooks/rollback.md).
