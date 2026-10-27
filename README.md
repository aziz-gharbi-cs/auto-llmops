# Antigravity LLMOps Platform

> Production-Grade, Documentation-Grounded Developer Assistance System built with Gemma 3, LoRA/PEFT, RAG, and End-to-End LLMOps Engineering.

[![CI](https://github.com/example/llmops/actions/workflows/ci.yml/badge.svg)](https://github.com/example/llmops/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![uv](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy-lang.org/)

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
Gemma 3 LoRA Fine-Tuning → MLflow Tracking → Multi-Layer Evaluation
     ↓
Quality Gate → Model Registry → vLLM / Ollama Serving → FastAPI + Nginx
     ↓
OpenTelemetry + Prometheus + Grafana Observability
```

---

## 2. Architecture & Design Principles

The system adheres to **Clean Architecture** with explicit boundaries:
- **`src/llmops/domain/`**: Pure entities, domain contracts, and business rules. Zero external library dependencies.
- **`src/llmops/application/`**: Use cases coordinating workflows and domain ports.
- **`src/llmops/infrastructure/`**: Adapters for vector search (Qdrant), storage (MinIO/S3), inference (Ollama/vLLM), embeddings, and databases.
- **`src/llmops/interfaces/`**: Delivery interfaces (FastAPI REST API, Click CLI, pipeline runners).

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
| **Workflow Orchestration** | Prefect |
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
make check

# Or on Windows PowerShell:
.\scripts\check.ps1
```

---

## 5. CLI Commands

```bash
llmops --help
llmops --version
llmops ingest     # Crawl documentation sources into raw snapshots
llmops validate   # Run deterministic dataset validation
llmops embed      # Generate embeddings and populate Qdrant index
llmops curate     # Generate synthetic Q&A and filter via LLM judge
llmops train      # Execute Gemma 3 LoRA fine-tuning
llmops evaluate   # Run multi-layer regression evaluation
llmops promote    # Promote candidate model through quality gate
llmops serve       # Start the FastAPI application server
```

---

## 6. Implementation Progress

See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for milestone roadmap and current status.
