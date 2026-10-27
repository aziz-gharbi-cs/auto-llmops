# Antigravity Master Build Specification --- Production-Grade LLMOps Platform

## 0. Mission

You are the principal engineer responsible for rebuilding this project
from scratch.

Do not treat this as a simple fine-tuning project. Build it as a
**complete, reproducible, production-oriented LLMOps platform**
demonstrating the full lifecycle:

``` text
Source acquisition
      ↓
Ingestion
      ↓
Validation / normalization
      ↓
Dataset versioning
      ↓
Data quality + LLM-as-a-Judge
      ↓
Curated training/evaluation datasets
      ↓
Embeddings / RAG index
      ↓
Baseline evaluation
      ↓
Gemma 3 + LoRA fine-tuning
      ↓
Experiment tracking
      ↓
Model evaluation
      ↓
Quality gate
      ↓
Model registry / promotion
      ↓
Containerized inference
      ↓
API gateway
      ↓
Observability
      ↓
Continuous evaluation / feedback
```

The goal is to produce a project that can be defended in a serious
technical interview as an LLMOps/MLOps/software-engineering system.

Do NOT blindly reproduce the old implementation. Preserve the core
project idea while fixing weak architectural decisions.

Do NOT ask the user to decide technologies that are explicitly decided
in this document. Make implementation decisions yourself and proceed.

If an implementation detail is genuinely unknown, choose a sensible
default, document it in `docs/decisions/`, and continue. Do not block
the project with unnecessary questions.

------------------------------------------------------------------------

# 1. Core Project Objective

Build an end-to-end LLMOps system around:

-   documentation-grounded developer assistance
-   official framework/library documentation as the primary knowledge
    source
-   RAG / semantic retrieval
-   high-quality synthetic instruction/Q&A generation
-   LLM-as-a-Judge data curation
-   Gemma 3 parameter-efficient fine-tuning
-   LoRA / PEFT
-   experiment tracking
-   dataset and artifact versioning
-   automated evaluation
-   model promotion
-   production-style inference
-   observability
-   CI/CD
-   reproducible deployment

The original project was associated with a PwC Tunisia internship and
involved:

-   Gemma 3
-   Hugging Face Transformers
-   Unsloth
-   LoRA / PEFT
-   DVC
-   MLflow
-   Docker
-   Ollama
-   Nginx
-   vector search
-   embeddings
-   LLM-as-a-Judge
-   GitHub Actions

The rebuilt version must be stronger and more coherent than the
original.

------------------------------------------------------------------------

# 2. The Domain Decision

The final domain is:

> **Documentation-grounded developer assistance using official
> documentation for selected frameworks/libraries.**

Do not use arbitrary internet data.

Do not use Stack Overflow as a default source.

Prefer official documentation and permissively usable technical
documentation.

Start with a deliberately small, controlled source set for the first
end-to-end implementation:

1.  FastAPI
2.  Pydantic
3.  SQLAlchemy

Design the ingestion system so additional documentation sources can
later be added without architectural changes.

The source list must be configuration-driven.

Example:

``` yaml
sources:
  - name: fastapi
    type: documentation
    base_url: ...
    allowed_domains: [...]
  - name: pydantic
    type: documentation
    base_url: ...
    allowed_domains: [...]
  - name: sqlalchemy
    type: documentation
    base_url: ...
    allowed_domains: [...]
```

Do not hard-code framework-specific behavior into generic ingestion
code.

------------------------------------------------------------------------

# 3. Final Technology Decisions

Use the following stack unless a concrete compatibility issue requires a
documented substitution.

## Language / runtime

-   Python 3.13
-   `uv` for Python dependency/environment management
-   `pyproject.toml` as the single Python project configuration
-   Ruff for linting/formatting
-   mypy for static typing
-   pytest for tests
-   pytest-cov for coverage
-   pre-commit for local quality gates

Do not use Poetry, Pipenv, or multiple competing Python environment
managers.

------------------------------------------------------------------------

# 4. Application Backend

Use:

-   FastAPI
-   Pydantic v2
-   Uvicorn for local execution
-   Gunicorn + Uvicorn workers only where appropriate for production API
    deployment

API responsibilities:

-   inference
-   RAG query
-   health/readiness endpoints
-   model/version information
-   retrieval diagnostics where authorized
-   evaluation administration endpoints if needed

The API must NOT contain:

-   training logic
-   dataset preprocessing logic
-   long-running embedding jobs
-   model fine-tuning logic

Separate application inference from ML pipelines.

------------------------------------------------------------------------

# 5. Database

Use:

-   PostgreSQL
-   SQLAlchemy 2.x
-   Alembic
-   psycopg

PostgreSQL stores application metadata, not vectors.

Store entities such as:

-   users
-   API keys or authentication metadata if implemented
-   documents
-   ingestion runs
-   dataset versions
-   evaluation runs
-   model versions
-   deployment records
-   pipeline runs
-   audit events
-   feedback
-   prompt versions

Use migrations.

Never modify production schema manually.

------------------------------------------------------------------------

# 6. Vector Database

Use:

-   Qdrant

Do not replace Qdrant with PostgreSQL pgvector for the primary
vector-search implementation.

Qdrant must support:

-   document chunks
-   embeddings
-   metadata filtering
-   source/version metadata
-   collection versioning strategy
-   retrieval by dataset/source/model metadata

Each vector payload should contain enough provenance to answer:

> Why was this chunk retrieved for this answer?

At minimum:

``` text
document_id
document_version
chunk_id
source_name
source_url
dataset_version
ingestion_run_id
content_hash
text
created_at
```

------------------------------------------------------------------------

# 7. Embeddings

Use Sentence Transformers / Hugging Face-compatible embedding models.

Choose a strong, openly available embedding model that is practical to
run locally.

Do NOT hard-code the embedding model deep inside business logic.

Configuration must contain:

``` yaml
embedding:
  provider: sentence_transformers
  model: ...
  dimensions: ...
  normalize: true
```

The exact model can be selected during implementation based on hardware
and compatibility, but once selected it must be version-pinned and
recorded in experiment metadata.

Changing embedding models must require a deliberate index rebuild.

------------------------------------------------------------------------

# 8. Reranking

Implement a second-stage reranker.

Use a Hugging Face cross-encoder/reranker model, preferably a BGE-family
reranker or equivalent strong open model.

Retrieval flow:

``` text
query
  ↓
query embedding
  ↓
Qdrant top-N retrieval
  ↓
metadata filtering
  ↓
cross-encoder reranking
  ↓
top-K context
```

Make `N` and `K` configurable.

Do not pretend vector similarity alone is sufficient for high-quality
retrieval.

------------------------------------------------------------------------

# 9. LLM Runtime Strategy

Use two distinct runtime paths.

## Development / local inference

Use:

-   Ollama

Purpose:

-   easy local inference
-   developer testing
-   reproducible local model execution

## Production GPU inference

Use:

-   vLLM

Purpose:

-   high-throughput serving
-   batching
-   production GPU inference
-   OpenAI-compatible inference API

This is a deliberate improvement over using Ollama as the only
production serving layer.

Architecture:

``` text
Developer
   ↓
Ollama
   ↓
local model
```

and:

``` text
Production client
   ↓
Nginx
   ↓
FastAPI
   ↓
vLLM
   ↓
Gemma model
```

Do not force Ollama into the production GPU serving role if vLLM is
available.

------------------------------------------------------------------------

# 10. Fine-Tuning

Use:

-   Hugging Face Transformers
-   PEFT
-   LoRA
-   Unsloth
-   PyTorch
-   Accelerate

Unsloth is an optimization layer, not the architecture.

The model family is:

> Gemma 3

Select the exact Gemma 3 variant based on the available GPU/hardware and
task requirements.

Do not automatically choose the largest model.

The exact model revision must be recorded.

Training must produce an adapter artifact rather than unnecessarily
copying the entire base model.

------------------------------------------------------------------------

# 11. Training Dataset

Use a clear instruction/conversation schema.

Prefer:

``` json
{
  "messages": [
    {
      "role": "system",
      "content": "..."
    },
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ]
}
```

Each training example must have provenance metadata outside the
model-facing message payload.

Track:

-   source document
-   source URL
-   source version
-   generation prompt version
-   judge version
-   generation model
-   quality score
-   acceptance decision
-   dataset version
-   content hash

Prevent train/evaluation leakage.

Use deterministic hashing to detect exact duplicates.

Implement near-duplicate detection where practical.

------------------------------------------------------------------------

# 12. Data Pipeline

Implement explicit stages:

``` text
acquire
  ↓
parse
  ↓
normalize
  ↓
clean
  ↓
deduplicate
  ↓
chunk
  ↓
generate candidates
  ↓
judge
  ↓
curate
  ↓
split
  ↓
version
```

Each stage must be independently testable.

Do not build one giant ingestion script.

------------------------------------------------------------------------

# 13. Crawler / Source Acquisition

The crawler is responsible only for:

-   discovering allowed URLs
-   downloading source content
-   respecting robots.txt where applicable
-   enforcing domain allowlists
-   handling retries/backoff
-   recording HTTP metadata
-   storing raw source snapshots
-   producing crawl manifests

The crawler is NOT responsible for:

-   writing final training data
-   generating embeddings
-   generating Q&A
-   judging examples
-   training models

The crawler outputs raw/source artifacts.

A separate parser/normalizer converts those artifacts into canonical
documents.

This separation is mandatory.

------------------------------------------------------------------------

# 14. Canonical Document Model

Create a canonical internal model such as:

``` text
Document
├── document_id
├── source_name
├── source_url
├── title
├── version
├── content
├── content_hash
├── fetched_at
├── parser_version
├── metadata
└── sections[]
```

Every downstream stage consumes canonical documents rather than raw
HTML.

------------------------------------------------------------------------

# 15. Dataset Versioning

Use:

-   DVC
-   Git

Relationship:

``` text
Git
 └── source code
 └── configs
 └── pipeline definitions
 └── DVC metadata

DVC
 └── datasets
 └── large artifacts
```

Do not commit large datasets directly to Git.

Use an object-storage backend for DVC.

Default self-hosted development choice:

-   MinIO

Production/cloud-compatible design:

-   S3-compatible object storage

The implementation must keep the storage abstraction portable.

------------------------------------------------------------------------

# 16. Object Storage

Use:

-   S3-compatible object storage
-   MinIO locally/self-hosted

Use object storage for:

-   raw documents
-   processed datasets
-   evaluation results
-   model artifacts
-   exported reports
-   DVC cache/remote where appropriate
-   MLflow artifacts

Do not store large binary artifacts in PostgreSQL.

------------------------------------------------------------------------

# 17. LLM-as-a-Judge

Implement the judge as a separate component.

The judge must return structured JSON.

Example:

``` json
{
  "relevance": 4,
  "correctness": 5,
  "groundedness": 5,
  "instruction_following": 4,
  "clarity": 4,
  "overall": 4.4,
  "pass": true,
  "reason": "..."
}
```

Use Pydantic models to validate judge output.

Judge prompts must be version-controlled.

Record:

-   judge model
-   judge model revision
-   judge prompt version
-   evaluation rubric version
-   dataset version
-   timestamp
-   run ID

Do not treat an LLM judge as absolute ground truth.

------------------------------------------------------------------------

# 18. Separate Two Evaluation Concepts

## Data curation

Question:

> Is this generated example good enough to enter the training dataset?

## Model evaluation

Question:

> Does the trained model perform better than the baseline?

These must be separate pipelines and metrics.

------------------------------------------------------------------------

# 19. Evaluation Stack

Use multiple evaluation layers.

## Layer A --- deterministic data validation

Check:

-   schema
-   missing fields
-   invalid records
-   duplicates
-   malformed content
-   excessive length
-   leakage
-   source provenance

## Layer B --- retrieval evaluation

Measure where applicable:

-   recall@K
-   precision@K
-   MRR
-   nDCG
-   context relevance

## Layer C --- generation evaluation

Measure:

-   correctness
-   relevance
-   groundedness
-   instruction following
-   answer completeness

## Layer D --- model comparison

Compare:

``` text
Base Gemma 3
vs
Fine-tuned Gemma 3
```

using the exact same held-out evaluation set.

## Layer E --- RAG comparison

Where applicable compare:

``` text
Base model
Fine-tuned model
Fine-tuned + RAG
```

This makes the project much more scientifically defensible.

------------------------------------------------------------------------

# 20. Evaluation Framework

Use a modern evaluation framework where it provides value.

Recommended:

-   Ragas for RAG evaluation
-   custom deterministic metrics for task-specific checks
-   LLM-as-a-Judge for qualitative scoring
-   pytest-based regression evaluation

Do not make Ragas the only source of truth.

Evaluation results must be machine-readable and stored.

------------------------------------------------------------------------

# 21. Experiment Tracking

Use:

-   MLflow

Track:

### Parameters

-   model
-   revision
-   LoRA rank
-   LoRA alpha
-   dropout
-   learning rate
-   scheduler
-   epochs
-   batch size
-   gradient accumulation
-   sequence length
-   seed
-   optimizer
-   dataset version
-   embedding model
-   reranker model

### Metrics

-   train loss
-   validation loss
-   task metrics
-   retrieval metrics
-   judge metrics
-   latency where relevant

### Artifacts

-   configs
-   evaluation JSON
-   plots
-   dataset statistics
-   adapter
-   model metadata
-   reports

------------------------------------------------------------------------

# 22. MLflow + Model Registry

Use MLflow Model Registry for model lifecycle metadata.

Lifecycle:

``` text
training
  ↓
candidate
  ↓
evaluation
  ↓
quality gate
  ↓
approved
  ↓
production
```

Do not deploy every successful training run.

A model becomes deployable only after the quality gate passes.

------------------------------------------------------------------------

# 23. Traceability Contract

Every model version must be traceable to:

``` text
model version
↓
MLflow run ID
↓
Git commit
↓
DVC dataset version
↓
training configuration
↓
base model revision
↓
LoRA configuration
↓
evaluation results
↓
deployment version
```

This is a mandatory LLMOps feature.

Create a machine-readable model manifest:

``` yaml
model_version: ...
git_commit: ...
dvc_revision: ...
mlflow_run_id: ...
base_model: ...
base_model_revision: ...
adapter_version: ...
training_config_hash: ...
evaluation_report: ...
approved: true
```

------------------------------------------------------------------------

# 24. Pipeline Orchestration

Use:

-   Prefect

Reason:

-   lightweight modern workflow orchestration
-   Python-native
-   easy local development
-   supports scheduled/triggered workflows
-   less operational overhead than adopting Kubernetes-native workflow
    systems too early

Use Prefect for:

-   ingestion flows
-   embedding flows
-   data curation flows
-   evaluation flows
-   training orchestration
-   deployment promotion workflows

Do not put every function into a Prefect task.

Keep domain logic in normal Python services and use Prefect for
orchestration.

------------------------------------------------------------------------

# 25. Background Jobs

Use:

-   Redis

Use it for:

-   caching
-   rate limiting
-   short-lived job state where appropriate

Do not introduce Celery unless a concrete workload requires it.

Prefect is the primary workflow orchestrator.

Avoid two competing orchestration systems.

------------------------------------------------------------------------

# 26. API Design

Implement:

``` text
GET  /health
GET  /ready
GET  /version

POST /v1/chat
POST /v1/generate
POST /v1/retrieve

GET  /v1/models
GET  /v1/models/{version}

POST /v1/feedback
```

Keep training and administrative pipeline endpoints separate from public
inference APIs.

Use API versioning.

Use structured request/response schemas.

Return provenance in controlled form for RAG responses.

Example:

``` json
{
  "answer": "...",
  "model_version": "...",
  "run_id": "...",
  "sources": [
    {
      "document_id": "...",
      "title": "...",
      "url": "...",
      "chunk_id": "..."
    }
  ]
}
```

------------------------------------------------------------------------

# 27. Authentication and Authorization

Implement production-oriented security.

Use:

-   OAuth2/OIDC-compatible design
-   JWT access tokens
-   Argon2 password hashing if local authentication is provided
-   RBAC

Roles:

``` text
viewer
developer
ml_engineer
admin
```

Do not expose MLflow, Qdrant, PostgreSQL, MinIO, Grafana, or Prefect
directly to the public internet.

Put administrative interfaces behind authentication/network controls.

------------------------------------------------------------------------

# 28. API Security

Implement:

-   request validation
-   size limits
-   rate limiting
-   timeout handling
-   CORS policy
-   security headers
-   secret management
-   structured audit logging
-   dependency vulnerability scanning

Never log:

-   passwords
-   tokens
-   API keys
-   sensitive user content unnecessarily

------------------------------------------------------------------------

# 29. Gateway / Reverse Proxy

Use:

-   Nginx

Responsibilities:

-   TLS termination
-   reverse proxy
-   routing
-   request size limits
-   rate limiting where appropriate
-   security headers
-   upstream health handling

Architecture:

``` text
Internet
   ↓
Nginx
   ↓
FastAPI
   ↓
vLLM
```

Do not claim Nginx is the authentication system.

------------------------------------------------------------------------

# 30. Production Inference

Use:

-   vLLM
-   Gemma 3
-   GPU acceleration where available

The API should not load the model itself in the production architecture
when vLLM is used as the dedicated serving engine.

The FastAPI service acts as the application/orchestration layer.

vLLM acts as the model serving layer.

------------------------------------------------------------------------

# 31. Local Development Inference

Use Ollama.

Provide:

``` text
make llm-up
make llm-pull
make dev
```

or equivalent commands.

The local developer workflow should be possible without requiring
Kubernetes.

------------------------------------------------------------------------

# 32. Containers

Use Docker.

Create separate images where appropriate:

``` text
api
worker/pipeline
training
evaluation
```

Do not create one giant image containing every ML dependency.

GPU training/inference images should be isolated from the lightweight
API image.

Use multi-stage builds where useful.

Run containers as non-root where possible.

Pin base images by digest in production-oriented configurations where
practical.

------------------------------------------------------------------------

# 33. Local Orchestration

Use Docker Compose for local development.

Compose services may include:

``` text
postgres
qdrant
redis
minio
mlflow
prefect
api
nginx
ollama
grafana
prometheus
loki
```

Do not require all services to run by default on a laptop.

Provide profiles:

``` text
core
ml
observability
all
```

Example:

``` bash
docker compose --profile core up -d
docker compose --profile ml up -d
docker compose --profile observability up -d
```

------------------------------------------------------------------------

# 34. Kubernetes

For production deployment, support Kubernetes.

Use:

-   Kubernetes
-   Helm
-   Ingress
-   Horizontal Pod Autoscaler where useful
-   Secrets / external secret strategy
-   PersistentVolumes only where appropriate

Do not make Kubernetes mandatory for local development.

------------------------------------------------------------------------

# 35. Kubernetes Deployment Architecture

Target:

``` text
                    Internet
                       ↓
                Load Balancer
                       ↓
                    Ingress
                       ↓
                 Nginx / Gateway
                       ↓
                 FastAPI replicas
                       ↓
             ┌─────────┴─────────┐
             ↓                   ↓
          Qdrant               vLLM
             ↓                   ↓
          Storage             GPU nodes
```

Managed infrastructure may be used for:

-   PostgreSQL
-   object storage
-   Kubernetes
-   observability

But the repository must remain cloud-portable.

------------------------------------------------------------------------

# 36. Helm

Create Helm charts for:

-   API
-   inference
-   workers/pipelines where required

Use values for:

-   image
-   replicas
-   resources
-   environment
-   secrets references
-   ingress
-   autoscaling
-   model configuration

Do not hard-code environment-specific values.

Provide:

``` text
values-dev.yaml
values-staging.yaml
values-prod.yaml
```

------------------------------------------------------------------------

# 37. Infrastructure as Code

Use:

-   Terraform

Terraform should provision the infrastructure boundary, not application
internals.

Target resources may include:

-   Kubernetes cluster
-   network
-   object storage
-   database
-   container registry
-   IAM/service accounts
-   monitoring resources

Keep Terraform modules reusable.

Do not hard-code cloud credentials.

Provide a local/dev path that does not require cloud resources.

------------------------------------------------------------------------

# 38. GitOps

Use:

-   Argo CD

Deployment flow:

``` text
GitHub
  ↓
GitHub Actions
  ↓
build image
  ↓
push image
  ↓
update deployment manifest / Helm values
  ↓
Argo CD
  ↓
Kubernetes
```

CI builds and validates.

Argo CD performs deployment synchronization.

Do not make GitHub Actions directly run arbitrary `kubectl` production
deployments as the primary mechanism.

------------------------------------------------------------------------

# 39. Container Registry

Use an OCI-compatible registry.

Default CI example:

-   GitHub Container Registry

Keep registry configuration abstract.

Images must be tagged with immutable identifiers, preferably:

``` text
git SHA
```

Do not use `latest` as the deployment identity.

------------------------------------------------------------------------

# 40. CI/CD

Use GitHub Actions.

Create separate workflows.

## `ci.yml`

Run on pull requests and pushes:

``` text
checkout
↓
uv sync
↓
ruff
↓
mypy
↓
pytest
↓
coverage
↓
security/dependency checks
```

## `docker.yml`

Build:

-   API image
-   pipeline image
-   inference image where required

Scan images.

Push only from trusted branches/tags.

## `data-validation.yml`

Validate dataset changes.

## `evaluation.yml`

Run lightweight evaluation/regression checks.

## `training.yml`

GPU training workflow.

Do NOT run expensive GPU training on every commit.

## `release.yml`

Promote an approved version.

------------------------------------------------------------------------

# 41. CI Quality Gates

A pull request should fail if:

-   formatting fails
-   linting fails
-   typing fails
-   unit tests fail
-   critical security scan fails
-   API contract tests fail

Training promotion should fail if:

-   evaluation quality is below threshold
-   regression is detected
-   dataset validation fails
-   model artifact is invalid
-   provenance manifest is incomplete

------------------------------------------------------------------------

# 42. Observability

Implement:

-   OpenTelemetry
-   Prometheus
-   Grafana
-   structured JSON logs
-   Loki

Use OpenTelemetry for:

-   request traces
-   retrieval spans
-   model invocation spans
-   database calls
-   external service calls

Prometheus metrics:

``` text
request_count
request_latency
error_count
retrieval_latency
embedding_latency
generation_latency
tokens_in
tokens_out
model_requests
cache_hits
```

Do not collect sensitive prompts by default.

------------------------------------------------------------------------

# 43. Logging

Use structured logging.

Each request should have:

``` text
request_id
trace_id
model_version
environment
latency
status
```

Do not dump full prompts and responses into logs by default.

Implement configurable redaction.

------------------------------------------------------------------------

# 44. Monitoring

Create Grafana dashboards for:

## API

-   throughput
-   latency
-   errors

## Retrieval

-   retrieval latency
-   top-K usage
-   empty retrieval rate

## Model

-   generation latency
-   token throughput
-   errors
-   active model version

## Pipeline

-   ingestion duration
-   dataset size
-   rejected examples
-   judge pass rate
-   training duration
-   evaluation score

------------------------------------------------------------------------

# 45. Continuous Evaluation

Implement a regression evaluation suite.

Every candidate model must be evaluated against a fixed evaluation
dataset.

Track:

``` text
current model
previous production model
baseline model
```

Promotion requires no unacceptable regression.

This is more important than simply reporting one impressive score.

------------------------------------------------------------------------

# 46. Prompt Management

Prompts are code/configuration.

Store versioned prompts under:

``` text
prompts/
  generation/
  judge/
  rag/
  system/
```

Every evaluation result must identify its prompt version.

Do not hide prompts inside Python strings scattered across the
repository.

------------------------------------------------------------------------

# 47. Configuration Management

Use YAML/TOML configuration plus environment variables.

Suggested structure:

``` text
configs/
  application.yaml
  ingestion.yaml
  embeddings.yaml
  retrieval.yaml
  generation.yaml
  training.yaml
  evaluation.yaml
  deployment.yaml
```

Secrets come from environment variables or secret stores.

Never commit `.env`.

Provide `.env.example`.

------------------------------------------------------------------------

# 48. Dependency Management

Use `uv`.

Pin dependencies through the generated lock file.

Do not manually maintain duplicate requirements files unless required
for a deployment image.

Use dependency groups:

``` text
dev
test
training
inference
docs
```

Keep the API dependency set lightweight.

------------------------------------------------------------------------

# 49. Code Architecture

Use clean architecture / modular boundaries.

Recommended:

``` text
src/llmops/
  domain/
  application/
  infrastructure/
  interfaces/
```

Domain layer:

-   entities
-   value objects
-   domain contracts

Application layer:

-   use cases
-   orchestration

Infrastructure:

-   PostgreSQL
-   Qdrant
-   S3/MinIO
-   MLflow
-   model runtimes
-   external services

Interfaces:

-   FastAPI
-   CLI
-   pipeline adapters

Do not allow infrastructure-specific code to leak everywhere.

------------------------------------------------------------------------

# 50. Repository Structure

Use approximately:

``` text
llmops/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── Makefile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── dvc.yaml
├── params.yaml
│
├── configs/
├── prompts/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── evaluation/
│   └── external/
│
├── src/
│   └── llmops/
│       ├── domain/
│       ├── application/
│       ├── infrastructure/
│       └── interfaces/
│
├── pipelines/
│
├── scripts/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── contract/
│   └── evaluation/
│
├── notebooks/
│
├── docker/
│   ├── api/
│   ├── pipeline/
│   ├── training/
│   └── inference/
│
├── deployment/
│   ├── compose/
│   ├── helm/
│   ├── terraform/
│   └── argocd/
│
├── observability/
│   ├── prometheus/
│   ├── grafana/
│   ├── loki/
│   └── otel/
│
├── nginx/
│
├── docs/
│   ├── architecture/
│   ├── decisions/
│   ├── runbooks/
│   ├── api/
│   └── development/
│
└── .github/
    └── workflows/
```

Adjust only when a concrete reason exists.

------------------------------------------------------------------------

# 51. Architecture Documentation

Create:

``` text
docs/architecture/
  overview.md
  data-flow.md
  training.md
  evaluation.md
  inference.md
  deployment.md
  observability.md
```

Include Mermaid diagrams.

The diagrams must match the actual implementation.

Never document components that do not exist.

------------------------------------------------------------------------

# 52. Architecture Decision Records

Every important technology choice gets an ADR.

Examples:

``` text
docs/decisions/
  001-python-runtime.md
  002-qdrant.md
  003-mlflow.md
  004-prefect.md
  005-vllm.md
  006-ollama.md
  007-kubernetes.md
  008-argocd.md
  009-observability.md
```

Each ADR should contain:

-   context
-   decision
-   alternatives
-   consequences

This prevents technology choices from becoming unexplained cargo cult.

------------------------------------------------------------------------

# 53. Testing Strategy

Use a test pyramid.

## Unit tests

Test:

-   parsers
-   normalizers
-   chunkers
-   deduplication
-   config
-   prompt construction
-   schemas
-   quality gates
-   metadata
-   repository logic

## Integration tests

Test:

``` text
API → PostgreSQL
API → Qdrant
API → model runtime
pipeline → object storage
pipeline → MLflow
```

Use disposable test infrastructure where possible.

## Contract tests

Verify:

-   API schemas
-   model serving contract
-   metadata contract
-   pipeline artifact contract

## Evaluation tests

Use fixed datasets for regression.

------------------------------------------------------------------------

# 54. Test Commands

Provide simple commands:

``` bash
make test
make lint
make typecheck
make format
make check
make test-integration
make evaluate
```

`make check` should run the complete lightweight local quality gate.

------------------------------------------------------------------------

# 55. Reproducibility

Record:

-   Git commit
-   DVC revision
-   Python version
-   dependency lock
-   model revision
-   embedding model
-   reranker model
-   training config
-   random seed
-   hardware
-   CUDA/PyTorch versions
-   MLflow run ID

Do not claim bit-for-bit reproducibility unless verified.

Target reproducible experiments, not unrealistic marketing claims.

------------------------------------------------------------------------

# 56. Data Leakage Protection

Implement checks for:

-   exact duplicate overlap
-   near-duplicate overlap where practical
-   source overlap
-   answer leakage
-   evaluation examples accidentally entering training

The evaluation dataset must be immutable once released as a benchmark
version.

------------------------------------------------------------------------

# 57. Data Quality Reports

Every dataset version should produce a report containing:

-   number of documents
-   number of chunks
-   token statistics
-   duplicate counts
-   rejected records
-   judge acceptance rate
-   source distribution
-   train/validation/test distribution

Store the report as an artifact.

------------------------------------------------------------------------

# 58. Training Configuration

Never hard-code training parameters.

Example:

``` yaml
model:
  name: ...
  revision: ...

lora:
  rank: ...
  alpha: ...
  dropout: ...

training:
  seed: 42
  learning_rate: ...
  batch_size: ...
  gradient_accumulation_steps: ...
  epochs: ...
  max_seq_length: ...
  warmup_ratio: ...
```

The selected values must be justified and tracked in MLflow.

------------------------------------------------------------------------

# 59. Baselines

The project must establish at least:

1.  base Gemma 3
2.  fine-tuned Gemma 3
3.  fine-tuned Gemma 3 + RAG where applicable

Use the same evaluation benchmark.

Do not claim fine-tuning is beneficial unless the measured results
support it.

------------------------------------------------------------------------

# 60. Quality Gate

Create an explicit quality-gate component.

Conceptually:

``` python
if candidate_metrics >= required_thresholds:
    approve()
else:
    reject()
```

The gate must consider:

-   task quality
-   regression
-   retrieval quality where relevant
-   safety/basic policy checks
-   latency/resource constraints where applicable

The thresholds must be configuration-driven.

------------------------------------------------------------------------

# 61. Model Promotion

Use:

``` text
candidate
approved
staging
production
archived
```

A deployment must reference an immutable model version.

Rollback must be possible.

Do not overwrite a production model in place.

------------------------------------------------------------------------

# 62. Deployment Environments

Support:

``` text
local
development
staging
production
```

Local:

-   Docker Compose
-   Ollama

Development/staging:

-   Kubernetes optional
-   vLLM where GPU exists

Production:

-   Kubernetes
-   Helm
-   Argo CD
-   vLLM
-   Nginx/Ingress
-   managed or self-hosted infrastructure

------------------------------------------------------------------------

# 63. Secrets

Never commit:

-   Hugging Face tokens
-   database passwords
-   cloud credentials
-   JWT secrets
-   API keys
-   MLflow credentials

Local:

``` text
.env
```

CI:

``` text
GitHub Secrets
```

Kubernetes:

-   Secrets
-   preferably External Secrets / cloud secret manager in production

------------------------------------------------------------------------

# 64. Security Scanning

Add:

-   Ruff security rules where practical
-   Bandit
-   pip-audit or equivalent dependency audit
-   Trivy for container scanning
-   secret scanning through GitHub security tooling

Do not block development on every low-severity finding; configure
sensible thresholds.

Critical vulnerabilities must block release.

------------------------------------------------------------------------

# 65. Performance Engineering

Measure:

-   ingestion throughput
-   embedding throughput
-   retrieval latency
-   reranking latency
-   generation latency
-   end-to-end latency

Use batching where appropriate.

Do not prematurely optimize.

Every performance optimization must have a measurable reason.

------------------------------------------------------------------------

# 66. Caching

Use Redis for:

-   repeated retrieval/cacheable metadata
-   rate limiting
-   short-lived application state

Do not cache user-specific responses without considering correctness and
privacy.

Cache keys must include relevant model/config/index versions.

------------------------------------------------------------------------

# 67. RAG Design

RAG should be a first-class subsystem.

Pipeline:

``` text
question
 ↓
query normalization
 ↓
query embedding
 ↓
Qdrant retrieval
 ↓
metadata filtering
 ↓
reranking
 ↓
context selection
 ↓
prompt construction
 ↓
Gemma/vLLM
 ↓
citation/provenance
```

The answer should cite the source documents used.

Do not allow arbitrary unrelated context into the prompt.

Implement configurable context limits.

------------------------------------------------------------------------

# 68. Hallucination / Grounding Evaluation

For RAG responses, evaluate:

-   citation correctness
-   context relevance
-   groundedness
-   answer relevance

Use both deterministic checks and LLM-based evaluation where useful.

The system should be able to detect when retrieval returns insufficient
evidence.

In such cases, the model should not confidently invent an answer.

------------------------------------------------------------------------

# 69. Documentation Ingestion Safety

The crawler must have:

-   allowlist
-   maximum page size
-   timeout
-   retry limit
-   rate limit
-   robots handling
-   URL normalization
-   content-type validation
-   SSRF-safe networking rules

Never let user-controlled URLs cause unrestricted server-side requests.

------------------------------------------------------------------------

# 70. API Reliability

Implement:

-   timeouts
-   retries only for safe/idempotent operations
-   circuit-breaking or graceful degradation where justified
-   health checks
-   readiness checks
-   graceful shutdown

Do not retry model generation blindly.

------------------------------------------------------------------------

# 71. Health Endpoints

Implement:

``` text
/health
```

for process liveness.

Implement:

``` text
/ready
```

for dependency readiness.

Readiness should verify required dependencies according to service role.

Do not make liveness depend on every external service.

------------------------------------------------------------------------

# 72. CLI

Provide a CLI for major operations.

Example:

``` bash
llmops ingest
llmops validate
llmops embed
llmops curate
llmops evaluate
llmops train
llmops promote
llmops deploy
```

The CLI should call application services, not duplicate business logic.

------------------------------------------------------------------------

# 73. Makefile / Developer Experience

Provide:

``` bash
make install
make dev
make test
make lint
make typecheck
make check
make up
make down
make logs
make ingest
make embed
make evaluate
make train
```

A new developer should be able to understand the project from:

``` text
README.md
make help
```

------------------------------------------------------------------------

# 74. Documentation Quality

README must include:

-   project objective
-   architecture
-   stack
-   prerequisites
-   local setup
-   first end-to-end run
-   testing
-   training
-   evaluation
-   deployment
-   observability
-   troubleshooting
-   reproducibility
-   project limitations

Do not write marketing claims unsupported by implementation.

------------------------------------------------------------------------

# 75. What NOT to Do

Do not:

-   add technologies only for CV keywords
-   create giant monolithic scripts
-   mix training and serving
-   mix crawling and dataset generation
-   hard-code secrets
-   hard-code model names throughout code
-   silently download arbitrary models
-   expose internal services publicly
-   deploy `latest`
-   use mutable production model tags
-   claim production readiness without evidence
-   claim model improvement without baseline evaluation
-   skip tests because ML code is "experimental"
-   hide important configuration in code
-   invent dataset facts
-   invent evaluation results
-   use fake telemetry
-   create fake Kubernetes manifests that do not work
-   create placeholder architecture documentation presented as complete

------------------------------------------------------------------------

# 76. Implementation Philosophy

Build vertically, not horizontally.

Do NOT spend weeks creating infrastructure before proving the core
pipeline.

Milestone order:

## Milestone 0 --- Repository foundation

Implement:

-   project structure
-   uv
-   pyproject
-   Ruff
-   mypy
-   pytest
-   pre-commit
-   Makefile
-   CI

Definition of done:

``` text
clone
→ uv sync
→ make check
→ green
```

## Milestone 1 --- Source acquisition

Implement:

-   documentation crawler
-   allowlists
-   raw snapshots
-   crawl manifest
-   canonical document model

## Milestone 2 --- Processing

Implement:

-   parser
-   normalization
-   chunking
-   deduplication
-   dataset reports

## Milestone 3 --- DVC

Implement:

-   DVC
-   object storage
-   dataset versioning
-   reproducibility metadata

## Milestone 4 --- Retrieval

Implement:

-   embeddings
-   Qdrant
-   retrieval
-   reranking
-   retrieval evaluation

## Milestone 5 --- Generation

Implement:

-   local Ollama
-   generation service
-   structured output
-   RAG

## Milestone 6 --- Data curation

Implement:

-   synthetic Q&A generation
-   LLM judge
-   quality filtering
-   curated dataset

## Milestone 7 --- Training

Implement:

-   Gemma 3
-   Unsloth
-   PEFT/LoRA
-   MLflow
-   reproducible training

## Milestone 8 --- Evaluation

Implement:

-   baseline
-   fine-tuned evaluation
-   RAG evaluation
-   regression suite
-   quality gates

## Milestone 9 --- Production inference

Implement:

-   vLLM
-   FastAPI
-   Nginx
-   model manifest
-   production-like Docker deployment

## Milestone 10 --- Observability

Implement:

-   OpenTelemetry
-   Prometheus
-   Grafana
-   Loki
-   dashboards
-   traces

## Milestone 11 --- Kubernetes

Implement:

-   Helm
-   Kubernetes manifests
-   resource requests/limits
-   autoscaling where justified

## Milestone 12 --- GitOps

Implement:

-   container registry
-   Argo CD
-   environment promotion
-   rollback

## Milestone 13 --- Infrastructure

Implement:

-   Terraform
-   environment provisioning
-   cloud-portable architecture

## Milestone 14 --- Continuous LLMOps

Implement:

``` text
source update
 ↓
ingestion
 ↓
dataset validation
 ↓
evaluation
 ↓
optional training
 ↓
model evaluation
 ↓
quality gate
 ↓
promotion
 ↓
deployment
 ↓
monitoring
```

Do not make automatic retraining mandatory for every data change. Use
policy/configuration to decide when retraining is justified.

------------------------------------------------------------------------

# 77. Definition of Done

The project is complete only when a clean machine can reproduce the
documented workflow.

At minimum:

``` text
git clone
uv sync
docker compose up
make check
```

must work for the local development path.

The full demonstration must show:

``` text
official docs
 ↓
crawl
 ↓
canonical documents
 ↓
DVC dataset
 ↓
Qdrant index
 ↓
RAG
 ↓
synthetic examples
 ↓
LLM judge
 ↓
curated dataset
 ↓
Gemma 3 LoRA training
 ↓
MLflow run
 ↓
evaluation
 ↓
quality gate
 ↓
model version
 ↓
vLLM/Ollama inference
 ↓
FastAPI
 ↓
Nginx
 ↓
observability
```

The Kubernetes/GitOps path must also be documented and tested at the
level realistically possible in the available environment.

------------------------------------------------------------------------

# 78. Acceptance Tests

Create an end-to-end acceptance test that verifies:

1.  source acquisition works
2.  canonical document is produced
3.  dataset version is recorded
4.  embeddings are generated
5.  Qdrant contains expected vectors
6.  retrieval returns relevant chunks
7.  reranking changes/validates ranking where appropriate
8.  RAG produces a grounded answer
9.  provenance is returned
10. synthetic training examples are generated
11. judge produces valid structured output
12. rejected examples are excluded
13. dataset is versioned
14. training can consume the dataset
15. MLflow records the training run
16. evaluation compares baseline vs candidate
17. quality gate makes a deterministic decision
18. model manifest is generated
19. inference serves the selected model
20. telemetry is emitted
21. rollback metadata exists

Do not fake this test. If GPU training is unavailable, create a clearly
marked lightweight integration path while keeping the real GPU path
implemented and documented.

------------------------------------------------------------------------

# 79. Hardware Strategy

Design for two hardware profiles.

## Developer laptop

-   CPU-first where possible
-   Ollama for local inference
-   lightweight embeddings
-   small datasets
-   mocked/minimal training smoke test

## GPU training/production

-   CUDA-capable GPU
-   Unsloth
-   PyTorch
-   vLLM
-   full evaluation

The code must detect unsupported hardware and fail with a useful message
rather than crashing mysteriously.

------------------------------------------------------------------------

# 80. Cost Awareness

Do not assume unlimited GPU or cloud resources.

Support:

-   small smoke datasets
-   full datasets
-   configurable training sizes
-   configurable evaluation sizes

Make expensive operations explicit.

For example:

``` bash
llmops train --profile smoke
llmops train --profile full
```

------------------------------------------------------------------------

# 81. Feature Flags / Profiles

Use configuration profiles rather than code forks.

Examples:

``` text
local
ci
smoke
gpu
staging
production
```

Feature toggles may control:

-   Ollama vs vLLM
-   local vs remote object storage
-   observability
-   expensive evaluation
-   training size

Do not scatter environment conditionals across business logic.

------------------------------------------------------------------------

# 82. Versioning

Version:

-   API
-   datasets
-   prompts
-   embedding model
-   reranker
-   base model
-   LoRA adapter
-   evaluation rubric
-   deployment
-   configuration

Do not rely solely on Git tags.

------------------------------------------------------------------------

# 83. Artifact Naming

Use immutable identifiers.

Examples:

``` text
dataset-{hash}
model-{version}
adapter-{version}
evaluation-{run_id}
```

Avoid ambiguous names such as:

``` text
final_model
new_model
latest_model
```

------------------------------------------------------------------------

# 84. Release Process

A release should follow:

``` text
PR
 ↓
CI
 ↓
merge
 ↓
image build
 ↓
candidate model evaluation
 ↓
quality gate
 ↓
model registry
 ↓
staging
 ↓
smoke test
 ↓
production promotion
 ↓
monitor
```

Production rollback:

``` text
previous approved model
 ↓
redeploy
```

------------------------------------------------------------------------

# 85. Final Engineering Rule

Whenever there are two possible implementations:

1.  prefer the simpler one
2.  prefer the one with a clear operational purpose
3.  prefer standard open-source tooling
4.  prefer reproducibility
5.  prefer explicit contracts
6.  prefer measurable evaluation
7.  prefer immutable artifacts
8.  prefer boring infrastructure over unnecessary complexity

The project should be technically ambitious but operationally coherent.

------------------------------------------------------------------------

# 86. Immediate Instructions to Antigravity

Start by inspecting the current repository.

Then:

1.  inventory existing code
2.  inventory current tests
3.  inventory existing dependencies
4.  identify what already works
5.  preserve working components when they satisfy this specification
6.  identify deviations from this specification
7.  create/update ADRs for decisions
8.  implement the project milestone by milestone
9.  run tests after every meaningful change
10. never silently replace working behavior
11. never delete tests to make the suite pass
12. never fabricate successful results

At the beginning of each milestone, state:

``` text
GOAL
FILES TO CHANGE
DESIGN DECISION
IMPLEMENTATION
TESTS
DEFINITION OF DONE
```

After implementation, report:

``` text
IMPLEMENTED
TESTS
KNOWN LIMITATIONS
NEXT MILESTONE
```

Do not ask for confirmation at every step.

Make reasonable engineering decisions autonomously.

------------------------------------------------------------------------

# 87. Final Success Criterion

The final repository should demonstrate that the developer understands
not only how to fine-tune an LLM, but how to operate an LLM system as an
engineered lifecycle:

``` text
DATA
→ VERSIONING
→ QUALITY
→ RETRIEVAL
→ GENERATION
→ EVALUATION
→ TRAINING
→ EXPERIMENT TRACKING
→ MODEL REGISTRY
→ SERVING
→ API
→ SECURITY
→ OBSERVABILITY
→ CI/CD
→ GITOPS
→ INFRASTRUCTURE
→ CONTINUOUS EVALUATION
```

The resulting project should be:

-   reproducible
-   testable
-   observable
-   secure
-   deployable
-   rollback-capable
-   measurable
-   cloud-portable
-   understandable
-   interview-defensible

Do not optimize for the number of technologies.

Optimize for demonstrating that every technology has a reason to exist
and that the complete lifecycle actually works.
