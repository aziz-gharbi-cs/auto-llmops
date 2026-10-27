# ADR 013: Continuous LLMOps Lifecycle and Acceptance Verification

## Status
Accepted

## Context
Section 77 and Section 78 of the Master Build Specification define the complete continuous LLMOps lifecycle and the mandatory end-to-end acceptance criteria. The platform must orchestrate documentation ingestion, vector retrieval indexing, synthetic curation with LLM-as-a-judge, conditional fine-tuning, multi-layer evaluation gates (Layers A through E), declarative model promotion, production serving, and full-stack observability. Furthermore, every single acceptance criterion from Criterion 1 to 21 must be deterministically verifiable offline and within local developer environments.

## Decision
1. **Continuous LLMOps Pipeline Orchestration**:
   - Implemented `ContinuousLLMOpsPipeline` in `src/llmops/application/use_cases/continuous_pipeline.py`.
   - The engine unifies:
     - Document ingestion, normalization, and semantic chunking (`ProcessDocumentsUseCase`).
     - Vector store indexing with hybrid dense embeddings (`RetrievalService`).
     - Synthetic Q&A candidate generation and LLM-as-a-judge filtering (`CurateDatasetUseCase`).
     - Conditional LoRA fine-tuning (`TrainModelUseCase`).
     - Multi-layer evaluation across schema, retrieval MRR, generation quality, and model win rates (`EvaluateModelUseCase`).
     - Deterministic model promotion gate generating Section 23 Model Manifests (`PromoteModelUseCase`).
     - Full tracing and Prometheus metric telemetry emission.
   - Exposed via CLI subcommand `llmops pipeline --config ...`.

2. **Section 78 Master Acceptance Verification**:
   - Implemented `tests/acceptance/test_section78_lifecycle.py` verifying all 21 criteria in sequence:
     1. Source acquisition via safe web crawler with domain whitelisting and SSRF protection.
     2. Canonical document generation with deterministic content hashes.
     3. Dataset version manifest recording document count and chunk provenance.
     4. Dense embedding generation with unit-norm normalization.
     5. Vector database indexing with collection existence checks.
     6. Multi-stage vector retrieval and reranking with threshold filtering.
     7. Dynamic prompt composition with versioned prompt templates.
     8. Inference execution via interchangeable LLM client contracts.
     9. Grounded RAG response generation combining context with citations.
     10. Synthetic Q&A generation from ingested chunks.
     11. LLM-as-a-judge evaluation scoring groundedness and overall quality.
     12. Deterministic rejection of low-quality or hallucinated synthetic data.
     13. Leakage-safe train/val/test dataset splitting with manifest generation.
     14. Supervised fine-tuning execution consuming curated splits.
     15. Experiment tracking recording parameters, loss metrics, and artifacts.
     16. Benchmark evaluation comparing candidate against baseline models.
     17. Multi-layer Quality Gate deterministic decision making.
     18. Complete Section 23 Model Manifest creation.
     19. FastAPI production inference serving with active model routing.
     20. Prometheus metric exposition and OpenTelemetry tracing.
     21. Automated declarative promotion and rollback runbook availability.

3. **Offline & Sandbox Reproducibility**:
   - In-memory mock clients, fallback embeddings, and synthetic evaluators allow complete lifecycle verification in sandboxed and offline environments without external cloud API dependencies.

## Consequences
- Full compliance with Section 77 (Definition of Done) and Section 78 (Acceptance Criteria).
- Unbroken traceability from crawled raw bytes to promoted production endpoints.
- High test coverage and confidence across all architectural layers.
