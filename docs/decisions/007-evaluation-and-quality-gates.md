# ADR 007: Multi-Layer Evaluation Stack and Deterministic Quality Gates

## Status
Accepted

## Context
Deploying LLMs without structured regression testing risks shipping models with severe factual hallucination, instruction breakage, or latency regressions. Traditional single-number benchmarks fail to distinguish between data errors, retrieval failures, and generator weaknesses.

## Decision
1. **Multi-Layer Evaluation Taxonomy (Section 19)**:
   - **Layer A (Data Validation)**: Deterministic verification of schemas, message roles, duplication rates, and leakage boundaries.
   - **Layer B (Retrieval Evaluation)**: Information retrieval scoring (Recall@K, MRR, nDCG, context relevance) against gold standard references.
   - **Layer C (Task Generation Quality)**: Multi-attribute scoring (correctness, relevance, groundedness, clarity) via `DataCurationJudge`.
   - **Layer D (Model Comparison)**: Direct head-to-head comparison between Base Gemma 3 and Fine-Tuned Gemma 3 on an identical held-out test split.
   - **Layer E (System Comparison)**: Evaluation of Fine-Tuned alone versus Fine-Tuned + RAG context to prove grounding superiority.
2. **Deterministic Quality Gate (Section 22)**:
   - Configurable thresholds (`QualityGateThresholds`) determine promotion eligibility.
   - A model is approved for staging/production only if all layers pass without regression.
   - Failures generate machine-readable `evaluation_report.json` with explicit rejection reasons.
3. **Traceable Promotion Contract**:
   - `PromoteModelUseCase` updates `ModelManifest.approved = True` and binds the evaluation report directly into the model manifest. Unvetted models cannot be promoted.

## Consequences
- Prevents silent regressions from reaching production environments.
- Provides transparent, auditable evidence for stakeholders defending model safety and domain accuracy.
