# ADR 005: Data Curation, LLM-as-a-Judge, and Leakage Prevention

## Status
Accepted

## Context
Fine-tuning models on uncurated synthetic data causes catastrophic quality degradation, hallucination loops, and style drift. Furthermore, improper train/validation/test dataset splitting introduces document-level data leakage, invalidating offline evaluation benchmarks and giving false confidence in fine-tuned performance.

## Decision
1. **Separation of Concerns (Data Curation vs Model Evaluation)**:
   - Data Curation answers: *"Is this generated Q&A pair good enough to enter the training dataset?"*
   - Model Evaluation answers: *"Does the fine-tuned model perform better than the baseline?"*
   - Curation is executed as an isolated pipeline stage before training.
2. **Structured LLM-as-a-Judge**:
   - `DataCurationJudge` evaluates candidate Q&A against source documentation chunks using a multi-dimensional rubric (relevance, correctness, groundedness, instruction following, clarity).
   - Structured JSON output conforms to strict Pydantic schema with deterministic scoring thresholds.
3. **External Audit Metadata Isolation**:
   - Model-facing datasets (`train.jsonl`, `val.jsonl`, `test.jsonl`) strictly contain conversation messages (`role`, `content`).
   - Audit provenance (source URLs, chunk IDs, generation prompt versions, judge scores, content hashes) is persisted in paired metadata files (`*_metadata.jsonl`) to ensure model training tokens are not polluted.
4. **Zero Document-Level Leakage**:
   - `DatasetSplitter` groups candidates by canonical `source_document_id`. Entire documents are assigned exclusively to either train, validation, or test partitions.
   - Exact deduplication via deterministic SHA-256 content hashes prevents duplicate sample admission.

## Consequences
- High-quality, grounded instruction datasets without hallucinated or trivial samples.
- Rigorous, leak-free validation and test sets ensuring reliable generalization metrics.
- Full provenance traceability for every single fine-tuning sample.
