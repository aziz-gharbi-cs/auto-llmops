# ADR 006: Gemma 3 Parameter-Efficient Fine-Tuning and Section 23 Traceability Contract

## Status
Accepted

## Context
Full parameter fine-tuning of multi-billion parameter foundation models is computationally prohibitive, creates immense artifact bloat (copying multi-gigabyte checkpoints per run), and risks catastrophic forgetting. Furthermore, production governance demands strict lineage tracking linking any deployed model back to the exact code commit, data revision, hyperparameters, and evaluation report.

## Decision
1. **Parameter-Efficient LoRA Adapters (PEFT)**:
   - We utilize Low-Rank Adaptation (LoRA) targeting attention projection and MLP feed-forward modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
   - Only lightweight adapter weights (`adapter_model.safetensors`, `adapter_config.json`) are exported, eliminating redundant copies of base Gemma 3 weights.
2. **Profile-Driven Hardware Strategy (Section 79/81)**:
   - `profile: smoke`: Deterministic CPU-first simulated training loop for local development, CI pipelines, and sandboxed testing.
   - `profile: full`: High-performance CUDA GPU execution utilizing PyTorch, Unsloth, and Transformers. Explicit pre-flight hardware checks fail fast with actionable guidance if CUDA is unavailable.
3. **Traceability Contract (Section 23)**:
   - Every training run produces a cryptographic, machine-readable `ModelManifest`:
     - `model_version`, `git_commit`, `dvc_revision`, `mlflow_run_id`
     - `base_model`, `base_model_revision`, `adapter_version`
     - `training_config_hash` (deterministic SHA-256 digest of training configuration)
     - `approved: false` (must pass downstream Quality Gate before promotion)
4. **Experiment Tracking Decoupling**:
   - `BaseExperimentTracker` port abstracts MLflow and local file-based tracking. In offline or development modes, `LocalExperimentTracker` records all runs, parameters, loss curves, and artifacts without external service dependencies.

## Consequences
- Fast, reproducible development workflow that runs anywhere (CPU laptops to multi-GPU clusters).
- 100% auditable model provenance conforming to strict enterprise MLOps compliance standards.
