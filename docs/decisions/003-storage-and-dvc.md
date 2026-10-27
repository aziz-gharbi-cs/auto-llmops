# ADR 003: Storage Abstraction, MinIO / S3, and DVC Dataset Versioning

## Status
Accepted

## Context
Machine learning datasets, raw HTML snapshots, and model checkpoints are large binary or JSONL artifacts. Committing them directly to Git causes repository bloat, sluggish clones, and merge conflicts. Furthermore, production and development environments often differ: developers may work locally on disk or MinIO containers, while production runs against AWS S3 or GCP Cloud Storage.

## Decision
1. **DVC (Data Version Control)**: We adopt DVC for pipeline tracking (`dvc.yaml`, `params.yaml`) and large file hashing. Git tracks DVC pointer files, code, configs, and parameters; DVC tracks data directories (`data/raw`, `data/interim`, `data/processed`).
2. **Object Storage Interface (`BaseStorage`)**: We enforce a provider-agnostic storage abstraction. Application use cases interact solely with `BaseStorage`.
3. **Pluggable Backends**:
   - `LocalStorageAdapter` for local development, tests, and air-gapped environments.
   - `S3StorageAdapter` using `boto3` for MinIO (locally/self-hosted) and AWS S3 in cloud deployments.
4. **Dataset Version Manifests (`DatasetVersionManifest`)**: Machine-readable JSON manifests tie every dataset release to a Git commit SHA, DVC revision hash, storage URI, and dataset quality report.

## Alternatives Considered
- **Committing Data directly to Git with Git-LFS**: Expensive, lacks declarative pipeline stage orchestration (`dvc.yaml`), and creates vendor lock-in.
- **Storing Large Blobs in PostgreSQL (BYTEA)**: Poor performance, database bloat, and operational complexity.
- **Direct S3 coupling in use cases**: Breaks portability and requires running live cloud services during unit testing.

## Consequences
- Clean separation between code versioning (Git) and data versioning (DVC/Storage).
- Fully testable without requiring an active AWS account or cloud credentials.
- Full provenance and audit trail for every trained model's underlying dataset.
