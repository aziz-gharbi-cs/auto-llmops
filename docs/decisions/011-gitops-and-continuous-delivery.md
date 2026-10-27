# ADR 011: GitOps Delivery Architecture (Argo CD & Immutable Artifact Promotion)

## Status
Accepted

## Context
Deploying LLM platforms across development, staging, and production clusters requires predictable, declarative synchronization without granting arbitrary production cluster write access directly to CI pipelines (Section 38-40). Furthermore, container images must use immutable identifiers (Git commit SHAs) rather than mutable `latest` tags (Section 39).

## Decision
1. **Argo CD as Single Point of Cluster Synchronization**:
   - Continuous Integration (GitHub Actions) builds, runs unit and regression tests, packages images, and pushes to GitHub Container Registry (GHCR).
   - Deployment repositories and values files (`deployment/helm/llmops/values-*.yaml`) are the sole sources of truth.
   - Argo CD continuously monitors target Git revisions (`main`, `staging`, `production`) and automatically reconciles cluster state with `prune: true` and `selfHeal: true`.
2. **Immutable Container & Model Identification**:
   - Every container image built by `docker.yml` is tagged with the exact 40-character Git commit SHA.
   - Model artifacts include full Section 23 traceability (`git_commit`, `dvc_data_revision`, `mlflow_run_id`).
3. **ApplicationSet Multi-Environment Pattern**:
   - `deployment/argocd/applicationset.yaml` generates environment-specific applications (`llmops-dev`, `llmops-staging`, `llmops-prod`) with isolated namespaces.
4. **Declarative Rollback Strategy**:
   - Routine rollbacks are executed by git-reverting values files on the target branch (`docs/runbooks/rollback.md`).
   - Emergency rollbacks are executable via `argocd app rollback` with manual sync override.

## Consequences
- Full compliance with Sections 38, 39, and 40.
- Production clusters remain completely isolated from CI pipeline credentials.
- Zero mutable tags in production environments.
