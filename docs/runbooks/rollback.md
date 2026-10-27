# Runbook: Rollback Procedures and Disaster Recovery

## 1. Overview
In accordance with Section 38 (GitOps) and Section 45 (Continuous Evaluation), production deployment synchronization is governed strictly through Git and Argo CD. Production rollbacks must restore cluster stability and model quality within minutes.

## 2. Emergency Rollback (Argo CD Immediate Rollback)
When a critical regression or outage occurs in production and requires instantaneous remediation before Git changes can be merged:

1. **Disable Auto-Sync** in Argo CD to prevent automated re-synchronization:
   ```bash
   argocd app set llmops-prod --sync-policy manual
   ```
2. **Rollback to Prior Successful Revision**:
   ```bash
   # List deployment history
   argocd app history llmops-prod

   # Rollback to revision ID (e.g. ID 12)
   argocd app rollback llmops-prod 12
   ```
3. **Verify Cluster Health**:
   ```bash
   argocd app get llmops-prod
   kubectl get pods -n llmops-prod -l app.kubernetes.io/component=api
   curl -k https://api.llmops.example.com/health
   curl -k https://api.llmops.example.com/ready
   ```

## 3. Declarative GitOps Rollback (Standard Operating Procedure)
To maintain the principle of Git as the single source of truth:

1. **Revert the Release Commit** on the `production` branch:
   ```bash
   git checkout production
   git pull origin production
   git revert HEAD --no-edit
   git push origin production
   ```
2. **Re-Enable Auto-Sync**:
   ```bash
   argocd app set llmops-prod --sync-policy auto --auto-prune --self-heal
   ```
3. Argo CD detects the commit and immediately synchronizes the previous immutable image SHA and model manifest.

## 4. Model Version Rollback
If the API service is healthy but the newly promoted model regresses on quality benchmarks:
1. Update `deployment/helm/llmops/values-prod.yaml`:
   ```yaml
   api:
     env:
       ACTIVE_MODEL_VERSION: "previous-approved-model-v0"
   ```
2. Commit and push:
   ```bash
   git commit -am "fix(model): rollback active model version to previous-approved-model-v0"
   git push origin production
   ```
3. Argo CD reconciles the configmap, triggering a rolling restart of the API pods with zero downtime.
