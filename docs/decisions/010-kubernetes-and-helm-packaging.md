# ADR 010: Kubernetes Deployment Architecture and Helm Packaging

## Status
Accepted

## Context
Production operations (Section 34-36) require scalable container orchestration with automated autoscaling, zero-downtime rolling updates, pod disruption budgets, resource isolation (GPU vs CPU nodes), and secure secrets management. While local development uses lightweight Docker Compose (Section 33), staging and production environments run on Kubernetes.

## Decision
1. **Helm Chart Structure**:
   - Package all platform deployment assets into `deployment/helm/llmops/`.
   - Separate concerns into clean templates:
     - `deployment-api.yaml`: FastAPI gateway with liveness (`/health`) and readiness (`/ready`) probes, non-root security context (`uid: 10001`), and environment configmaps/secrets.
     - `deployment-inference.yaml`: Dedicated model inference pods with GPU nodeSelectors (`accelerator: nvidia-gpu`) and GPU tolerations.
     - `service-api.yaml` & `service-inference.yaml`: ClusterIP network services.
     - `ingress.yaml`: Ingress with TLS termination and security annotations (`proxy-body-size: 2m`).
     - `hpa-api.yaml`: HorizontalPodAutoscaler scaling API replicas based on target CPU and memory thresholds.
     - `pdb-api.yaml`: PodDisruptionBudget ensuring minimum available replicas during node draining.
2. **Environment Parameterization**:
   - Provide explicit environment values files:
     - `values-dev.yaml`: Single replica, autoscaling disabled, CPU mock inference.
     - `values-staging.yaml`: 2 replicas, basic autoscaling, staging ingress.
     - `values-prod.yaml`: 3-15 replicas, strict HPA, GPU inference nodes, PDB minAvailable=2, production TLS ingress.
3. **Decoupling Application Internals from Infrastructure**:
   - Manifests consume cloud-agnostic primitives, allowing portable deployment across EKS, GKE, AKS, or bare-metal Kubernetes.

## Consequences
- Full compliance with Sections 34, 35, and 36 of the specification.
- Clean separation between local Docker Compose development and production Kubernetes deployments.
- Ready for automated GitOps reconciliation with Argo CD.
