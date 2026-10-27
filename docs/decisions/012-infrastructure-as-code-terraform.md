# ADR 012: Infrastructure as Code Architecture (Terraform)

## Status
Accepted

## Context
Deploying the LLMOps platform across environments requires declarative provisioning of the infrastructure boundary (Section 37) without hard-coding cloud credentials or embedding internal application configurations. The infrastructure must provision networking, object storage, and Kubernetes clusters with dedicated CPU and GPU node pools, while preserving a zero-cloud local development path.

## Decision
1. **Modular Infrastructure Boundary**:
   - Create reusable, parameter-driven Terraform modules under `deployment/terraform/modules/`:
     - `networking`: Multi-AZ VPC, isolated public/private subnets, and routing.
     - `storage`: S3/object storage bucket with AES-256 server-side encryption, versioning, and public access blocks.
     - `kubernetes`: Managed Kubernetes control plane (EKS) with separate CPU and GPU node groups. GPU node pools include `nvidia.com/gpu` taints and `accelerator = nvidia-gpu` labels.
2. **Environment Root Separation**:
   - `deployment/terraform/environments/dev/`: Lightweight infrastructure boundary for local/development.
   - `deployment/terraform/environments/prod/`: Full production topology spanning 3 AZs with 3-15 CPU nodes, dedicated GPU inference pools, and persistent object storage.
3. **Zero Hardcoded Secrets**:
   - Provider authentication delegates strictly to IAM roles, OIDC federation, or environment variables. No credentials are stored in `.tf` or `.tfvars` files.

## Consequences
- Full compliance with Section 37 of the master specification.
- Clean separation between infrastructure boundary (Terraform) and application deployment (Helm & Argo CD).
- Seamless reproducibility across development and production environments.
