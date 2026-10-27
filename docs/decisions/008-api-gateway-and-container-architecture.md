# ADR 008: API Gateway Design, Reverse Proxy, and Isolated Containers

## Status
Accepted

## Context
Deploying full deep learning dependencies directly within the web serving container creates bloated image sizes (>15GB), exposes massive attack surfaces, causes Python GIL contention, and prevents independent autoscaling between inference engines and API gateways.

## Decision
1. **Three-Tier Architecture (Internet -> Nginx -> FastAPI -> vLLM)**:
   - **Nginx**: Handles TLS termination, connection keepalive, request body buffering (2MB limit), rate-limiting (20r/s per IP), and security headers.
   - **FastAPI**: Acts purely as the application logic and orchestration gateway. Handles request validation, RBAC, RAG context assembly, and feedback collection.
   - **vLLM**: Dedicated, isolated model serving cluster on internal networks, scaling with GPU resources.
2. **Container Specialization (Section 32)**:
   - `docker/api/Dockerfile`: Multi-stage Python 3.13-slim image (~150MB), non-root user `appuser` (UID 10001), no CUDA/PyTorch dependencies.
   - `docker/inference/Dockerfile`: Dedicated vLLM GPU inference container with bfloat16 Gemma 3 optimizations.
   - `docker/pipeline/Dockerfile`: Worker container for DVC/Prefect pipelines.
   - `docker/training/Dockerfile`: PyTorch / CUDA container for LoRA fine-tuning.
3. **Enterprise API Security (Section 27/28)**:
   - RBAC roles (`viewer`, `developer`, `ml_engineer`, `admin`) enforced via declarative route dependencies.
   - Global HTTP security headers (`nosniff`, `DENY`, `HSTS`, `CSP`).
   - Request tracing with unique `X-Request-ID` and latency headers.

## Consequences
- Fast deployment rollouts and minimal memory footprint for API gateway pods.
- Zero GPU dependency in the web tier.
- Complete container attack surface minimization adhering to least-privilege standards.
