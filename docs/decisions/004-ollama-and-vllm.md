# ADR 004: Dual Runtime Strategy --- Ollama for Local Dev and vLLM for Production

## Status
Accepted

## Context
Serving open-weights models (such as Google's Gemma 3) requires balancing developer productivity on laptops against high-throughput batching on production GPU clusters. Developer laptops often lack CUDA GPUs or have memory constraints, whereas production serving demands continuous batching, PagedAttention, and OpenAI-compatible endpoints.

## Decision
We establish a two-tier LLM runtime strategy decoupled behind `BaseLLMClient`:
1. **Development Runtime (Ollama)**:
   - Used for local developer testing, smoke tests, and CPU-first workflows.
   - Low resource overhead, simple binary/container distribution, easy local model pulls (`ollama pull gemma3:1b`).
   - Implemented via `OllamaClient`.
2. **Production GPU Runtime (vLLM)**:
   - Dedicated serving engine for production Kubernetes deployments with GPUs.
   - High-throughput continuous batching, tensor parallelism, and OpenAI-compatible `/v1/chat/completions`.
   - The FastAPI backend acts as an orchestration/API gateway layer, delegating model execution to vLLM.
   - Implemented via `VLLMClient`.
3. **Testing / Mock Runtime (`MockLLMClient`)**:
   - Zero-dependency client for deterministic unit tests and CI pipelines.

## Alternatives Considered
- **Ollama as the only Serving Layer in Production**: Rejected because Ollama lacks production-grade continuous batching, multi-GPU scaling, and advanced tensor parallel optimizations offered by vLLM.
- **Loading Hugging Face Transformers directly inside the FastAPI API process**: Creates severe GIL contention, GPU memory fragmentation across worker processes, and couples application server lifecycle to model memory allocation.

## Consequences
- Developers can run end-to-end local generation without GPU hardware.
- Production serving scales independently from application API workers.
- Interchangeable clients ensure business logic never depends on serving engine specifics.
