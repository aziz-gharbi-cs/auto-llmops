# Dedicated vLLM Production Inference Container (Section 30/32)
# GPU-accelerated serving engine decoupled from application API gateway

FROM vllm/vllm-openai:v0.7.3

WORKDIR /workspace

ENV MODEL_NAME="google/gemma-3-1b-it" \
    MAX_MODEL_LEN=4096 \
    GPU_MEMORY_UTILIZATION=0.90 \
    PORT=8000

EXPOSE 8000

# Health check against OpenAI-compatible model listing endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/v1/models || exit 1

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", \
            "--model", "google/gemma-3-1b-it", \
            "--host", "0.0.0.0", \
            "--port", "8000", \
            "--dtype", "bfloat16", \
            "--max-model-len", "4096"]
