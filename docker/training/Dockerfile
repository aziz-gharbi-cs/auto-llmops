# GPU Training Container for Gemma 3 LoRA Fine-Tuning (Section 10/32)

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install transformers peft trl datasets accelerate mlflow pyyaml

COPY src/ /workspace/src/
COPY configs/ /workspace/configs/

ENTRYPOINT ["python", "-m", "llmops.interfaces.cli", "train", "--profile", "full"]
