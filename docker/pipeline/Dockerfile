# Pipeline and Worker Container for Ingestion, DVC, and Evaluation (Section 32)

FROM python:3.13-slim

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install boto3 beautifulsoup4 httpx pydantic pyyaml prefect

COPY src/ /workspace/src/
COPY configs/ /workspace/configs/
COPY prompts/ /workspace/prompts/

ENTRYPOINT ["python", "-m", "llmops.interfaces.cli"]
