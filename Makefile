.PHONY: help install dev test lint typecheck format check up down logs ingest embed evaluate train clean

PYTHON ?= python
UV ?= uv

help:
	@echo "Available commands:"
	@echo "  make install      Install all dependencies using uv"
	@echo "  make dev          Run API server in development mode"
	@echo "  make test         Run test suite with pytest and coverage"
	@echo "  make lint         Run Ruff linter"
	@echo "  make typecheck    Run Mypy static type checker"
	@echo "  make format       Format code using Ruff"
	@echo "  make check        Run full local quality gate (lint + typecheck + test)"
	@echo "  make up           Start core Docker Compose services"
	@echo "  make down         Stop Docker Compose services"
	@echo "  make logs         Tail Docker Compose logs"
	@echo "  make ingest       Run source documentation ingestion pipeline"
	@echo "  make embed        Build embeddings and Qdrant index"
	@echo "  make evaluate     Run evaluation suite"
	@echo "  make train        Run LoRA fine-tuning pipeline"
	@echo "  make clean        Remove build artifacts and caches"

install:
	$(UV) sync --all-groups

dev:
	$(PYTHON) -m uvicorn llmops.interfaces.api.app:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PYTHON) -m pytest tests/unit

lint:
	$(PYTHON) -m ruff check src tests

typecheck:
	$(PYTHON) -m mypy src tests

format:
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

check: lint typecheck test
	@echo "All quality checks passed!"

up:
	docker compose --profile core up -d

down:
	docker compose --profile core down

logs:
	docker compose logs -f

ingest:
	$(PYTHON) -m llmops.interfaces.cli ingest

embed:
	$(PYTHON) -m llmops.interfaces.cli embed

evaluate:
	$(PYTHON) -m llmops.interfaces.cli evaluate

train:
	$(PYTHON) -m llmops.interfaces.cli train

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov build dist *.egg-info
