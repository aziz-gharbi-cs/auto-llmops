# ADR 001: Python 3.13 Runtime and uv Package Manager

## Status
Accepted

## Context
The LLMOps platform requires a consistent, modern, and high-performance Python environment across development, training, evaluation, and production serving. Historically, Python tooling suffered from fragmented dependency managers (pip, virtualenv, poetry, pipenv) resulting in slow installs, non-deterministic resolutions, and friction across environments.

## Decision
We adopt **Python 3.13** as the primary runtime and **uv** (by Astral) as the unified dependency and environment manager:
- `pyproject.toml` is the single source of truth for project metadata, dependencies, and tool settings.
- Standard dependency groups (`dev`, `ml`, `orchestration`, `observability`) are managed using PEP 735 standard groups.
- `ruff` is selected for formatting and linting.
- `mypy` is selected for static type checking in strict mode.

## Alternatives Considered
- **Poetry**: Slower resolver, divergent dependency group syntax, heavier runtime overhead.
- **Pipenv**: Outdated dependency resolution mechanics, high latency in CI.
- **Standard pip + requirements.txt**: Lacks unified lockfile integrity and group isolation without fragile workarounds.

## Consequences
- Fast, reproducible dependency resolution in local development and GitHub Actions CI.
- Single command workflows (`uv sync`, `uv run`).
- Requires developers and CI pipelines to have `uv` installed.
