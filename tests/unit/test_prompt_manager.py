"""Unit tests for the PromptManager service."""

from pathlib import Path

import pytest

from llmops.domain.exceptions import ConfigurationError
from llmops.domain.services.prompt_manager import PromptManager


def test_prompt_manager_loads_system_prompt() -> None:
    """Verify loading and caching the developer_assistant system prompt."""
    manager = PromptManager()
    prompt_def = manager.get_prompt("system", "developer_assistant")
    assert prompt_def.name == "developer_assistant"
    assert prompt_def.version == "1.0.0"
    assert "FastAPI" in prompt_def.template

    # Verify cache hit
    prompt_def_cached = manager.get_prompt("system", "developer_assistant")
    assert prompt_def is prompt_def_cached


def test_prompt_manager_renders_rag_prompt() -> None:
    """Verify rendering rag_grounded_answer prompt with context and question."""
    manager = PromptManager()
    text, version = manager.render(
        category="rag",
        name="rag_grounded_answer",
        context="Sample context content",
        question="How do I declare a FastAPI route?",
    )
    assert version == "1.0.0"
    assert "Sample context content" in text
    assert "How do I declare a FastAPI route?" in text


def test_prompt_manager_missing_variable_raises() -> None:
    """Verify missing required template variables raises ConfigurationError."""
    manager = PromptManager()
    with pytest.raises(ConfigurationError, match="Missing parameter"):
        manager.render("rag", "rag_grounded_answer", context="Only context")


def test_prompt_manager_nonexistent_prompt_raises(tmp_path: Path) -> None:
    """Verify missing prompt file raises ConfigurationError."""
    manager = PromptManager(prompts_base_dir=tmp_path)
    with pytest.raises(ConfigurationError, match="not found"):
        manager.get_prompt("system", "non_existent")
