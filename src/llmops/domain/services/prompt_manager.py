"""Prompt management service for versioned prompt loading and formatting."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from llmops.domain.exceptions import ConfigurationError


class PromptDefinition(BaseModel):
    """Schema for version-controlled prompt YAML files."""

    name: str
    version: str
    description: str = ""
    template: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptManager:
    """Loads and formats version-controlled prompts from the prompts/ directory."""

    def __init__(self, prompts_base_dir: str | Path = "prompts") -> None:
        self.base_dir = Path(prompts_base_dir)
        self._cache: dict[str, PromptDefinition] = {}

    def get_prompt(self, category: str, name: str) -> PromptDefinition:
        """Load and cache a prompt definition from prompts/{category}/{name}.yaml."""
        cache_key = f"{category}/{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt_file = self.base_dir / category / f"{name}.yaml"
        if not prompt_file.exists():
            raise ConfigurationError(f"Prompt template not found at '{prompt_file}'")

        try:
            raw_data = yaml.safe_load(prompt_file.read_text(encoding="utf-8"))
            prompt_def = PromptDefinition.model_validate(raw_data)
            self._cache[cache_key] = prompt_def
            return prompt_def
        except Exception as exc:  # noqa: BLE001
            raise ConfigurationError(
                f"Failed to load prompt template at '{prompt_file}': {exc}"
            ) from exc

    def render(self, category: str, name: str, **kwargs: Any) -> tuple[str, str]:
        """Render a prompt template with keyword arguments.

        Returns:
            (rendered_text, prompt_version)
        """
        prompt_def = self.get_prompt(category, name)
        try:
            rendered = prompt_def.template.format(**kwargs)
            return rendered, prompt_def.version
        except KeyError as err:
            raise ConfigurationError(
                f"Missing parameter '{err}' when rendering prompt '{prompt_def.name}'"
            ) from err
