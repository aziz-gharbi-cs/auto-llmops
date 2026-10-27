"""Mock LLM client adapter for deterministic testing and offline CI."""

from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from llmops.domain.interfaces.llm import BaseLLMClient

T = TypeVar("T", bound=BaseModel)


class MockLLMClient(BaseLLMClient):
    """Deterministic mock client for testing RAG and generation pipelines."""

    def __init__(
        self,
        model_name: str = "mock-gemma-3-1b",
        default_response: str = "This is a grounded mock answer citing [c1].",
        structured_payload: dict[str, object] | None = None,
    ) -> None:
        self._model_name = model_name
        self.default_response = default_response
        self.structured_payload = structured_payload or {}
        self.recorded_calls: list[list[dict[str, str]]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        self.recorded_calls.append(list(messages))
        return self.default_response

    def generate_structured(
        self,
        messages: Sequence[dict[str, str]],
        schema: type[T],
        temperature: float = 0.0,
    ) -> T:
        self.recorded_calls.append(list(messages))
        return schema.model_validate(self.structured_payload)
