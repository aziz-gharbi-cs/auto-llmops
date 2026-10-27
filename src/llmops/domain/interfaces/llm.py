"""Abstract LLM client port for local Ollama and production vLLM serving."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseLLMClient(ABC):
    """Abstract interface for large language model generation providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the active model."""

    @abstractmethod
    def generate(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Generate response text given a sequence of conversation messages."""

    @abstractmethod
    def generate_structured(
        self,
        messages: Sequence[dict[str, str]],
        schema: type[T],
        temperature: float = 0.0,
    ) -> T:
        """Generate structured output adhering to a Pydantic schema."""
