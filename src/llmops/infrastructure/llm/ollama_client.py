"""Ollama client adapter for local developer LLM inference."""

import json
import logging
from collections.abc import Sequence
from typing import TypeVar

import httpx
from pydantic import BaseModel

from llmops.domain.exceptions import ModelInferenceError
from llmops.domain.interfaces.llm import BaseLLMClient

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class OllamaClient(BaseLLMClient):
    """Client communicating with local Ollama inference server."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma3:1b",
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._model = model
        self.timeout = timeout
        self._injected_client = client

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self) -> httpx.Client:
        if self._injected_client is not None:
            return self._injected_client
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Call Ollama /api/chat endpoint."""
        client = self._get_client()
        payload = {
            "model": self._model,
            "messages": list(messages),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            if resp.status_code != 200:
                raise ModelInferenceError(
                    f"Ollama generation failed ({resp.status_code}): {resp.text}"
                )
            data = resp.json()
            message_obj = data.get("message", {})
            return str(message_obj.get("content", ""))
        except httpx.HTTPError as err:
            raise ModelInferenceError(f"HTTP error communicating with Ollama: {err}") from err
        finally:
            if self._injected_client is None:
                client.close()

    def generate_structured(
        self,
        messages: Sequence[dict[str, str]],
        schema: type[T],
        temperature: float = 0.0,
    ) -> T:
        """Call Ollama with JSON format constraint and parse into Pydantic schema."""
        client = self._get_client()
        payload = {
            "model": self._model,
            "messages": list(messages),
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
            },
        }

        try:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            if resp.status_code != 200:
                raise ModelInferenceError(
                    f"Ollama structured generation failed ({resp.status_code}): {resp.text}"
                )
            data = resp.json()
            raw_content = data.get("message", {}).get("content", "{}")
            return schema.model_validate_json(raw_content)
        except (httpx.HTTPError, json.JSONDecodeError, Exception) as err:
            raise ModelInferenceError(f"Failed to generate structured response: {err}") from err
        finally:
            if self._injected_client is None:
                client.close()
