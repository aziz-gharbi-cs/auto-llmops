"""vLLM client adapter communicating with production OpenAI-compatible serving."""

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


class VLLMClient(BaseLLMClient):
    """Client for production vLLM OpenAI-compatible serving engine."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "google/gemma-3-1b-it",
        api_key: str | None = None,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._model = model
        self.api_key = api_key
        self.timeout = timeout
        self._injected_client = client

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self) -> httpx.Client:
        if self._injected_client is not None:
            return self._injected_client
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return httpx.Client(base_url=self.base_url, headers=headers, timeout=self.timeout)

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Call vLLM /chat/completions endpoint."""
        client = self._get_client()
        payload = {
            "model": self._model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload)
            if resp.status_code != 200:
                raise ModelInferenceError(
                    f"vLLM generation failed ({resp.status_code}): {resp.text}"
                )
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return ""
            return str(choices[0].get("message", {}).get("content", ""))
        except httpx.HTTPError as err:
            raise ModelInferenceError(f"HTTP error communicating with vLLM: {err}") from err
        finally:
            if self._injected_client is None:
                client.close()

    def generate_structured(
        self,
        messages: Sequence[dict[str, str]],
        schema: type[T],
        temperature: float = 0.0,
    ) -> T:
        """Call vLLM with json_object response format constraint and parse into schema."""
        client = self._get_client()
        payload = {
            "model": self._model,
            "messages": list(messages),
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        try:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload)
            if resp.status_code != 200:
                raise ModelInferenceError(
                    f"vLLM structured generation failed ({resp.status_code}): {resp.text}"
                )
            data = resp.json()
            raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            return schema.model_validate_json(raw_content)
        except (httpx.HTTPError, json.JSONDecodeError, Exception) as err:
            raise ModelInferenceError(f"Failed to generate structured response: {err}") from err
        finally:
            if self._injected_client is None:
                client.close()
