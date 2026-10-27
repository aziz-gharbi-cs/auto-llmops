"""Unit tests for Ollama, vLLM, and Mock LLM client adapters."""

import httpx
from pydantic import BaseModel

from llmops.infrastructure.llm.mock_llm import MockLLMClient
from llmops.infrastructure.llm.ollama_client import OllamaClient
from llmops.infrastructure.llm.vllm_client import VLLMClient


class SampleStructuredOutput(BaseModel):
    summary: str
    confidence: float


def test_mock_llm_client() -> None:
    """Verify MockLLMClient generates text and records calls."""
    client = MockLLMClient(
        model_name="test-gemma",
        default_response="Mocked response text.",
        structured_payload={"summary": "test", "confidence": 0.99},
    )
    assert client.model_name == "test-gemma"

    res = client.generate([{"role": "user", "content": "hi"}])
    assert res == "Mocked response text."
    assert len(client.recorded_calls) == 1

    structured = client.generate_structured(
        [{"role": "user", "content": "structure this"}],
        schema=SampleStructuredOutput,
    )
    assert structured.summary == "test"
    assert structured.confidence == 0.99


def test_ollama_client_generate() -> None:
    """Verify OllamaClient translates messages and parses /api/chat response."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Ollama generated answer."}},
        )

    mock_http = httpx.Client(transport=httpx.MockTransport(handler))
    client = OllamaClient(client=mock_http)

    res = client.generate([{"role": "user", "content": "Hello"}])
    assert res == "Ollama generated answer."


def test_vllm_client_generate() -> None:
    """Verify VLLMClient translates messages and parses /chat/completions response."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        msg = {"role": "assistant", "content": "vLLM generated answer."}
        return httpx.Response(200, json={"choices": [{"message": msg}]})

    mock_http = httpx.Client(transport=httpx.MockTransport(handler))
    client = VLLMClient(client=mock_http)

    res = client.generate([{"role": "user", "content": "Hello vLLM"}])
    assert res == "vLLM generated answer."
