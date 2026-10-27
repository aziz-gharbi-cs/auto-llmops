"""Contract test for BaseLLMClient implementations."""

import pytest
from pydantic import BaseModel

from llmops.domain.interfaces.llm import BaseLLMClient
from llmops.infrastructure.llm.mock_llm import MockLLMClient


class TargetSchema(BaseModel):
    name: str
    score: int


@pytest.fixture
def llm_client() -> BaseLLMClient:
    """Fixture providing a BaseLLMClient instance."""
    return MockLLMClient(
        model_name="contract-model",
        default_response="Contract verification response.",
        structured_payload={"name": "fastapi", "score": 10},
    )


def test_llm_client_fulfills_contract(llm_client: BaseLLMClient) -> None:
    """Verify BaseLLMClient satisfies the generation contract."""
    # 1. model_name property
    assert isinstance(llm_client.model_name, str)
    assert len(llm_client.model_name) > 0

    # 2. Text generation
    messages = [{"role": "user", "content": "Hello"}]
    answer = llm_client.generate(messages, temperature=0.1, max_tokens=100)
    assert isinstance(answer, str)
    assert len(answer) > 0

    # 3. Structured generation
    structured = llm_client.generate_structured(messages, schema=TargetSchema)
    assert isinstance(structured, TargetSchema)
    assert structured.name == "fastapi"
    assert structured.score == 10
