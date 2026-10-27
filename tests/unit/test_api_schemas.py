"""Unit tests for FastAPI Pydantic request/response schemas (Section 26)."""

from llmops.domain.models.curation import ChatMessage
from llmops.interfaces.api.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    GenerateRequest,
    HealthResponse,
    ModelsListResponse,
    ReadinessResponse,
    RetrieveRequest,
    SourceCitationResponse,
    VersionResponse,
)


def test_health_and_readiness_schemas() -> None:
    health = HealthResponse()
    assert health.status == "ok"
    assert health.timestamp is not None

    ready = ReadinessResponse(
        status="ready",
        checks={"storage": True, "vector_store": True},
        details={"storage": "ok"},
    )
    assert ready.status == "ready"
    assert ready.checks["vector_store"] is True


def test_version_schema() -> None:
    ver = VersionResponse(
        version="0.1.0",
        git_commit="abcdef123456",
        active_model="gemma-3-lora-v1",
    )
    assert ver.version == "0.1.0"
    assert ver.environment == "production"


def test_chat_schemas() -> None:
    req = ChatRequest(
        messages=[
            ChatMessage(role="user", content="How do I use FastAPI dependencies?"),
        ],
        use_rag=True,
    )
    assert len(req.messages) == 1
    assert req.use_rag is True

    sources = [
        SourceCitationResponse(
            document_id="doc-1",
            title="FastAPI Docs",
            url="https://fastapi.tiangolo.com",
            chunk_id="chk-1",
        )
    ]
    resp = ChatResponse(
        answer="Use Depends() in path operations.",
        model_version="gemma-3-lora-v1",
        run_id="chat-123",
        sources=sources,
    )
    assert len(resp.sources) == 1
    assert resp.sources[0].chunk_id == "chk-1"


def test_generate_and_retrieve_schemas() -> None:
    gen_req = GenerateRequest(prompt="Write a Python hello world function.")
    assert gen_req.temperature == 0.2

    ret_req = RetrieveRequest(query="FastAPI APIRouter", top_k=3)
    assert ret_req.top_k == 3


def test_models_list_schema() -> None:
    resp = ModelsListResponse(models=[], active_model="gemma-3-lora-v1")
    assert resp.active_model == "gemma-3-lora-v1"
    assert resp.models == []


def test_feedback_schema() -> None:
    fb = FeedbackRequest(query_id="chat-123", rating=5, comment="Spot on!")
    assert fb.rating == 5
    assert fb.comment == "Spot on!"
