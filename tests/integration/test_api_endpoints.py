"""Integration tests for all FastAPI gateway endpoints and middleware (Milestone 9)."""

from collections.abc import Coroutine
from typing import Any

import pytest
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from llmops.application.use_cases.rag_service import RAGService
from llmops.application.use_cases.retrieval_service import RetrievalService
from llmops.domain.models.curation import ChatMessage
from llmops.infrastructure.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from llmops.infrastructure.llm.mock_llm import MockLLMClient
from llmops.infrastructure.reranker.cross_encoder import CrossEncoderReranker
from llmops.infrastructure.vector.in_memory_store import InMemoryVectorStore
from llmops.interfaces.api.middleware import (
    RequestSizeLimitMiddleware,
    RequestTracingMiddleware,
    SecurityHeadersMiddleware,
)
from llmops.interfaces.api.routes import feedback, health, inference, models
from llmops.interfaces.api.schemas import (
    ChatRequest,
    FeedbackRequest,
    GenerateRequest,
    RetrieveRequest,
)


def run_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Drive a synchronous coroutine to completion without an event loop self-pipe."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value
    raise RuntimeError("Coroutine did not complete synchronously")


def create_mock_request(
    path: str = "/health", headers: list[tuple[bytes, bytes]] | None = None
) -> Request:
    """Create a mock Starlette Request with wired application state."""
    active_client = MockLLMClient(model_name="gemma-3-lora-v1")
    vector_store = InMemoryVectorStore()
    vector_store.create_collection("docs_fastapi_pydantic_sqlalchemy", vector_size=384)

    # Seed a sample chunk so retrieval succeeds
    from llmops.domain.models.document import ChunkProvenance, DocumentChunk

    text = "Pydantic models are defined by subclassing BaseModel and defining typed fields."
    sample_chunk = DocumentChunk(
        chunk_id="chunk-test-1",
        document_id="doc-test-1",
        content=text,
        token_count=15,
        provenance=ChunkProvenance(
            document_id="doc-test-1",
            document_version="1.0.0",
            chunk_id="chunk-test-1",
            source_name="pydantic",
            source_url="https://docs.pydantic.dev",
            ingestion_run_id="run-1",
            content_hash="test-hash-1",
            text=text,
        ),
    )
    vector_store.upsert_chunks(
        "docs_fastapi_pydantic_sqlalchemy",
        [sample_chunk],
        [[0.1] * 384],
    )

    active_retrieval = RetrievalService(
        vector_store=vector_store,
        embedding_model=SentenceTransformerEmbeddings(use_fallback=True),
        reranker=CrossEncoderReranker(use_fallback=True),
    )
    active_rag = RAGService(
        retrieval_service=active_retrieval,
        llm_client=active_client,
    )

    scope: dict[str, Any] = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": headers or [],
        "app": type(
            "MockApp",
            (),
            {
                "state": type(
                    "State",
                    (),
                    {
                        "llm_client": active_client,
                        "retrieval_service": active_retrieval,
                        "rag_service": active_rag,
                        "active_model_version": "gemma-3-lora-v1",
                        "environment": "production",
                    },
                )()
            },
        )(),
    }
    return Request(scope)


def test_api_health_and_readiness_endpoints() -> None:
    req = create_mock_request("/health")

    # Health probe
    resp_health = health.health_probe()
    assert resp_health.status == "ok"
    assert resp_health.timestamp is not None

    # Readiness probe
    resp_ready = health.readiness_probe(req)
    assert resp_ready.status == "ready"
    assert resp_ready.checks["inference_runtime"] is True
    assert resp_ready.checks["storage"] is True

    # Version endpoint
    resp_ver = health.system_version(req)
    assert resp_ver.version is not None
    assert resp_ver.active_model == "gemma-3-lora-v1"


def test_api_security_headers_and_tracing() -> None:
    async def dummy_app(scope: Any, receive: Any, send: Any) -> None:
        pass

    async def dummy_call_next(request: Request) -> Response:
        return Response(content="ok", media_type="text/plain")

    # Security headers middleware
    sec_mw = SecurityHeadersMiddleware(app=dummy_app)
    req = create_mock_request("/health")
    resp_sec = run_sync(sec_mw.dispatch(req, dummy_call_next))

    assert resp_sec.headers["X-Content-Type-Options"] == "nosniff"
    assert resp_sec.headers["X-Frame-Options"] == "DENY"
    assert "Strict-Transport-Security" in resp_sec.headers
    assert resp_sec.headers["Content-Security-Policy"] == "default-src 'self'"
    assert resp_sec.headers["Referrer-Policy"] == "no-referrer"

    # Verify Swagger docs route has relaxed CSP for CDN stylesheets and scripts
    req_docs = create_mock_request("/docs")
    resp_docs = run_sync(sec_mw.dispatch(req_docs, dummy_call_next))
    assert "cdn.jsdelivr.net" in resp_docs.headers["Content-Security-Policy"]
    assert "unsafe-inline" in resp_docs.headers["Content-Security-Policy"]

    # Request tracing middleware
    trace_mw = RequestTracingMiddleware(app=dummy_app)
    resp_trace = run_sync(trace_mw.dispatch(req, dummy_call_next))
    assert "X-Request-ID" in resp_trace.headers
    assert "X-Response-Time-Ms" in resp_trace.headers


def test_api_request_size_limiting() -> None:
    async def dummy_app(scope: Any, receive: Any, send: Any) -> None:
        pass

    async def dummy_call_next(request: Request) -> Response:
        return Response(content="ok", media_type="text/plain")

    # 3 MB exceeds 2 MB limit
    size_mw = RequestSizeLimitMiddleware(app=dummy_app, max_size_bytes=2 * 1024 * 1024)
    req_oversized = create_mock_request(
        path="/v1/chat",
        headers=[(b"content-length", str(3 * 1024 * 1024).encode("utf-8"))],
    )
    resp_oversized = run_sync(size_mw.dispatch(req_oversized, dummy_call_next))
    assert resp_oversized.status_code == 413

    # Normal size is allowed
    req_normal = create_mock_request(
        path="/v1/chat",
        headers=[(b"content-length", b"1024")],
    )
    resp_normal = run_sync(size_mw.dispatch(req_normal, dummy_call_next))
    assert resp_normal.status_code == 200


def test_chat_endpoint_grounded_rag() -> None:
    req = create_mock_request("/v1/chat")
    chat_req = ChatRequest(
        messages=[
            ChatMessage(role="user", content="How do I define schemas in Pydantic?"),
        ],
        use_rag=True,
    )
    resp = inference.chat_endpoint(request_body=chat_req, request=req)

    assert resp.answer != ""
    assert resp.model_version == "gemma-3-lora-v1"
    assert resp.run_id != ""
    assert isinstance(resp.sources, list)


def test_chat_endpoint_without_rag() -> None:
    req = create_mock_request("/v1/chat")
    chat_req = ChatRequest(
        messages=[
            ChatMessage(role="user", content="What is Python?"),
        ],
        use_rag=False,
    )
    resp = inference.chat_endpoint(request_body=chat_req, request=req)

    assert resp.answer != ""
    assert resp.sources == []


def test_generate_endpoint() -> None:
    req = create_mock_request("/v1/generate")
    gen_req = GenerateRequest(prompt="Write a test function", max_tokens=100)
    resp = inference.generate_endpoint(request_body=gen_req, request=req)

    assert resp.text != ""
    assert resp.model_version == "gemma-3-lora-v1"


def test_retrieve_endpoint() -> None:
    req = create_mock_request("/v1/retrieve")
    ret_req = RetrieveRequest(query="FastAPI dependencies", top_k=3)
    resp = inference.retrieve_endpoint(request_body=ret_req, request=req)

    assert resp.query == "FastAPI dependencies"
    assert isinstance(resp.chunks, list)
    assert resp.latency_ms >= 0


def test_models_registry_endpoints() -> None:
    req = create_mock_request("/v1/models")

    # List models
    list_resp = models.list_models(request=req)
    assert len(list_resp.models) >= 1
    assert list_resp.active_model == "gemma-3-lora-v1"

    # Detail endpoint
    detail_resp = models.get_model_details(version="gemma-3-lora-v1", request=req)
    assert detail_resp.model_version == "gemma-3-lora-v1"
    assert detail_resp.approved is True

    # Unknown model -> 404
    with pytest.raises(HTTPException) as exc_info:
        models.get_model_details(version="unknown-model-xyz", request=req)
    assert exc_info.value.status_code == 404


def test_feedback_endpoint() -> None:
    fb_req = FeedbackRequest(
        query_id="chat-test-001",
        rating=5,
        comment="Accurate response with valid citations.",
    )
    resp = feedback.submit_feedback(request_body=fb_req)

    assert resp.status == "received"
    assert resp.feedback_id.startswith("fb-")
