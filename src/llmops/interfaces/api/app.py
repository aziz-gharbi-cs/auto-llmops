"""FastAPI application factory with middleware, routers, and dependency configuration."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llmops import __version__
from llmops.application.use_cases.rag_service import RAGService
from llmops.application.use_cases.retrieval_service import RetrievalService
from llmops.domain.interfaces.llm import BaseLLMClient
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


def create_app(
    llm_client: BaseLLMClient | None = None,
    retrieval_service: RetrievalService | None = None,
    rag_service: RAGService | None = None,
    active_model_version: str = "gemma-3-lora-v1",
    environment: str = "production",
    use_fallback: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="Antigravity LLMOps Platform API",
        description="Production API Gateway for documentation-grounded developer assistance.",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 1. Register Middlewares (Order: Outer to Inner)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware, max_size_bytes=2 * 1024 * 1024)
    app.add_middleware(RequestTracingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # 2. Dependency Wiring
    active_client = llm_client or MockLLMClient(model_name=active_model_version)
    active_retrieval = retrieval_service or RetrievalService(
        vector_store=InMemoryVectorStore(),
        embedding_model=SentenceTransformerEmbeddings(use_fallback=use_fallback),
        reranker=CrossEncoderReranker(use_fallback=use_fallback),
    )
    active_rag = rag_service or RAGService(
        retrieval_service=active_retrieval,
        llm_client=active_client,
    )

    app.state.llm_client = active_client
    app.state.retrieval_service = active_retrieval
    app.state.rag_service = active_rag
    app.state.active_model_version = active_model_version
    app.state.environment = environment

    # 3. Register Routers
    app.include_router(health.router)
    app.include_router(inference.router)
    app.include_router(models.router)
    app.include_router(feedback.router)

    return app
