"""Core inference endpoints: multi-turn chat, generation, and vector retrieval (Section 26)."""

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status

from llmops.application.use_cases.rag_service import RAGService
from llmops.application.use_cases.retrieval_service import RetrievalService
from llmops.domain.interfaces.llm import BaseLLMClient
from llmops.domain.models.rag import RAGRequest
from llmops.domain.models.retrieval import RetrievalFilter, RetrievalQuery
from llmops.interfaces.api.schemas import (
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
    RetrievedChunkResponse,
    RetrieveRequest,
    RetrieveResponse,
    SourceCitationResponse,
)
from llmops.interfaces.api.security import Role, require_roles

router = APIRouter(prefix="/v1", tags=["Inference & Retrieval"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_roles([Role.VIEWER, Role.DEVELOPER]))],
)
def chat_endpoint(request_body: ChatRequest, request: Request) -> ChatResponse:
    """Multi-turn grounded conversation endpoint supporting RAG with citations."""
    if not request_body.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chat request must contain at least one message.",
        )

    last_user_message = next(
        (m.content for m in reversed(request_body.messages) if m.role == "user"),
        None,
    )
    if not last_user_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No user message found in conversation.",
        )

    rag_service: RAGService | None = getattr(request.app.state, "rag_service", None)
    llm_client: BaseLLMClient = request.app.state.llm_client
    run_id = f"chat-{uuid.uuid4().hex[:8]}"
    model_version = str(
        request_body.model or getattr(request.app.state, "active_model_version", "gemma-3-lora-v1")
    )

    if request_body.use_rag and rag_service:
        # Grounded RAG flow
        rag_resp = rag_service.answer_question(RAGRequest(question=last_user_message))
        sources = [
            SourceCitationResponse(
                document_id=c.document_id,
                title=c.title,
                url=c.url,
                chunk_id=c.chunk_id,
            )
            for c in rag_resp.sources
        ]
        return ChatResponse(
            answer=rag_resp.answer,
            model_version=model_version,
            run_id=run_id,
            sources=sources,
        )

    # Raw conversational generation
    formatted_messages = [{"role": m.role, "content": m.content} for m in request_body.messages]
    raw_answer = llm_client.generate(
        messages=formatted_messages,
        temperature=request_body.temperature,
        max_tokens=request_body.max_tokens,
    )
    return ChatResponse(
        answer=raw_answer,
        model_version=model_version,
        run_id=run_id,
        sources=[],
    )


@router.post(
    "/generate",
    response_model=GenerateResponse,
    dependencies=[Depends(require_roles([Role.DEVELOPER]))],
)
def generate_endpoint(request_body: GenerateRequest, request: Request) -> GenerateResponse:
    """Single prompt raw generation endpoint."""
    llm_client: BaseLLMClient = request.app.state.llm_client
    messages = [{"role": "user", "content": request_body.prompt}]
    text = llm_client.generate(
        messages=messages,
        temperature=request_body.temperature,
        max_tokens=request_body.max_tokens,
    )
    gen_model = str(
        request_body.model or getattr(request.app.state, "active_model_version", "gemma-3-lora-v1")
    )
    return GenerateResponse(
        text=text,
        model_version=gen_model,
    )


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    dependencies=[Depends(require_roles([Role.DEVELOPER]))],
)
def retrieve_endpoint(request_body: RetrieveRequest, request: Request) -> RetrieveResponse:
    """Direct vector search endpoint returning top chunks with provenance."""
    retrieval_service: RetrievalService | None = getattr(
        request.app.state, "retrieval_service", None
    )
    if not retrieval_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval service is not initialized on this instance.",
        )

    start_t = time.perf_counter()
    filt = (
        RetrievalFilter(source_name=request_body.source_name) if request_body.source_name else None
    )
    query = RetrievalQuery(
        query_text=request_body.query,
        top_k=request_body.top_k,
        filter_criteria=filt,
    )
    retrieved = retrieval_service.retrieve(query)
    latency = (time.perf_counter() - start_t) * 1000.0

    chunk_responses = [
        RetrievedChunkResponse(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            content=r.content,
            score=round(r.score, 4),
            source_name=r.provenance.source_name,
            source_url=r.provenance.source_url,
            section_heading=r.section_heading,
        )
        for r in retrieved
    ]
    return RetrieveResponse(
        query=request_body.query,
        chunks=chunk_responses,
        latency_ms=round(latency, 2),
    )
