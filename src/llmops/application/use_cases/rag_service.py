"""RAG service executing retrieval, prompt assembly, and grounded generation."""

import logging
import uuid

from llmops.application.use_cases.retrieval_service import RetrievalService
from llmops.domain.interfaces.llm import BaseLLMClient
from llmops.domain.models.rag import Citation, RAGRequest, RAGResponse
from llmops.domain.models.retrieval import RetrievalFilter, RetrievalQuery, RetrievedChunk
from llmops.domain.services.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

INSUFFICIENT_CONTEXT_MESSAGE = (
    "I do not have sufficient documentation evidence in the official framework documentation "
    "to answer this question accurately."
)


class RAGService:
    """Orchestrates end-to-end grounded question answering with verified citations."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_client: BaseLLMClient,
        prompt_manager: PromptManager | None = None,
        min_relevance_threshold: float = 0.05,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager or PromptManager()
        self.min_relevance_threshold = min_relevance_threshold

    def _build_context_block(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into numbered context blocks for prompt injection."""
        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.provenance.source_name
            heading = f" - {chunk.section_heading}" if chunk.section_heading else ""
            header = f"[{i}] (Chunk ID: {chunk.chunk_id} | Source: {source}{heading})"
            context_parts.append(f"{header}\n{chunk.content}")
        return "\n\n".join(context_parts)

    def answer_question(self, request: RAGRequest) -> RAGResponse:
        """Execute end-to-end RAG query."""
        run_id = f"rag_{uuid.uuid4().hex[:8]}"
        logger.info("Executing RAG query (run_id: %s): '%s'", run_id, request.question)

        # 1. Retrieve candidate chunks
        filter_criteria = None
        if request.source_name:
            filter_criteria = RetrievalFilter(source_name=request.source_name)

        query = RetrievalQuery(
            query_text=request.question,
            top_n=request.top_n,
            top_k=request.top_k,
            filter_criteria=filter_criteria,
        )
        retrieved_chunks = self.retrieval_service.retrieve(query)

        # 2. Check for sufficient evidence
        valid_chunks = [
            c
            for c in retrieved_chunks
            if (c.rerank_score is not None and c.rerank_score >= self.min_relevance_threshold)
            or (c.rerank_score is None and c.score >= self.min_relevance_threshold)
        ]

        if not valid_chunks:
            logger.info("Insufficient evidence retrieved for query '%s'", request.question)
            return RAGResponse(
                answer=INSUFFICIENT_CONTEXT_MESSAGE,
                model_version=self.llm_client.model_name,
                prompt_version=request.prompt_version,
                run_id=run_id,
                sources=[],
                has_sufficient_context=False,
            )

        # 3. Assemble context and load versioned prompts
        context_str = self._build_context_block(valid_chunks)
        system_text, _ = self.prompt_manager.render(
            category="system",
            name="developer_assistant",
        )
        user_prompt_text, prompt_version = self.prompt_manager.render(
            category="rag",
            name="rag_grounded_answer",
            context=context_str,
            question=request.question,
        )

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_prompt_text},
        ]

        # 4. Generate grounded answer
        answer = self.llm_client.generate(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # 5. Build citations
        citations: list[Citation] = []
        for chunk in valid_chunks:
            citations.append(
                Citation(
                    document_id=chunk.document_id,
                    title=chunk.section_heading,
                    url=chunk.provenance.source_url,
                    chunk_id=chunk.chunk_id,
                    section_heading=chunk.section_heading,
                    relevance_score=chunk.rerank_score
                    if chunk.rerank_score is not None
                    else chunk.score,
                )
            )

        return RAGResponse(
            answer=answer,
            model_version=self.llm_client.model_name,
            prompt_version=prompt_version,
            run_id=run_id,
            sources=citations,
            has_sufficient_context=True,
        )
