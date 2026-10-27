"""Synthetic Q&A generator creating training candidates from canonical document chunks."""

import logging

from llmops.domain.interfaces.llm import BaseLLMClient
from llmops.domain.models.curation import SyntheticQACandidate
from llmops.domain.models.document import DocumentChunk
from llmops.domain.services.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class SyntheticQAGenerator:
    """Generates synthetic instruction and response pairs from canonical chunks."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_manager: PromptManager | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager or PromptManager()

    def generate_candidate(self, chunk: DocumentChunk) -> tuple[SyntheticQACandidate, str]:
        """Generate a candidate Q&A pair from a document chunk.

        Returns:
            (candidate, prompt_version)
        """
        prompt_text, prompt_version = self.prompt_manager.render(
            category="generation",
            name="synthetic_qa",
            source_name=chunk.provenance.source_name,
            section_heading=chunk.section_heading or "Overview",
            chunk_text=chunk.content,
        )

        messages = [{"role": "user", "content": prompt_text}]

        try:
            candidate = self.llm_client.generate_structured(
                messages=messages,
                schema=SyntheticQACandidate,
                temperature=0.7,
            )
            return candidate, prompt_version
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Structured generation failed (%s). Generating heuristic fallback.",
                exc,
            )
            # Heuristic fallback for offline/mock scenarios
            heading = chunk.section_heading or "general concepts"
            source = chunk.provenance.source_name
            question = f"How do I work with {heading} in {source}?"
            fallback_candidate = SyntheticQACandidate(
                instruction=question,
                response=chunk.content,
            )
            return fallback_candidate, prompt_version
