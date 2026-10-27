"""LLM-as-a-Judge adapter for evaluating synthetic Q&A candidate quality."""

import logging
import re
from collections.abc import Sequence

from llmops.domain.interfaces.llm import BaseLLMClient
from llmops.domain.models.curation import JudgeScore
from llmops.domain.services.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class DataCurationJudge:
    """Evaluates synthetic Q&A pairs for inclusion in training datasets (Section 17/18)."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_manager: PromptManager | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager or PromptManager()

    @property
    def model_name(self) -> str:
        return self.llm_client.model_name

    def evaluate(
        self,
        context: str,
        instruction: str,
        response: str,
        threshold: float = 3.5,
    ) -> tuple[JudgeScore, str]:
        """Evaluate a generated Q&A pair against authoritative source context.

        Returns:
            (judge_score, prompt_version)
        """
        prompt_text, prompt_version = self.prompt_manager.render(
            category="judge",
            name="data_curation_judge",
            context=context,
            instruction=instruction,
            response=response,
            threshold=str(threshold),
        )

        messages: Sequence[dict[str, str]] = [{"role": "user", "content": prompt_text}]

        try:
            score = self.llm_client.generate_structured(
                messages=messages,
                schema=JudgeScore,
                temperature=0.0,
            )
            return score, prompt_version
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Structured judge evaluation failed (%s). Evaluating with deterministic fallback.",
                exc,
            )
            # Deterministic heuristic fallback for offline/mock environments
            return self._heuristic_fallback(
                context=context,
                instruction=instruction,
                response=response,
                threshold=threshold,
                prompt_version=prompt_version,
            )

    def _heuristic_fallback(
        self,
        context: str,
        instruction: str,
        response: str,
        threshold: float,
        prompt_version: str,
    ) -> tuple[JudgeScore, str]:
        """Deterministic heuristic fallback scoring based on token overlap and presence."""
        resp_tokens = set(re.findall(r"\w+", response.lower()))
        ctx_tokens = set(re.findall(r"\w+", context.lower()))

        grounded_overlap = len(resp_tokens.intersection(ctx_tokens))
        has_min_length = len(instruction.strip()) >= 10 and len(response.strip()) >= 15

        if has_min_length and grounded_overlap >= 2:
            score = JudgeScore(
                relevance=4,
                correctness=4,
                groundedness=5 if grounded_overlap >= 5 else 4,
                instruction_following=4,
                clarity=4,
                overall=4.0,
                is_pass=threshold <= 4.0,
                reason="Candidate passed heuristic fallback validation.",
            )
        else:
            score = JudgeScore(
                relevance=2,
                correctness=2,
                groundedness=2,
                instruction_following=2,
                clarity=2,
                overall=2.0,
                is_pass=False,
                reason="Candidate failed heuristic minimum length or overlap requirements.",
            )
        return score, prompt_version
