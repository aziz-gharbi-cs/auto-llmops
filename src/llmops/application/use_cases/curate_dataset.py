"""Use case for synthetic data generation, LLM judging, and curated dataset creation."""

import json
import logging
from pathlib import Path

from llmops.domain.interfaces.llm import BaseLLMClient
from llmops.domain.models.curation import DatasetSplitSummary, TrainingExample
from llmops.domain.models.document import DocumentChunk
from llmops.domain.services.dataset_splitter import DatasetSplitter
from llmops.domain.services.prompt_manager import PromptManager
from llmops.infrastructure.curation.judge import DataCurationJudge
from llmops.infrastructure.curation.synthetic_generator import SyntheticQAGenerator

logger = logging.getLogger(__name__)


class CurateDatasetUseCase:
    """Orchestrates end-to-end dataset curation with generation, judging,
    and leakage-safe splits.
    """

    def __init__(
        self,
        generator_client: BaseLLMClient,
        judge_client: BaseLLMClient | None = None,
        prompt_manager: PromptManager | None = None,
        splitter: DatasetSplitter | None = None,
        dataset_version: str = "v1",
    ) -> None:
        self.prompt_manager = prompt_manager or PromptManager()
        self.generator = SyntheticQAGenerator(
            llm_client=generator_client,
            prompt_manager=self.prompt_manager,
        )
        self.judge = DataCurationJudge(
            llm_client=judge_client or generator_client,
            prompt_manager=self.prompt_manager,
        )
        self.splitter = splitter or DatasetSplitter(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )
        self.dataset_version = dataset_version

    def load_chunks(self, chunks_path: Path) -> list[DocumentChunk]:
        """Load document chunks from a JSONL file."""
        if not chunks_path.exists():
            logger.warning("Chunks file not found at '%s'", chunks_path)
            return []

        chunks: list[DocumentChunk] = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    try:
                        data = json.loads(stripped)
                        chunks.append(DocumentChunk.model_validate(data))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Failed to parse chunk line: %s", exc)
        return chunks

    def execute(
        self,
        chunks_path: str | Path = "data/processed/chunks.jsonl",
        output_dir: str | Path = "data/processed",
        pass_threshold: float = 3.5,
        threshold: float | None = None,
        max_candidates: int | None = None,
        chunks: list[DocumentChunk] | None = None,
        run_id: str | None = None,
    ) -> DatasetSplitSummary:
        """Curate dataset from document chunks and write train/val/test splits."""
        if threshold is not None:
            pass_threshold = threshold
        path_out = Path(output_dir)
        path_out.mkdir(parents=True, exist_ok=True)

        if chunks is None:
            path_chunks = Path(chunks_path)
            chunks = self.load_chunks(path_chunks)
        if max_candidates is not None and max_candidates > 0:
            chunks = chunks[:max_candidates]

        if not chunks:
            logger.info("No chunks available for curation.")
            return DatasetSplitSummary(metadata={"message": "No chunks found to curate"})

        # Load system prompt
        system_prompt, _ = self.prompt_manager.render(
            category="system",
            name="developer_assistant",
        )

        passed_examples: list[TrainingExample] = []
        total_candidates = len(chunks)
        rejected_count = 0

        for idx, chunk in enumerate(chunks, start=1):
            # Step 1: Generate candidate Q&A pair
            candidate, gen_version = self.generator.generate_candidate(chunk)

            # Step 2: LLM-as-a-Judge evaluation
            score, judge_version = self.judge.evaluate(
                context=chunk.content,
                instruction=candidate.instruction,
                response=candidate.response,
                threshold=pass_threshold,
            )

            if score.is_pass:
                ex = TrainingExample.create(
                    example_id=f"ex-{chunk.chunk_id[:8]}-{idx:04d}",
                    system_prompt=system_prompt,
                    instruction=candidate.instruction,
                    response=candidate.response,
                    source_document_id=chunk.document_id,
                    source_url=chunk.provenance.source_url,
                    chunk_id=chunk.chunk_id,
                    generation_model=self.generator.llm_client.model_name,
                    generation_prompt_version=gen_version,
                    judge_model=self.judge.model_name,
                    judge_prompt_version=judge_version,
                    judge_score=score,
                    dataset_version=self.dataset_version,
                )
                passed_examples.append(ex)
            else:
                rejected_count += 1

        # Step 3: Leakage-safe train/val/test splitting
        train_examples, val_examples, test_examples = self.splitter.split(passed_examples)

        # Step 4: Write splits and external metadata (Section 11)
        self._write_split(path_out, "train", train_examples)
        self._write_split(path_out, "val", val_examples)
        self._write_split(path_out, "test", test_examples)

        # Step 5: Summary metrics
        passed_count = len(passed_examples)
        pass_rate = (passed_count / total_candidates) if total_candidates > 0 else 0.0

        summary = DatasetSplitSummary(
            total_candidates=total_candidates,
            passed_candidates=passed_count,
            rejected_candidates=rejected_count,
            pass_rate=round(pass_rate, 4),
            train_count=len(train_examples),
            val_count=len(val_examples),
            test_count=len(test_examples),
            metadata={
                "dataset_version": self.dataset_version,
                "generator_model": self.generator.llm_client.model_name,
                "judge_model": self.judge.model_name,
                "pass_threshold": pass_threshold,
            },
        )

        summary_path = path_out / "dataset_summary.json"
        summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

        logger.info(
            "Curation completed: %d total, %d passed (%.1f%%), %d train, %d val, %d test",
            total_candidates,
            passed_count,
            pass_rate * 100,
            len(train_examples),
            len(val_examples),
            len(test_examples),
        )
        return summary

    def _write_split(
        self,
        output_dir: Path,
        split_name: str,
        examples: list[TrainingExample],
    ) -> None:
        """Write model-facing dataset file and audit metadata file."""
        model_file = output_dir / f"{split_name}.jsonl"
        meta_file = output_dir / f"{split_name}_metadata.jsonl"

        with (
            model_file.open("w", encoding="utf-8") as f_model,
            meta_file.open("w", encoding="utf-8") as f_meta,
        ):
            for ex in examples:
                payload = {"messages": [msg.model_dump() for msg in ex.messages]}
                f_model.write(json.dumps(payload, ensure_ascii=False) + "\n")
                f_meta.write(ex.metadata.model_dump_json() + "\n")
