"""Continuous LLMOps lifecycle orchestration engine (Section 77, 78, Milestone 14)."""

import logging
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from llmops.application.use_cases.curate_dataset import CurateDatasetUseCase
from llmops.application.use_cases.evaluate_model import EvaluateModelUseCase
from llmops.application.use_cases.process_documents import ProcessDocumentsUseCase
from llmops.application.use_cases.promote_model import PromoteModelUseCase
from llmops.application.use_cases.retrieval_service import RetrievalService
from llmops.application.use_cases.train_model import TrainModelUseCase
from llmops.domain.interfaces.storage import BaseStorage
from llmops.domain.models.evaluation import QualityGateThresholds
from llmops.infrastructure.llm.mock_llm import MockLLMClient
from llmops.infrastructure.observability.metrics import registry
from llmops.infrastructure.observability.tracer import tracer

logger = logging.getLogger(__name__)


class ContinuousPipelineResult(BaseModel):
    """Execution report returned by the continuous pipeline."""

    pipeline_run_id: str
    timestamp: float
    stage_durations_sec: dict[str, float] = Field(default_factory=dict)
    crawled_docs_count: int = 0
    chunks_count: int = 0
    curated_examples_count: int = 0
    training_executed: bool = False
    evaluation_passed: bool = False
    model_version: str = "gemma-3-lora-v1"
    promoted_stage: str | None = None
    manifest_path: str | None = None


class ContinuousLLMOpsPipeline:
    """Orchestrates continuous integration, training, evaluation, and promotion."""

    def __init__(
        self,
        storage: BaseStorage,
        retrieval_service: RetrievalService | None = None,
        output_dir: Path | str = "data/continuous_run",
    ) -> None:
        self.storage = storage
        self.retrieval_service = retrieval_service
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_lifecycle(
        self,
        raw_documents_dir: Path | str,
        force_retrain: bool = True,
        training_profile: str = "smoke",
        thresholds: QualityGateThresholds | None = None,
    ) -> ContinuousPipelineResult:
        """Execute end-to-end continuous LLMOps lifecycle."""
        run_id = f"pipe-{uuid.uuid4().hex[:8]}"
        durations: dict[str, float] = {}
        t_total_start = time.perf_counter()

        with tracer.start_span("pipeline.continuous_run", trace_id=run_id):
            # 1. Processing & Semantic Chunking
            t0 = time.perf_counter()
            proc_use_case = ProcessDocumentsUseCase(storage=self.storage)
            proc_result = proc_use_case.execute(
                raw_dir=raw_documents_dir,
                output_dir=self.output_dir / "processed",
                run_id=run_id,
            )
            durations["processing"] = time.perf_counter() - t0

            # 2. Vector Indexing into Retrieval Subsystem
            t0 = time.perf_counter()
            if self.retrieval_service and proc_result.chunks:
                self.retrieval_service.index_chunks(proc_result.chunks)
            durations["indexing"] = time.perf_counter() - t0

            # 3. Synthetic QA Generation & LLM-as-a-Judge Curation
            t0 = time.perf_counter()
            curation_use_case = CurateDatasetUseCase(generator_client=MockLLMClient())
            curation_dir = self.output_dir / "curated"
            curation_dir.mkdir(parents=True, exist_ok=True)
            curate_result = curation_use_case.execute(
                chunks=proc_result.chunks,
                output_dir=curation_dir,
                run_id=run_id,
            )
            durations["curation"] = time.perf_counter() - t0

            # 4. Optional / Conditional Training Decision (Section 14 & 78)
            training_executed = False
            manifest_path: str | None = None
            if force_retrain or curate_result.train_count > 0:
                t0 = time.perf_counter()
                train_use_case = TrainModelUseCase()
                checkpoints_dir = self.output_dir / "checkpoints"
                train_result, _ = train_use_case.execute(
                    dataset_dir=curation_dir,
                    output_dir=checkpoints_dir,
                    profile=training_profile,
                )
                manifest_path = train_result.manifest_path
                training_executed = True
                durations["training"] = time.perf_counter() - t0

            # 5. Multi-Layer Evaluation Gate (Layers A through E)
            t0 = time.perf_counter()
            eval_use_case = EvaluateModelUseCase()
            eval_report = eval_use_case.execute(
                test_dataset_path=curation_dir / "test.jsonl",
                output_report_path=self.output_dir / "evaluation_report.json",
                thresholds=thresholds or QualityGateThresholds(),
            )
            durations["evaluation"] = time.perf_counter() - t0

            # 6. Promotion Gate
            promoted_stage = None
            if eval_report.quality_gate_passed and manifest_path:
                promote_use_case = PromoteModelUseCase()
                promote_res = promote_use_case.execute(
                    manifest_path=manifest_path,
                    target_stage="production",
                )
                if promote_res.promoted:
                    promoted_stage = "production"

            # 7. Record Telemetry
            total_duration = time.perf_counter() - t_total_start
            durations["total"] = total_duration
            registry.counter("model_requests").inc(labels={"model": "gemma-3-lora-v1"})

            return ContinuousPipelineResult(
                pipeline_run_id=run_id,
                timestamp=time.time(),
                stage_durations_sec=durations,
                crawled_docs_count=proc_result.document_count,
                chunks_count=len(proc_result.chunks),
                curated_examples_count=curate_result.train_count,
                training_executed=training_executed,
                evaluation_passed=eval_report.quality_gate_passed,
                model_version="gemma-3-lora-v1",
                promoted_stage=promoted_stage,
                manifest_path=manifest_path,
            )
