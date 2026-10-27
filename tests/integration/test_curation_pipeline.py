"""Integration tests for the complete dataset curation pipeline (Milestone 6)."""

import json
from pathlib import Path

from llmops.application.use_cases.curate_dataset import CurateDatasetUseCase
from llmops.domain.models.document import ChunkProvenance, DocumentChunk, compute_sha256
from llmops.infrastructure.llm.mock_llm import MockLLMClient


def test_curation_pipeline_end_to_end(tmp_path: Path) -> None:
    # 1. Create temporary chunks.jsonl
    chunks_file = tmp_path / "chunks.jsonl"
    out_dir = tmp_path / "curated"

    chunks: list[DocumentChunk] = []
    # Create chunks across 4 distinct documents
    for doc_idx in range(4):
        for chunk_idx in range(2):
            cid = f"chk-{doc_idx}-{chunk_idx}"
            chunk_text = (
                f"FastAPI and Pydantic provide schema validation for endpoint {doc_idx} "
                f"part {chunk_idx}."
            )
            chunk = DocumentChunk(
                chunk_id=cid,
                document_id=f"doc_{doc_idx:02d}",
                content=chunk_text,
                token_count=15,
                section_heading=f"Section {chunk_idx}",
                provenance=ChunkProvenance(
                    document_id=f"doc_{doc_idx:02d}",
                    document_version="1.0.0",
                    chunk_id=cid,
                    source_name="fastapi",
                    source_url=f"https://fastapi.tiangolo.com/doc{doc_idx}",
                    dataset_version="v1",
                    ingestion_run_id="run-001",
                    content_hash=compute_sha256(chunk_text),
                    text=chunk_text,
                ),
            )
            chunks.append(chunk)

    with chunks_file.open("w", encoding="utf-8") as f:
        for chk in chunks:
            f.write(chk.model_dump_json() + "\n")

    # 2. Setup mock client for generation & judging
    generator_client = MockLLMClient(
        model_name="gemma-3-generator",
        structured_payload={
            "instruction": "How do FastAPI and Pydantic provide schema validation?",
            "response": (
                "FastAPI leverages Pydantic models to automatically validate request schemas "
                "and types."
            ),
        },
    )
    judge_client = MockLLMClient(
        model_name="gemma-3-judge",
        structured_payload={
            "relevance": 5,
            "correctness": 5,
            "groundedness": 5,
            "instruction_following": 5,
            "clarity": 5,
            "overall": 5.0,
            "pass": True,
            "reason": "Accurate, well-grounded schema explanation.",
        },
    )

    use_case = CurateDatasetUseCase(
        generator_client=generator_client,
        judge_client=judge_client,
    )

    # 3. Execute curation
    summary = use_case.execute(
        chunks_path=chunks_file,
        output_dir=out_dir,
        pass_threshold=3.5,
    )

    # 4. Verify summary
    assert summary.total_candidates == 8
    assert summary.passed_candidates >= 1
    assert summary.pass_rate > 0.0

    # 5. Verify output files exist
    train_file = out_dir / "train.jsonl"
    val_file = out_dir / "val.jsonl"
    test_file = out_dir / "test.jsonl"
    train_meta_file = out_dir / "train_metadata.jsonl"
    summary_file = out_dir / "dataset_summary.json"

    assert train_file.exists()
    assert val_file.exists()
    assert test_file.exists()
    assert train_meta_file.exists()
    assert summary_file.exists()

    # 6. Verify Section 11 format in train.jsonl
    train_lines = train_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(train_lines) == summary.train_count

    first_example = json.loads(train_lines[0])
    assert "messages" in first_example
    assert len(first_example["messages"]) == 3
    assert first_example["messages"][0]["role"] == "system"
    assert first_example["messages"][1]["role"] == "user"
    assert first_example["messages"][2]["role"] == "assistant"

    # 7. Verify metadata separation
    train_meta_lines = train_meta_file.read_text(encoding="utf-8").strip().splitlines()
    first_meta = json.loads(train_meta_lines[0])
    assert "source_document_id" in first_meta
    assert "judge_score" in first_meta
    assert "content_hash" in first_meta
    assert first_meta["generation_model"] == "gemma-3-generator"
    assert first_meta["judge_model"] == "gemma-3-judge"
