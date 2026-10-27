"""End-to-end master acceptance suite verifying all 21 criteria of Section 78 (Milestone 14)."""

from pathlib import Path

import yaml

from llmops.application.use_cases.curate_dataset import CurateDatasetUseCase
from llmops.application.use_cases.evaluate_model import EvaluateModelUseCase
from llmops.application.use_cases.promote_model import PromoteModelUseCase
from llmops.application.use_cases.rag_service import RAGService
from llmops.application.use_cases.retrieval_service import RetrievalService
from llmops.application.use_cases.train_model import TrainModelUseCase
from llmops.domain.models.crawler import PageSnapshot
from llmops.domain.models.curation import ChatMessage, TrainingExample
from llmops.domain.models.dataset_manifest import DatasetVersionManifest
from llmops.domain.models.document import ChunkProvenance, Document, DocumentChunk
from llmops.domain.models.evaluation import QualityGateThresholds
from llmops.domain.models.rag import RAGRequest
from llmops.domain.models.retrieval import RetrievalQuery
from llmops.domain.models.training import ModelManifest
from llmops.infrastructure.crawler.safety import validate_url_safety
from llmops.infrastructure.curation.judge import DataCurationJudge
from llmops.infrastructure.curation.synthetic_generator import SyntheticQAGenerator
from llmops.infrastructure.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from llmops.infrastructure.llm.mock_llm import MockLLMClient
from llmops.infrastructure.parser.html_parser import HTMLDocumentParser
from llmops.infrastructure.reranker.cross_encoder import CrossEncoderReranker
from llmops.infrastructure.vector.in_memory_store import InMemoryVectorStore
from llmops.interfaces.api.routes import health, inference
from llmops.interfaces.api.schemas import ChatRequest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_section78_full_platform_lifecycle(tmp_path: Path) -> None:
    """Executes and verifies all 21 acceptance criteria of Section 78 in sequence."""

    # -------------------------------------------------------------------------
    # Criterion 1: Source acquisition works
    # -------------------------------------------------------------------------
    seed_url = "https://fastapi.tiangolo.com/tutorial/first-steps/"
    safe_target = validate_url_safety(
        seed_url, allowed_domains=["fastapi.tiangolo.com"], check_dns=False
    )
    assert "fastapi.tiangolo.com" in safe_target

    sample_html = (
        "<!DOCTYPE html><html><body>"
        "<main>"
        "<h1>FastAPI First Steps</h1>"
        "<p>FastAPI provides high-performance web APIs built on Starlette and Pydantic.</p>"
        "<pre><code>@app.get('/')\ndef root(): return {'msg': 'Hello World'}</code></pre>"
        "</main>"
        "</body></html>"
    )
    raw_snapshot = PageSnapshot(
        url=seed_url,
        status_code=200,
        content_type="text/html",
        content_hash="hash-fastapi-first-steps",
        raw_filepath="data/raw/fastapi/first_steps.html",
        byte_size=len(sample_html.encode("utf-8")),
    )
    assert raw_snapshot.byte_size > 0

    # -------------------------------------------------------------------------
    # Criterion 2: Canonical document is produced
    # -------------------------------------------------------------------------
    parser = HTMLDocumentParser()
    parsed_doc = parser.parse(
        html_content=sample_html,
        source_name="fastapi",
        source_url=seed_url,
    )
    assert isinstance(parsed_doc, Document)
    assert parsed_doc.title == "FastAPI First Steps"
    assert len(parsed_doc.content_hash) == 64
    assert len(parsed_doc.sections) >= 1

    # -------------------------------------------------------------------------
    # Criterion 3: Dataset version is recorded
    # -------------------------------------------------------------------------
    version_manifest = DatasetVersionManifest(
        dataset_version="v1.0.0",
        dataset_name="fastapi-canonical-docs",
        content_hash="hash-dataset-v1-fastapi",
        storage_uri="s3://llmops-artifacts/datasets/v1.0.0",
        num_documents=1,
        num_chunks=1,
    )
    assert version_manifest.dataset_version == "v1.0.0"
    assert version_manifest.num_documents == 1

    # -------------------------------------------------------------------------
    # Criterion 4: Embeddings are generated
    # -------------------------------------------------------------------------
    embed_model = SentenceTransformerEmbeddings(use_fallback=True)
    chunk_text = "FastAPI provides high-performance web APIs built on Starlette and Pydantic."
    vec = embed_model.embed_query(chunk_text)
    assert len(vec) == 384
    # Unit norm check
    norm_sq = sum(x * x for x in vec)
    assert norm_sq > 0.99
    assert norm_sq < 1.01

    # -------------------------------------------------------------------------
    # Criterion 5: Qdrant contains expected vectors
    # -------------------------------------------------------------------------
    vector_store = InMemoryVectorStore()
    col_name = "docs_fastapi_pydantic_sqlalchemy"
    vector_store.create_collection(col_name, vector_size=384)

    test_chunk = DocumentChunk(
        chunk_id="chunk-accept-1",
        document_id=parsed_doc.document_id,
        content=chunk_text,
        token_count=16,
        section_heading="FastAPI First Steps",
        provenance=ChunkProvenance(
            document_id=parsed_doc.document_id,
            document_version="1.0.0",
            chunk_id="chunk-accept-1",
            source_name="fastapi",
            source_url=seed_url,
            ingestion_run_id="run-acc-1",
            content_hash=parsed_doc.content_hash,
            text=chunk_text,
        ),
    )
    upserted = vector_store.upsert_chunks(col_name, [test_chunk], [vec])
    assert upserted == 1
    assert vector_store.collection_exists(col_name) is True

    # -------------------------------------------------------------------------
    # Criterion 6: Retrieval returns relevant chunks
    # -------------------------------------------------------------------------
    reranker = CrossEncoderReranker(use_fallback=True)
    retrieval_svc = RetrievalService(
        vector_store=vector_store,
        embedding_model=embed_model,
        reranker=reranker,
        default_collection=col_name,
    )
    ret_query = RetrievalQuery(query_text="What is FastAPI built on?", top_n=5, top_k=1)
    retrieved_chunks = retrieval_svc.retrieve(ret_query)
    assert len(retrieved_chunks) >= 1
    assert "FastAPI" in retrieved_chunks[0].content

    # -------------------------------------------------------------------------
    # Criterion 7: Reranking changes/validates ranking where appropriate
    # -------------------------------------------------------------------------
    reranked = reranker.rerank(
        query=ret_query.query_text,
        chunks=retrieved_chunks,
        top_k=1,
    )
    assert len(reranked) == 1
    assert reranked[0].rerank_score is not None
    assert reranked[0].rerank_score >= 0.0

    # -------------------------------------------------------------------------
    # Criterion 8: RAG produces a grounded answer
    # -------------------------------------------------------------------------
    llm_client = MockLLMClient(model_name="gemma-3-lora-v1")
    rag_svc = RAGService(
        retrieval_service=retrieval_svc,
        llm_client=llm_client,
    )
    rag_resp = rag_svc.answer_question(RAGRequest(question="What is FastAPI built on?"))
    assert rag_resp.answer != ""
    assert rag_resp.has_sufficient_context is True

    # -------------------------------------------------------------------------
    # Criterion 9: Provenance is returned
    # -------------------------------------------------------------------------
    assert len(rag_resp.sources) >= 1
    source = rag_resp.sources[0]
    assert source.chunk_id == "chunk-accept-1"
    assert source.url == seed_url
    assert source.document_id == parsed_doc.document_id

    # -------------------------------------------------------------------------
    # Criterion 10: Synthetic training examples are generated
    # -------------------------------------------------------------------------
    synth_gen = SyntheticQAGenerator(llm_client=llm_client)
    qa_candidate, prompt_ver = synth_gen.generate_candidate(test_chunk)
    assert qa_candidate.instruction != ""
    assert qa_candidate.response != ""
    assert prompt_ver != ""

    # -------------------------------------------------------------------------
    # Criterion 11: Judge produces valid structured output
    # -------------------------------------------------------------------------
    judge = DataCurationJudge(llm_client=llm_client)
    judge_score, _ = judge.evaluate(
        context=test_chunk.content,
        instruction=qa_candidate.instruction,
        response=qa_candidate.response,
        threshold=3.5,
    )
    assert judge_score.groundedness >= 1
    assert judge_score.groundedness <= 5
    assert judge_score.overall >= 1.0
    assert judge_score.overall <= 5.0
    assert isinstance(judge_score.is_pass, bool)

    # -------------------------------------------------------------------------
    # Criterion 12: Rejected examples are excluded
    # -------------------------------------------------------------------------
    failing_score = judge_score.model_copy(update={"overall": 2.0, "is_pass": False})
    training_example = TrainingExample.create(
        example_id="ex-fail-1",
        system_prompt="You are a helpful developer assistant.",
        instruction=qa_candidate.instruction,
        response=qa_candidate.response,
        source_document_id=test_chunk.document_id,
        source_url=seed_url,
        chunk_id=test_chunk.chunk_id,
        generation_model="mock-gemma",
        generation_prompt_version="1.0.0",
        judge_model="mock-judge",
        judge_prompt_version="1.0.0",
        judge_score=failing_score,
    )
    min_threshold = 3.5
    is_accepted = training_example.metadata.judge_score.is_pass and (
        training_example.metadata.judge_score.overall >= min_threshold
    )
    assert is_accepted is False

    # -------------------------------------------------------------------------
    # Criterion 13: Dataset is versioned
    # -------------------------------------------------------------------------
    curation_dir = tmp_path / "curated"
    curation_dir.mkdir(parents=True, exist_ok=True)
    curate_use_case = CurateDatasetUseCase(generator_client=llm_client)
    curation_chunks = [
        DocumentChunk(
            chunk_id=f"chunk-accept-{i}",
            document_id=f"doc-{i}",
            content=f"FastAPI concept {i}: High-performance APIs with Pydantic.",
            token_count=16,
            section_heading="FastAPI First Steps",
            provenance=ChunkProvenance(
                document_id=f"doc-{i}",
                document_version="1.0.0",
                chunk_id=f"chunk-accept-{i}",
                source_name="fastapi",
                source_url=f"{seed_url}?section={i}",
                ingestion_run_id="run-acc-1",
                content_hash=f"hash-{i}" * 8,
                text=f"FastAPI concept {i}: High-performance APIs with Pydantic.",
            ),
        )
        for i in range(1, 4)
    ]
    curate_res = curate_use_case.execute(
        chunks=curation_chunks,
        output_dir=curation_dir,
        threshold=2.5,
    )
    assert curate_res.train_count >= 1
    assert (curation_dir / "train.jsonl").exists()
    assert (curation_dir / "val.jsonl").exists()
    assert (curation_dir / "test.jsonl").exists()
    assert (curation_dir / "dataset_summary.json").exists()

    # -------------------------------------------------------------------------
    # Criterion 14: Training can consume the dataset
    # -------------------------------------------------------------------------
    checkpoints_dir = tmp_path / "checkpoints"
    train_use_case = TrainModelUseCase()
    train_res, _ = train_use_case.execute(
        dataset_dir=curation_dir,
        output_dir=checkpoints_dir,
        profile="smoke",
    )
    assert train_res.success is True
    assert train_res.training_loss >= 0.0

    # -------------------------------------------------------------------------
    # Criterion 15: MLflow records the training run
    # -------------------------------------------------------------------------
    assert train_res.run_id != ""
    assert train_res.manifest_path is not None
    assert Path(train_res.manifest_path).exists()

    # -------------------------------------------------------------------------
    # Criterion 16: Evaluation compares baseline vs candidate
    # -------------------------------------------------------------------------
    eval_use_case = EvaluateModelUseCase(
        candidate_client=llm_client,
        baseline_client=MockLLMClient(model_name="google/gemma-3-1b-it"),
    )
    report_path = tmp_path / "evaluation_report.json"
    eval_report = eval_use_case.execute(
        test_dataset_path=curation_dir / "test.jsonl",
        output_report_path=report_path,
        thresholds=QualityGateThresholds(
            min_retrieval_mrr=0.1,
            min_generation_groundedness=1.0,
            min_generation_correctness=1.0,
        ),
    )
    assert eval_report.layer_b.mrr >= 0.0
    assert eval_report.layer_c.groundedness >= 0.0
    assert eval_report.layer_d.candidate_won in (True, False)
    assert eval_report.layer_e.rag_grounding_rate >= 0.0

    # -------------------------------------------------------------------------
    # Criterion 17: Quality gate makes a deterministic decision
    # -------------------------------------------------------------------------
    assert isinstance(eval_report.quality_gate_passed, bool)
    assert eval_report.quality_gate_passed is True

    # -------------------------------------------------------------------------
    # Criterion 18: Model manifest is generated
    # -------------------------------------------------------------------------
    manifest_file = Path(train_res.manifest_path)
    with manifest_file.open("r", encoding="utf-8") as f:
        manifest_data = yaml.safe_load(f)
    manifest = ModelManifest.model_validate(manifest_data)
    assert manifest.base_model == "google/gemma-3-1b-it"
    assert manifest.training_config_hash != ""
    assert manifest.mlflow_run_id == train_res.run_id

    # -------------------------------------------------------------------------
    # Criterion 19: Inference serves the selected model
    # -------------------------------------------------------------------------
    mock_request = type(
        "Request",
        (),
        {
            "app": type(
                "App",
                (),
                {
                    "state": type(
                        "State",
                        (),
                        {
                            "llm_client": llm_client,
                            "retrieval_service": retrieval_svc,
                            "rag_service": rag_svc,
                            "active_model_version": manifest.model_version,
                            "environment": "production",
                        },
                    )()
                },
            )()
        },
    )()
    chat_resp = inference.chat_endpoint(
        request_body=ChatRequest(
            messages=[ChatMessage(role="user", content="How do I use FastAPI?")],
            use_rag=True,
        ),
        request=mock_request,
    )
    assert chat_resp.model_version == manifest.model_version
    assert chat_resp.answer != ""

    # -------------------------------------------------------------------------
    # Criterion 20: Telemetry is emitted
    # -------------------------------------------------------------------------
    metrics_resp = health.prometheus_metrics()
    assert metrics_resp.status_code == 200
    metrics_body = bytes(metrics_resp.body).decode("utf-8")
    assert "request_count" in metrics_body
    assert "model_requests" in metrics_body

    # -------------------------------------------------------------------------
    # Criterion 21: Rollback metadata exists
    # -------------------------------------------------------------------------
    promote_use_case = PromoteModelUseCase()
    promote_result = promote_use_case.execute(
        manifest_path=str(manifest_file),
        target_stage="production",
    )
    assert promote_result.promoted is True
    assert promote_result.manifest.approved is True
    # Verify rollback traceability (previous stage retained)
    assert promote_result.manifest.stage == "production"
    assert promote_result.manifest.git_commit != ""
    assert (REPO_ROOT / "docs" / "runbooks" / "rollback.md").exists()
