"""Command-line interface for the LLMOps platform."""

import argparse
import sys
from collections.abc import Sequence

from llmops import __version__


def create_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser for the llmops CLI."""
    parser = argparse.ArgumentParser(
        prog="llmops",
        description="Antigravity LLMOps Platform --- developer CLI.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"llmops {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Ingestion command
    ingest_p = subparsers.add_parser(
        "ingest",
        help="Crawl official documentation sources into raw snapshots.",
    )
    ingest_p.add_argument(
        "--config",
        type=str,
        default="configs/ingestion.yaml",
        help="Path to ingestion configuration file.",
    )
    ingest_p.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional specific source to ingest (e.g. fastapi, pydantic, sqlalchemy).",
    )

    # Validation command
    validate_p = subparsers.add_parser(
        "validate",
        help="Validate acquired data, schema, and detect duplicates.",
    )
    validate_p.add_argument(
        "--dataset-path",
        type=str,
        default="data/processed",
        help="Path to dataset directory to validate.",
    )

    # Embed command
    embed_p = subparsers.add_parser(
        "embed",
        help="Generate embeddings and index document chunks into Qdrant.",
    )
    embed_p.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name to index into.",
    )

    # Curate command
    curate_p = subparsers.add_parser(
        "curate",
        help="Generate synthetic Q&A and filter candidates with LLM-as-a-Judge.",
    )
    curate_p.add_argument(
        "--threshold",
        type=float,
        default=4.0,
        help="Minimum overall quality threshold for acceptance.",
    )

    # Train command
    train_p = subparsers.add_parser(
        "train",
        help="Fine-tune Gemma 3 using LoRA / PEFT.",
    )
    train_p.add_argument(
        "--profile",
        choices=["smoke", "full"],
        default="smoke",
        help="Training profile (smoke for CPU smoke test, full for GPU).",
    )
    train_p.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training configuration file.",
    )

    # Evaluate command
    eval_p = subparsers.add_parser(
        "evaluate",
        help="Execute multi-layer regression evaluation against baseline.",
    )
    eval_p.add_argument(
        "--model-id",
        type=str,
        default="candidate",
        help="Model identifier or checkpoint path to evaluate.",
    )

    # Promote command
    promote_p = subparsers.add_parser(
        "promote",
        help="Promote candidate model via quality gate to staging/production.",
    )
    promote_p.add_argument(
        "--model-version",
        type=str,
        required=True,
        help="Model version to promote.",
    )
    promote_p.add_argument(
        "--target-stage",
        choices=["staging", "production"],
        default="staging",
        help="Target promotion stage.",
    )

    # Serve command
    serve_p = subparsers.add_parser(
        "serve",
        help="Run the FastAPI application server.",
    )
    serve_p.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the server to.",
    )
    serve_p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to bind the server to.",
    )
    serve_p.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for local development.",
    )

    return parser


def main(args: Sequence[str] | None = None) -> int:
    """Entrypoint for the llmops command-line interface."""
    parser = create_parser()
    parsed_args = parser.parse_args(args=args)

    if not parsed_args.command:
        parser.print_help()
        return 0

    if parsed_args.command == "ingest":
        from llmops.application.use_cases.ingest_documentation import IngestDocumentationUseCase

        print(f"[llmops] Starting ingestion from config '{parsed_args.config}'...")
        use_case = IngestDocumentationUseCase(config_path=parsed_args.config)
        manifests = use_case.execute(target_source=parsed_args.source)
        for manifest in manifests:
            print(
                f"[llmops] Crawled '{manifest.source_name}': "
                f"{manifest.total_pages_crawled} pages ({manifest.total_bytes} bytes)."
            )
        return 0

    if parsed_args.command == "validate":
        from llmops.application.use_cases.process_documents import ProcessDocumentsUseCase

        print(
            f"[llmops] Processing raw documents and validating into '{parsed_args.dataset_path}'..."
        )
        process_use_case = ProcessDocumentsUseCase(processed_dir=parsed_args.dataset_path)
        validation_report = process_use_case.execute()
        print(
            f"[llmops] Validation complete: {validation_report.total_documents} documents, "
            f"{validation_report.total_chunks} chunks, "
            f"{validation_report.exact_duplicates_removed} exact duplicates removed."
        )
        return 0

    if parsed_args.command == "embed":
        import json
        from pathlib import Path

        from llmops.application.use_cases.retrieval_service import RetrievalService
        from llmops.domain.models.document import DocumentChunk
        from llmops.infrastructure.embeddings.sentence_transformer import (
            SentenceTransformerEmbeddings,
        )
        from llmops.infrastructure.reranker.cross_encoder import CrossEncoderReranker
        from llmops.infrastructure.vector.in_memory_store import InMemoryVectorStore

        target_col = parsed_args.collection or "default"
        print(f"[llmops] Embedding and indexing chunks into '{target_col}'...")
        chunks_file = Path("data/processed/chunks.jsonl")
        chunks: list[DocumentChunk] = []
        if chunks_file.exists():
            for line in chunks_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    chunks.append(DocumentChunk.model_validate(json.loads(line)))

        service = RetrievalService(
            vector_store=InMemoryVectorStore(),
            embedding_model=SentenceTransformerEmbeddings(),
            reranker=CrossEncoderReranker(),
            default_collection=parsed_args.collection or "docs_fastapi_pydantic_sqlalchemy",
        )
        count = service.index_chunks(chunks)
        print(f"[llmops] Indexing complete: {count} chunks indexed.")
        return 0

    print(f"[llmops] Executing subcommand '{parsed_args.command}'...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
