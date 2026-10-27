"""Semantic heading-aware document chunker preserving code blocks and generating provenance."""

import hashlib
import re
from collections.abc import Callable

from llmops.domain.models.document import ChunkProvenance, Document, DocumentChunk


def default_token_estimator(text: str) -> int:
    """Fast, deterministic token estimator based on whitespace and punctuation.

    Approximates BPE tokenizers without requiring heavy models during ingestion.
    """
    words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return max(1, len(words))


class SemanticChunker:
    """Splits canonical Documents into semantic chunks bounded by token limits and headings."""

    def __init__(
        self,
        target_chunk_size: int = 400,
        max_chunk_size: int = 600,
        chunk_overlap: int = 50,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.count_tokens = token_counter or default_token_estimator

    def _split_text_into_paragraphs(self, text: str) -> list[str]:
        """Split text by double newlines while preserving code blocks intact."""
        paragraphs: list[str] = []
        raw_parts = text.split("\n\n")
        in_code_block = False
        code_accumulator: list[str] = []

        for part in raw_parts:
            fence_count = part.count("```")
            if in_code_block:
                code_accumulator.append(part)
                if fence_count % 2 == 1:
                    in_code_block = False
                    paragraphs.append("\n\n".join(code_accumulator))
                    code_accumulator = []
            else:
                if fence_count % 2 == 1:
                    in_code_block = True
                    code_accumulator.append(part)
                else:
                    if part.strip():
                        paragraphs.append(part.strip())

        if code_accumulator:
            paragraphs.append("\n\n".join(code_accumulator))

        return paragraphs

    def chunk_document(
        self,
        document: Document,
        dataset_version: str = "v1",
        ingestion_run_id: str = "run_default",
    ) -> list[DocumentChunk]:
        """Chunk a document into a sequence of DocumentChunk items with full provenance."""
        chunks: list[DocumentChunk] = []
        chunk_counter = 0

        # If document has structured sections, process section by section
        sections = document.sections
        if not sections:
            # Fallback if no sections extracted
            from llmops.domain.models.document import Section

            sections = [
                Section(
                    section_id="sec-000",
                    heading=document.title or "Overview",
                    level=1,
                    content=document.content,
                )
            ]

        for sec in sections:
            sec_heading = sec.heading
            paragraphs = self._split_text_into_paragraphs(sec.content)

            current_chunk_parts: list[str] = []
            current_tokens = 0

            for para in paragraphs:
                para_tokens = self.count_tokens(para)

                # If a single paragraph exceeds max_chunk_size, split by sentences
                if para_tokens > self.max_chunk_size:
                    # Flush accumulated parts first
                    if current_chunk_parts:
                        chunk_text = "\n\n".join(current_chunk_parts).strip()
                        c_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                        c_id = f"{document.document_id}_c{chunk_counter:03d}"
                        prov = ChunkProvenance(
                            document_id=document.document_id,
                            document_version=document.version,
                            chunk_id=c_id,
                            source_name=document.source_name,
                            source_url=document.source_url,
                            dataset_version=dataset_version,
                            ingestion_run_id=ingestion_run_id,
                            content_hash=c_hash,
                            text=chunk_text,
                        )
                        chunks.append(
                            DocumentChunk(
                                chunk_id=c_id,
                                document_id=document.document_id,
                                content=chunk_text,
                                content_hash=c_hash,
                                token_count=current_tokens,
                                section_heading=sec_heading,
                                provenance=prov,
                            )
                        )
                        chunk_counter += 1
                        current_chunk_parts = []
                        current_tokens = 0

                    # Split large paragraph by lines or sentences
                    subparts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", para) if p.strip()]
                    for subpart in subparts:
                        sub_tokens = self.count_tokens(subpart)
                        if (
                            current_tokens + sub_tokens > self.target_chunk_size
                            and current_chunk_parts
                        ):
                            chunk_text = " ".join(current_chunk_parts).strip()
                            c_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                            c_id = f"{document.document_id}_c{chunk_counter:03d}"
                            prov = ChunkProvenance(
                                document_id=document.document_id,
                                document_version=document.version,
                                chunk_id=c_id,
                                source_name=document.source_name,
                                source_url=document.source_url,
                                dataset_version=dataset_version,
                                ingestion_run_id=ingestion_run_id,
                                content_hash=c_hash,
                                text=chunk_text,
                            )
                            chunks.append(
                                DocumentChunk(
                                    chunk_id=c_id,
                                    document_id=document.document_id,
                                    content=chunk_text,
                                    content_hash=c_hash,
                                    token_count=current_tokens,
                                    section_heading=sec_heading,
                                    provenance=prov,
                                )
                            )
                            chunk_counter += 1
                            current_chunk_parts = []
                            current_tokens = 0

                        current_chunk_parts.append(subpart)
                        current_tokens += sub_tokens
                    continue

                if current_tokens + para_tokens > self.max_chunk_size and current_chunk_parts:
                    chunk_text = "\n\n".join(current_chunk_parts).strip()
                    c_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                    c_id = f"{document.document_id}_c{chunk_counter:03d}"
                    prov = ChunkProvenance(
                        document_id=document.document_id,
                        document_version=document.version,
                        chunk_id=c_id,
                        source_name=document.source_name,
                        source_url=document.source_url,
                        dataset_version=dataset_version,
                        ingestion_run_id=ingestion_run_id,
                        content_hash=c_hash,
                        text=chunk_text,
                    )
                    chunks.append(
                        DocumentChunk(
                            chunk_id=c_id,
                            document_id=document.document_id,
                            content=chunk_text,
                            content_hash=c_hash,
                            token_count=current_tokens,
                            section_heading=sec_heading,
                            provenance=prov,
                        )
                    )
                    chunk_counter += 1

                    # Apply overlap if feasible
                    if self.chunk_overlap > 0 and current_chunk_parts:
                        last_part = current_chunk_parts[-1]
                        last_tokens = self.count_tokens(last_part)
                        if last_tokens <= self.chunk_overlap:
                            current_chunk_parts = [last_part]
                            current_tokens = last_tokens
                        else:
                            current_chunk_parts = []
                            current_tokens = 0
                    else:
                        current_chunk_parts = []
                        current_tokens = 0

                current_chunk_parts.append(para)
                current_tokens += para_tokens

            # Flush section remainder
            if current_chunk_parts:
                chunk_text = "\n\n".join(current_chunk_parts).strip()
                c_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                c_id = f"{document.document_id}_c{chunk_counter:03d}"
                prov = ChunkProvenance(
                    document_id=document.document_id,
                    document_version=document.version,
                    chunk_id=c_id,
                    source_name=document.source_name,
                    source_url=document.source_url,
                    dataset_version=dataset_version,
                    ingestion_run_id=ingestion_run_id,
                    content_hash=c_hash,
                    text=chunk_text,
                )
                chunks.append(
                    DocumentChunk(
                        chunk_id=c_id,
                        document_id=document.document_id,
                        content=chunk_text,
                        content_hash=c_hash,
                        token_count=current_tokens,
                        section_heading=sec_heading,
                        provenance=prov,
                    )
                )
                chunk_counter += 1

        return chunks
