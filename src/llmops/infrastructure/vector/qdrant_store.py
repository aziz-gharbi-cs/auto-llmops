"""Qdrant vector store adapter implementing BaseVectorStore via Qdrant's REST API."""

import logging
import uuid
from collections.abc import Sequence
from typing import Any

import httpx

from llmops.domain.exceptions import VectorStoreError
from llmops.domain.interfaces.vector_store import BaseVectorStore
from llmops.domain.models.document import ChunkProvenance, DocumentChunk
from llmops.domain.models.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """Vector database adapter communicating with Qdrant over HTTP REST API."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: str | None = None,
        https: bool = False,
        client: httpx.Client | None = None,
    ) -> None:
        scheme = "https" if https else "http"
        self.base_url = f"{scheme}://{host}:{port}"
        self.api_key = api_key
        self._injected_client = client

    def _get_client(self) -> httpx.Client:
        if self._injected_client is not None:
            return self._injected_client
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        return httpx.Client(base_url=self.base_url, headers=headers, timeout=10.0)

    def create_collection(
        self, collection_name: str, vector_size: int = 384, distance: str = "Cosine"
    ) -> None:
        """Create a new vector collection in Qdrant."""
        client = self._get_client()
        try:
            url = f"{self.base_url}/collections/{collection_name}"
            payload = {
                "vectors": {
                    "size": vector_size,
                    "distance": distance,
                }
            }
            resp = client.put(url, json=payload)
            if resp.status_code not in {200, 201}:
                raise VectorStoreError(
                    f"Failed to create collection '{collection_name}': {resp.text}"
                )
        except httpx.HTTPError as err:
            raise VectorStoreError(
                f"HTTP error creating collection '{collection_name}': {err}"
            ) from err
        finally:
            if self._injected_client is None:
                client.close()

    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        client = self._get_client()
        try:
            url = f"{self.base_url}/collections/{collection_name}"
            resp = client.get(url)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
        finally:
            if self._injected_client is None:
                client.close()

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[list[float]],
    ) -> int:
        """Upsert document chunks and vectors into Qdrant collection with full provenance."""
        if len(chunks) != len(embeddings):
            msg = f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            raise VectorStoreError(msg)

        if not chunks:
            return 0

        points: list[dict[str, Any]] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            prov = chunk.provenance
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            payload = {
                "document_id": chunk.document_id,
                "document_version": prov.document_version,
                "chunk_id": chunk.chunk_id,
                "source_name": prov.source_name,
                "source_url": prov.source_url,
                "dataset_version": prov.dataset_version,
                "ingestion_run_id": prov.ingestion_run_id,
                "content_hash": chunk.content_hash,
                "text": chunk.content,
                "section_heading": chunk.section_heading,
                "token_count": chunk.token_count,
                "created_at": prov.created_at.isoformat(),
            }
            points.append(
                {
                    "id": point_id,
                    "vector": embedding,
                    "payload": payload,
                }
            )

        client = self._get_client()
        try:
            url = f"{self.base_url}/collections/{collection_name}/points?wait=true"
            resp = client.put(url, json={"points": points})
            if resp.status_code not in {200, 201}:
                raise VectorStoreError(
                    f"Failed to upsert points into '{collection_name}': {resp.text}"
                )
            return len(points)
        except httpx.HTTPError as err:
            raise VectorStoreError(f"HTTP error upserting points: {err}") from err
        finally:
            if self._injected_client is None:
                client.close()

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Perform semantic similarity search with optional metadata filtering."""
        search_body: dict[str, Any] = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
        }

        # Apply metadata filters if provided
        if filter_criteria:
            must_filters: list[dict[str, Any]] = []
            for key, val in filter_criteria.items():
                if val is not None:
                    must_filters.append({"key": key, "match": {"value": val}})
            if must_filters:
                search_body["filter"] = {"must": must_filters}

        client = self._get_client()
        try:
            url = f"{self.base_url}/collections/{collection_name}/points/search"
            resp = client.post(url, json=search_body)
            if resp.status_code != 200:
                raise VectorStoreError(
                    f"Search failed on collection '{collection_name}': {resp.text}"
                )

            results_data = resp.json().get("result", [])
            retrieved: list[RetrievedChunk] = []

            for hit in results_data:
                payload = hit.get("payload", {})
                prov = ChunkProvenance(
                    document_id=payload.get("document_id", ""),
                    document_version=payload.get("document_version", "1.0.0"),
                    chunk_id=payload.get("chunk_id", ""),
                    source_name=payload.get("source_name", ""),
                    source_url=payload.get("source_url", ""),
                    dataset_version=payload.get("dataset_version", "v1"),
                    ingestion_run_id=payload.get("ingestion_run_id", "run_default"),
                    content_hash=payload.get("content_hash", ""),
                    text=payload.get("text", ""),
                )
                retrieved.append(
                    RetrievedChunk(
                        chunk_id=payload.get("chunk_id", ""),
                        document_id=payload.get("document_id", ""),
                        content=payload.get("text", ""),
                        score=float(hit.get("score", 0.0)),
                        section_heading=payload.get("section_heading", ""),
                        provenance=prov,
                    )
                )

            return retrieved
        except httpx.HTTPError as err:
            raise VectorStoreError(f"HTTP search error: {err}") from err
        finally:
            if self._injected_client is None:
                client.close()

    def delete_collection(self, collection_name: str) -> bool:
        """Delete collection from Qdrant."""
        client = self._get_client()
        try:
            url = f"{self.base_url}/collections/{collection_name}"
            resp = client.delete(url)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
        finally:
            if self._injected_client is None:
                client.close()
