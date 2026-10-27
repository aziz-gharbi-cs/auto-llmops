# ADR 002: Modular Clean Architecture and Provider Abstraction

## Status
Accepted

## Context
LLM platforms evolve rapidly. Model providers (Ollama, vLLM, OpenAI), vector databases (Qdrant, Milvus), storage providers (MinIO, AWS S3), and embedding models change over time. Tightly coupling business logic to specific client libraries results in fragile codebases that require rewriting core logic whenever an infrastructure component is replaced.

## Decision
We enforce a strict four-layer **Clean Architecture**:
1. **Domain Layer (`src/llmops/domain/`)**: Pure business models (e.g., `Document`, `Chunk`, `Provenance`, `EvaluationResult`), value objects, custom exceptions, and abstract repository/service interfaces. No framework or provider dependencies.
2. **Application Layer (`src/llmops/application/`)**: Use cases and orchestration logic coordinating domain models and ports (e.g., `IngestDocumentationUseCase`, `GenerateAnswerUseCase`).
3. **Infrastructure Layer (`src/llmops/infrastructure/`)**: Concrete adapters implementing domain ports (e.g., `QdrantVectorStore`, `OllamaLLMClient`, `vLLMClient`, `S3StorageAdapter`, `SentenceTransformerEmbeddings`).
4. **Interfaces Layer (`src/llmops/interfaces/`)**: External delivery mechanisms including FastAPI REST endpoints, CLI commands, and pipeline runners.

## Alternatives Considered
- **Flat Script Architecture**: Ingestion scripts and training notebooks containing database connections and direct API calls. Rejected as unmaintainable, hard to test, and interview-indefensible.
- **Framework-Bound Architecture (e.g. LangChain / LlamaIndex tightly coupled throughout)**: Creates heavy dependency lock-in and obscured failure modes.

## Consequences
- Domain logic is 100% unit-testable using in-memory mock adapters without requiring network, GPU, or active databases.
- Infrastructure adapters can be swapped or upgraded via configuration without changing use-case logic.
- Clear contract boundaries enforce rigorous code structure.
