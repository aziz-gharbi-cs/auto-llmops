"""Unit tests for DatasetSplitter ensuring zero document leakage and deduplication."""

from llmops.domain.models.curation import JudgeScore, TrainingExample
from llmops.domain.services.dataset_splitter import DatasetSplitter


def _make_example(example_id: str, doc_id: str, text: str) -> TrainingExample:
    score = JudgeScore(
        relevance=5,
        correctness=5,
        groundedness=5,
        instruction_following=5,
        clarity=5,
        overall=5.0,
        is_pass=True,
    )
    return TrainingExample.create(
        example_id=example_id,
        system_prompt="System",
        instruction=f"Question about {text}",
        response=f"Answer for {text}",
        source_document_id=doc_id,
        source_url=f"https://example.com/{doc_id}",
        chunk_id=f"chk-{example_id}",
        generation_model="gen-model",
        generation_prompt_version="1.0.0",
        judge_model="judge-model",
        judge_prompt_version="1.0.0",
        judge_score=score,
    )


def test_zero_document_leakage() -> None:
    """Verify that all examples from a given document are in the same partition."""
    examples: list[TrainingExample] = []
    # 5 documents, 2 examples each
    for i in range(5):
        doc_id = f"doc_{i:02d}"
        examples.append(_make_example(f"ex_{i}_a", doc_id, f"topic {i} part 1"))
        examples.append(_make_example(f"ex_{i}_b", doc_id, f"topic {i} part 2"))

    splitter = DatasetSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    train, val, test = splitter.split(examples)

    train_doc_ids = {ex.metadata.source_document_id for ex in train}
    val_doc_ids = {ex.metadata.source_document_id for ex in val}
    test_doc_ids = {ex.metadata.source_document_id for ex in test}

    # Strict isolation: document sets must not overlap
    assert train_doc_ids.isdisjoint(val_doc_ids)
    assert train_doc_ids.isdisjoint(test_doc_ids)
    assert val_doc_ids.isdisjoint(test_doc_ids)
    assert len(train) + len(val) + len(test) == len(examples)


def test_deduplication() -> None:
    """Verify duplicate examples are eliminated."""
    ex1 = _make_example("ex1", "doc1", "identical topic")
    ex2 = _make_example("ex2", "doc1", "identical topic")  # Same instruction & response

    splitter = DatasetSplitter()
    deduped = splitter.deduplicate([ex1, ex2])
    assert len(deduped) == 1


def test_single_document_partition() -> None:
    """Verify fallback splitting when only one document exists."""
    examples = [_make_example(f"ex_{i}", "single_doc", f"subtopic {i}") for i in range(6)]

    splitter = DatasetSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    train, val, test = splitter.split(examples)

    assert len(train) + len(val) + len(test) == 6
    assert len(train) >= 1
