"""Dataset splitter with strict document-level leakage prevention and deduplication."""

from collections import defaultdict

from llmops.domain.models.curation import TrainingExample


class DatasetSplitter:
    """Splits curated training examples into train, validation, and test sets.

    Guarantees:
    1. Zero document-level leakage: all examples from a source document are assigned
       to the exact same split whenever more than one document is present.
    2. Deterministic partitioning based on sorted document identifiers.
    3. Content-level deduplication to eliminate identical examples.
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> None:
        total = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def deduplicate(self, examples: list[TrainingExample]) -> list[TrainingExample]:
        """Deduplicate examples based on content_hash."""
        seen_hashes: set[str] = set()
        deduped: list[TrainingExample] = []
        for ex in examples:
            if ex.metadata.content_hash not in seen_hashes:
                seen_hashes.add(ex.metadata.content_hash)
                deduped.append(ex)
        return deduped

    def split(
        self,
        examples: list[TrainingExample],
    ) -> tuple[list[TrainingExample], list[TrainingExample], list[TrainingExample]]:
        """Split examples into (train, val, test) ensuring zero document leakage.

        Returns:
            (train_examples, val_examples, test_examples)
        """
        deduped = self.deduplicate(examples)
        if not deduped:
            return [], [], []

        # Group examples by source_document_id
        doc_groups: dict[str, list[TrainingExample]] = defaultdict(list)
        for ex in deduped:
            doc_groups[ex.metadata.source_document_id].append(ex)

        sorted_doc_ids = sorted(doc_groups.keys())
        n_docs = len(sorted_doc_ids)

        # If only one document exists, partition at the example level
        if n_docs == 1:
            all_items = doc_groups[sorted_doc_ids[0]]
            n = len(all_items)
            train_end = max(1, int(n * self.train_ratio))
            val_end = min(n, train_end + max(1, int(n * self.val_ratio))) if n >= 3 else train_end
            return all_items[:train_end], all_items[train_end:val_end], all_items[val_end:]

        # If 2 documents, assign 1 to train, 1 to val (test empty)
        if n_docs == 2:
            return doc_groups[sorted_doc_ids[0]], doc_groups[sorted_doc_ids[1]], []

        # If 3 or more documents, assign whole documents to partitions
        val_count = max(1, round(n_docs * self.val_ratio))
        test_count = max(1, round(n_docs * self.test_ratio))
        train_count = n_docs - val_count - test_count

        if train_count < 1:
            train_count = 1
            if val_count > test_count:
                val_count -= 1
            else:
                test_count -= 1

        train_doc_ids = set(sorted_doc_ids[:train_count])
        val_doc_ids = set(sorted_doc_ids[train_count : train_count + val_count])

        train_examples: list[TrainingExample] = []
        val_examples: list[TrainingExample] = []
        test_examples: list[TrainingExample] = []

        for doc_id, items in doc_groups.items():
            if doc_id in train_doc_ids:
                train_examples.extend(items)
            elif doc_id in val_doc_ids:
                val_examples.extend(items)
            else:
                test_examples.extend(items)

        return train_examples, val_examples, test_examples
