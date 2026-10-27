"""Prometheus metrics registry and standard exposition engine (Section 42)."""

import threading
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass
class MetricSample:
    """A single metric observation with label key-values."""

    labels: dict[str, str]
    value: float
    timestamp: float = field(default_factory=time.time)


class Counter:
    """Thread-safe cumulative Prometheus Counter metric."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self._lock = threading.Lock()
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)

    def inc(self, value: float = 1.0, labels: Mapping[str, str] | None = None) -> None:
        """Increment counter by non-negative value."""
        if value < 0:
            raise ValueError("Counters can only be incremented by non-negative numbers.")
        label_key = tuple(sorted(labels.items())) if labels else ()
        with self._lock:
            self._values[label_key] += value

    def get_samples(self) -> list[MetricSample]:
        """Return snapshot of counter samples."""
        with self._lock:
            return [
                MetricSample(labels=dict(label_tuple), value=val)
                for label_tuple, val in self._values.items()
            ]


class Histogram:
    """Thread-safe Prometheus Histogram with configurable bucket boundaries."""

    DEFAULT_BUCKETS: Sequence[float] = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
    )

    def __init__(
        self,
        name: str,
        description: str,
        buckets: Sequence[float] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._lock = threading.Lock()
        self._counts: dict[tuple[tuple[str, str], ...], int] = defaultdict(int)
        self._sums: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._bucket_counts: dict[tuple[tuple[str, str], ...], dict[float, int]] = defaultdict(
            lambda: dict.fromkeys(self.buckets, 0)
        )

    def observe(self, value: float, labels: Mapping[str, str] | None = None) -> None:
        """Observe a measured value (e.g. latency in seconds)."""
        label_key = tuple(sorted(labels.items())) if labels else ()
        with self._lock:
            self._counts[label_key] += 1
            self._sums[label_key] += value
            b_dict = self._bucket_counts[label_key]
            for boundary in self.buckets:
                if value <= boundary:
                    b_dict[boundary] += 1

    def get_stats(
        self,
    ) -> list[tuple[dict[str, str], int, float, dict[float, int]]]:
        """Return snapshot of (labels, count, sum, bucket_counts)."""
        with self._lock:
            results = []
            for label_tuple in list(self._counts.keys()):
                results.append(
                    (
                        dict(label_tuple),
                        self._counts[label_tuple],
                        self._sums[label_tuple],
                        dict(self._bucket_counts[label_tuple]),
                    )
                )
            return results


class MetricsRegistry:
    """Thread-safe Prometheus registry tracking Section 42 metrics."""

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._init_standard_metrics()

    def _init_standard_metrics(self) -> None:
        """Register standard Section 42 metrics."""
        self.register_counter("request_count", "Total HTTP requests received")
        self.register_histogram("request_latency", "HTTP request latency in seconds")
        self.register_counter("error_count", "Total HTTP and application errors")
        self.register_histogram("retrieval_latency", "Vector retrieval latency in seconds")
        self.register_histogram("embedding_latency", "Embedding generation latency in seconds")
        self.register_histogram("generation_latency", "LLM generation latency in seconds")
        self.register_counter("tokens_in", "Total prompt tokens consumed")
        self.register_counter("tokens_out", "Total completion tokens produced")
        self.register_counter("model_requests", "Total model inference invocations")
        self.register_counter("cache_hits", "Total semantic or retrieval cache hits")

    def register_counter(self, name: str, description: str) -> Counter:
        c = Counter(name, description)
        self._counters[name] = c
        return c

    def register_histogram(
        self, name: str, description: str, buckets: Sequence[float] | None = None
    ) -> Histogram:
        h = Histogram(name, description, buckets)
        self._histograms[name] = h
        return h

    def counter(self, name: str) -> Counter:
        if name not in self._counters:
            return self.register_counter(name, f"Counter metric {name}")
        return self._counters[name]

    def histogram(self, name: str) -> Histogram:
        if name not in self._histograms:
            return self.register_histogram(name, f"Histogram metric {name}")
        return self._histograms[name]

    def generate_exposition(self) -> str:
        """Render metrics in Prometheus standard text format 0.0.4."""
        lines: list[str] = []

        # 1. Output Counters
        for name, counter in sorted(self._counters.items()):
            lines.append(f"# HELP {name} {counter.description}")
            lines.append(f"# TYPE {name} counter")
            samples = counter.get_samples()
            if not samples:
                lines.append(f"{name} 0.0")
            else:
                for s in samples:
                    if s.labels:
                        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(s.labels.items()))
                        lines.append(f"{name}{{{label_str}}} {s.value}")
                    else:
                        lines.append(f"{name} {s.value}")

        # 2. Output Histograms
        for name, hist in sorted(self._histograms.items()):
            lines.append(f"# HELP {name} {hist.description}")
            lines.append(f"# TYPE {name} histogram")
            stats = hist.get_stats()
            if not stats:
                lines.append(f"{name}_count 0")
                lines.append(f"{name}_sum 0.0")
            else:
                for labels, count, total_sum, b_counts in stats:
                    cumulative = 0
                    for bound in hist.buckets:
                        cumulative = b_counts[bound]
                        bucket_labels = dict(labels)
                        bucket_labels["le"] = str(bound)
                        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(bucket_labels.items()))
                        lines.append(f"{name}_bucket{{{label_str}}} {cumulative}")
                    # +Inf bucket
                    inf_labels = dict(labels)
                    inf_labels["le"] = "+Inf"
                    inf_str = ",".join(f'{k}="{v}"' for k, v in sorted(inf_labels.items()))
                    lines.append(f"{name}_bucket{{{inf_str}}} {count}")

                    # Sum and Count
                    if labels:
                        lbl_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
                        lines.append(f"{name}_sum{{{lbl_str}}} {total_sum}")
                        lines.append(f"{name}_count{{{lbl_str}}} {count}")
                    else:
                        lines.append(f"{name}_sum {total_sum}")
                        lines.append(f"{name}_count {count}")

        return "\n".join(lines) + "\n"


# Global singleton metrics registry
registry = MetricsRegistry()
