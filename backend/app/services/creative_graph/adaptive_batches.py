"""Shared batch sizing rules for downstream LangGraph stages.

The sample-analysis fan-out is the best available signal for request pressure.
When it already took more than one request, downstream stages use one business
object per model call so that a large source set cannot turn into another large
review request later in the graph.
"""

from __future__ import annotations


ADAPTIVE_BATCH_POLICY_VERSION = "adaptive-batches-v1"
MAX_PARALLEL_BATCHES = 3


def sample_batch_count(metadata: dict | None) -> int:
    """Read a validated sample batch count, defaulting to the old path."""
    value = (metadata or {}).get("sample_batch_count", 1)
    return value if type(value) is int and value >= 1 else 1


def downstream_batch_policy(sample_batches: int, *, default_size: int) -> dict:
    """Return a serializable policy used in batch fingerprints and diagnostics."""
    count = sample_batches if type(sample_batches) is int and sample_batches >= 1 else 1
    fine_grained = count > 1
    return {
        "version": ADAPTIVE_BATCH_POLICY_VERSION,
        "sample_batch_count": count,
        "mode": "per_item" if fine_grained else "bounded_group",
        "batch_size": 1 if fine_grained else default_size,
    }


def adaptive_batch_size(sample_batches: int, *, default_size: int) -> int:
    """Use one item once upstream sample analysis was itself split."""
    return downstream_batch_policy(sample_batches, default_size=default_size)["batch_size"]
