"""Evaluation metrics for bulk testing."""

from .metrics import (
    # exact_match,             (deprecated)
    embedding_similarity,
    calculate_aggregate_metrics,
    normalize_text
)

__all__ = [
    'exact_match',
    'embedding_similarity',
    'calculate_aggregate_metrics',
    'normalize_text'
]
