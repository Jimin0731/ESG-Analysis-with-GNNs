"""Evaluation metrics for ESG model experiments."""

from __future__ import annotations


def mean_absolute_error(actual: list[float], predicted: list[float]) -> float:
    """Compute mean absolute error for equal-length score vectors."""
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if not actual:
        raise ValueError("actual must contain at least one value")
    return sum(abs(left - right) for left, right in zip(actual, predicted, strict=True)) / len(actual)
