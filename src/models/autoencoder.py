"""Autoencoder-style feature compression placeholders."""

from __future__ import annotations


def min_max_scale(values: list[float]) -> list[float]:
    """Scale values into the [0, 1] range for simple feature experiments."""
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return [0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]
