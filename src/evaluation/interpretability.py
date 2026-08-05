"""Interpretability helpers for ESG graph experiments."""

from __future__ import annotations


def rank_feature_importance(weights: dict[str, float]) -> list[tuple[str, float]]:
    """Return feature weights ordered by absolute contribution."""
    return sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)
