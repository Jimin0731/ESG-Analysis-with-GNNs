"""Attention visualization data preparation helpers."""

from __future__ import annotations


def normalize_attention(weights: dict[str, float]) -> dict[str, float]:
    """Normalize attention weights so their absolute values sum to one."""
    total = sum(abs(weight) for weight in weights.values())
    if total == 0:
        return {key: 0.0 for key in weights}
    return {key: abs(weight) / total for key, weight in weights.items()}
