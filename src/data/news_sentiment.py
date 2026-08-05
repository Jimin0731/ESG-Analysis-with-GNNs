"""Lightweight news sentiment utilities for ESG event features."""

from __future__ import annotations

POSITIVE_TERMS = frozenset({"improve", "growth", "award", "clean", "safe"})
NEGATIVE_TERMS = frozenset({"risk", "fine", "spill", "fraud", "unsafe"})


def score_headline(headline: str) -> float:
    """Score a headline using a transparent keyword baseline."""
    tokens = {token.strip(".,:;!?()[]{}\"'").lower() for token in headline.split()}
    return float(len(tokens & POSITIVE_TERMS) - len(tokens & NEGATIVE_TERMS))
