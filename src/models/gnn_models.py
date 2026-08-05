"""Baseline graph-model interfaces for ESG experiments."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass
class MeanScoreBaseline:
    """Predict the mean ESG score observed during fitting."""

    mean_score: float | None = None

    def fit(self, targets: list[float]) -> "MeanScoreBaseline":
        """Fit the baseline using target ESG scores."""
        if not targets:
            raise ValueError("targets must contain at least one score")
        self.mean_score = mean(targets)
        return self

    def predict(self, n_samples: int) -> list[float]:
        """Predict the fitted mean for each requested sample."""
        if self.mean_score is None:
            raise ValueError("Model must be fitted before calling predict().")
        return [self.mean_score] * n_samples
