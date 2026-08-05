"""Simple baseline models for ESG score prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MeanScoreBaseline:
    """Predict the mean ESG score observed during fitting."""

    mean_score: float | None = None

    def fit(self, y: np.ndarray) -> "MeanScoreBaseline":
        """Fit the baseline using target ESG scores."""
        self.mean_score = float(np.mean(y))
        return self

    def predict(self, n_samples: int) -> np.ndarray:
        """Predict the fitted mean for each requested sample."""
        if self.mean_score is None:
            raise ValueError("Model must be fitted before calling predict().")
        return np.full(n_samples, self.mean_score)
