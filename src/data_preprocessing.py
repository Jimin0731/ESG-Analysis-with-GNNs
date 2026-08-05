"""Data preprocessing utilities for ESG graph analysis."""

from __future__ import annotations

import pandas as pd


def load_sample_esg_data() -> pd.DataFrame:
    """Return a small synthetic ESG dataset for smoke tests and demos."""
    return pd.DataFrame(
        {
            "company": ["Alpha Energy", "Beta Foods", "Gamma Tech", "Delta Bank"],
            "sector": ["Energy", "Consumer", "Technology", "Financials"],
            "environmental": [62.0, 74.0, 81.0, 69.0],
            "social": [58.0, 79.0, 77.0, 72.0],
            "governance": [64.0, 71.0, 83.0, 75.0],
        }
    )


def add_composite_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a composite ESG score column to a dataframe."""
    score_columns = ["environmental", "social", "governance"]
    result = frame.copy()
    result["esg_score"] = result[score_columns].mean(axis=1)
    return result
