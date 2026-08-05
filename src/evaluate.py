"""Evaluate the sample ESG baseline model."""

from __future__ import annotations

from sklearn.metrics import mean_absolute_error

from data_preprocessing import add_composite_score, load_sample_esg_data
from models import MeanScoreBaseline


def main() -> None:
    """Run a minimal evaluation workflow."""
    data = add_composite_score(load_sample_esg_data())
    target = data["esg_score"].to_numpy()
    predictions = MeanScoreBaseline().fit(target).predict(len(target))
    mae = mean_absolute_error(target, predictions)
    print(f"Baseline mean absolute error: {mae:.2f}")


if __name__ == "__main__":
    main()
