"""Train a baseline ESG model on sample data."""

from __future__ import annotations

from data_preprocessing import add_composite_score, load_sample_esg_data
from models import MeanScoreBaseline


def main() -> None:
    """Run a minimal training workflow."""
    data = add_composite_score(load_sample_esg_data())
    model = MeanScoreBaseline().fit(data["esg_score"].to_numpy())
    print(f"Trained baseline mean ESG score: {model.mean_score:.2f}")


if __name__ == "__main__":
    main()
