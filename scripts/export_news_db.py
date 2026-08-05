"""Export scored news headlines to a CSV-friendly stream."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.news_sentiment import score_headline


def format_scored_headline(headline: str) -> str:
    """Return a CSV row for a headline and its sentiment score."""
    return f'"{headline}",{score_headline(headline):.1f}'


if __name__ == "__main__":
    print("headline,sentiment_score")
    print(format_scored_headline("Clean growth award improves ESG outlook"))
