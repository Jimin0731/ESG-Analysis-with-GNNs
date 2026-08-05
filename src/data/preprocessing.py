"""Preprocessing utilities for ESG graph analysis."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class ESGRecord:
    """A single company-level ESG observation."""

    company: str
    sector: str
    environmental: float
    social: float
    governance: float

    @property
    def esg_score(self) -> float:
        """Return the equally weighted composite ESG score."""
        return mean((self.environmental, self.social, self.governance))


def load_sample_esg_data() -> list[ESGRecord]:
    """Return a small synthetic ESG dataset for smoke tests and demos."""
    return [
        ESGRecord("Alpha Energy", "Energy", 62.0, 58.0, 64.0),
        ESGRecord("Beta Foods", "Consumer", 74.0, 79.0, 71.0),
        ESGRecord("Gamma Tech", "Technology", 81.0, 77.0, 83.0),
        ESGRecord("Delta Bank", "Financials", 69.0, 72.0, 75.0),
    ]


def add_composite_scores(records: list[ESGRecord]) -> list[dict[str, float | str]]:
    """Convert ESG records into dictionaries with a composite score field."""
    return [
        {
            "company": record.company,
            "sector": record.sector,
            "environmental": record.environmental,
            "social": record.social,
            "governance": record.governance,
            "esg_score": record.esg_score,
        }
        for record in records
    ]
