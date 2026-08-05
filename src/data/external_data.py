"""External ESG and macroeconomic data loading helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def load_csv_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a CSV file as a list of dictionaries."""
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def normalize_header(header: str) -> str:
    """Normalize provider column names for consistent downstream use."""
    return header.strip().lower().replace(" ", "_").replace("-", "_")
