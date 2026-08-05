"""Inspect and normalize BEA-style CSV headers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.external_data import normalize_header


def normalize_headers(headers: list[str]) -> list[str]:
    """Normalize a list of source headers."""
    return [normalize_header(header) for header in headers]


if __name__ == "__main__":
    print(normalize_headers(["Geo Name", "Line-Code", "Description"]))
