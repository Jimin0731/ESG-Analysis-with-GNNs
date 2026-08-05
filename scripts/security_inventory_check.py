#!/usr/bin/env python3
"""Lightweight repository security and inventory checks without printing secret values."""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KNOWN_NEWSAPI_SHA256 = "d9d80879dc4f5ff3a92a2a2df056cbba65770b74dbf7f9f9ac60299616d2a53d"
REQUIRED_DATASET_IDS = {
    "esg_company_data", "use_table", "make_table", "oecd_icio_usa_io",
    "bea_real_value_added", "bea_real_intermediate_input", "bea_intermediate_input_price_indexes",
    "bea_real_gross_output", "bea_gross_output_price_indexes", "bea_gross_output",
    "emissions_ghgp_data", "environmental_tax_data", "optional_news_api_data",
    "optional_sqlite_news_database",
}


def tracked_files() -> list[Path]:
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [ROOT / line for line in out.splitlines()]


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_inventory_ids(text: str) -> set[str]:
    # The inventory intentionally uses a simple YAML subset: dataset entries are `- id: ...`.
    ids = set(re.findall(r"(?m)^\s*-\s+id:\s*([A-Za-z0-9_\-]+)\s*$", text))
    if not ids:
        fail("configs/datasets.yaml did not parse into dataset entries")
    return ids


def main() -> None:
    tracked = tracked_files()
    rels = {p.relative_to(ROOT).as_posix() for p in tracked}
    if ".env" in rels:
        fail("tracked .env file detected")

    path_suffixes = {".py", ".md", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".ipynb"}
    users_path_offenders = []
    known_secret_offenders = []
    for path in tracked:
        if not path.is_file() or path.suffix == ".pdf":
            continue
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(errors="ignore")
        if path.suffix in path_suffixes and re.search(r"/Users/[^\s\"']+", text):
            users_path_offenders.append(rel)
        for match in re.findall(r"[A-Za-z0-9_\-]{20,}", text):
            if hashlib.sha256(match.encode()).hexdigest() == KNOWN_NEWSAPI_SHA256:
                known_secret_offenders.append(rel)
                break

    if users_path_offenders:
        fail("absolute /Users paths detected in tracked active files: " + ", ".join(sorted(users_path_offenders)))
    if known_secret_offenders:
        fail("known exposed NewsAPI credential detected in current tree; values suppressed")

    inventory_path = ROOT / "configs" / "datasets.yaml"
    ids = parse_inventory_ids(inventory_path.read_text())
    missing = REQUIRED_DATASET_IDS - ids
    if missing:
        fail("dataset inventory missing required IDs: " + ", ".join(sorted(missing)))

    env_values = {}
    for line in (ROOT / ".env.example").read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_values[key] = value
    if env_values.get("NEWS_API_KEY") != "":
        fail("NEWS_API_KEY placeholder missing or populated in .env.example")
    populated = [key for key, value in env_values.items() if value]
    if populated:
        fail(".env.example contains non-placeholder values: " + ", ".join(sorted(populated)))

    print("Security inventory checks passed")


if __name__ == "__main__":
    main()
