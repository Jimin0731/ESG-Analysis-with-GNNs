#!/usr/bin/env python3
"""Lightweight repository security and inventory checks without printing secret values."""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
KNOWN_NEWSAPI_SHA256 = "d9d80879dc4f5ff3a92a2a2df056cbba65770b74dbf7f9f9ac60299616d2a53d"
REQUIRED_DATASET_IDS = {
    "esg_company_data",
    "use_table",
    "make_table",
    "oecd_icio_usa_io",
    "bea_real_value_added",
    "bea_real_intermediate_input",
    "bea_intermediate_input_price_indexes",
    "bea_real_gross_output",
    "bea_gross_output_price_indexes",
    "bea_gross_output",
    "emissions_ghgp_data",
    "environmental_tax_data",
    "optional_news_api_data",
    "optional_sqlite_news_database",
}
REQUIRED_DATASET_FIELDS = {
    "id",
    "description",
    "expected_path",
    "provider",
    "format",
    "required",
    "used_by",
    "redistribution",
    "synthetic_fixture_available",
}
ACTIVE_FILE_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".ipynb"}
# Build this string without the contiguous absolute-path literal so the scanner checks itself.
MACOS_USERS_PREFIX = "/" + "Users" + "/"
MACOS_USERS_PATH_PATTERN = re.compile(re.escape(MACOS_USERS_PREFIX) + r"[^\s\"']+")
CANDIDATE_SECRET_PATTERN = re.compile(r"[A-Za-z0-9_\-]{20,}")


def tracked_files() -> list[Path]:
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [ROOT / line for line in out.splitlines()]


def relative_path_label(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def validate_dataset_inventory(path: Path | None = None) -> dict[str, Any]:
    inventory_path = path or ROOT / "configs" / "datasets.yaml"
    document = yaml.safe_load(inventory_path.read_text())
    if not isinstance(document, dict):
        raise ValueError("dataset inventory must be a YAML mapping")

    datasets = document.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("dataset inventory must contain a datasets list")

    ids: list[str] = []
    for index, entry in enumerate(datasets):
        if not isinstance(entry, dict):
            raise ValueError(f"dataset entry {index} must be a mapping")
        missing_fields = REQUIRED_DATASET_FIELDS - set(entry)
        if missing_fields:
            raise ValueError(f"dataset entry {index} missing required fields: {sorted(missing_fields)}")
        dataset_id = entry["id"]
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f"dataset entry {index} id must be a non-empty string")
        ids.append(dataset_id)
        if not isinstance(entry["used_by"], list):
            raise ValueError(f"dataset entry {dataset_id} used_by must be a list")
        for field in ("required", "synthetic_fixture_available"):
            if not isinstance(entry[field], bool):
                raise ValueError(f"dataset entry {dataset_id} field {field} must be a boolean")

    duplicate_ids = sorted({dataset_id for dataset_id in ids if ids.count(dataset_id) > 1})
    if duplicate_ids:
        raise ValueError(f"dataset IDs must be unique; duplicates: {duplicate_ids}")

    missing_ids = REQUIRED_DATASET_IDS - set(ids)
    if missing_ids:
        raise ValueError(f"dataset inventory missing required IDs: {sorted(missing_ids)}")

    return document


def find_macos_users_path_offenders(paths: list[Path]) -> list[str]:
    offenders = []
    for path in paths:
        if not path.is_file() or path.suffix not in ACTIVE_FILE_SUFFIXES:
            continue
        rel = relative_path_label(path)
        text = path.read_text(errors="ignore")
        if MACOS_USERS_PATH_PATTERN.search(text):
            offenders.append(rel)
    return sorted(offenders)


def find_known_secret_offenders(paths: list[Path]) -> list[str]:
    offenders = []
    for path in paths:
        if not path.is_file() or path.suffix == ".pdf":
            continue
        rel = relative_path_label(path)
        text = path.read_text(errors="ignore")
        for match in CANDIDATE_SECRET_PATTERN.findall(text):
            if hashlib.sha256(match.encode()).hexdigest() == KNOWN_NEWSAPI_SHA256:
                offenders.append(rel)
                break
    return sorted(offenders)


def validate_env_example(path: Path | None = None) -> dict[str, str]:
    env_path = path or ROOT / ".env.example"
    values = {}
    for line in env_path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    if values.get("NEWS_API_KEY") != "":
        raise ValueError("NEWS_API_KEY placeholder missing or populated in .env.example")
    populated = [key for key, value in values.items() if value]
    if populated:
        raise ValueError(f".env.example contains non-placeholder values: {sorted(populated)}")
    return values


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    tracked = tracked_files()
    rels = {p.relative_to(ROOT).as_posix() for p in tracked}
    if ".env" in rels:
        fail("tracked .env file detected")

    users_path_offenders = find_macos_users_path_offenders(tracked)
    if users_path_offenders:
        fail("absolute macOS user paths detected in tracked active files: " + ", ".join(users_path_offenders))

    known_secret_offenders = find_known_secret_offenders(tracked)
    if known_secret_offenders:
        fail("known exposed NewsAPI credential detected in current tree; values suppressed")

    try:
        validate_dataset_inventory()
        validate_env_example()
    except ValueError as exc:
        fail(str(exc))

    print("Security inventory checks passed")


if __name__ == "__main__":
    main()
