"""Lightweight checks for the public portfolio surface and committed summary."""
from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_root_surface_and_duplicate_architecture_cleanup():
    required = {
        "README.md", "requirements.txt", "pyproject.toml", "src", "configs",
        "scripts", "tests", "notebooks", "assets", "results", "docs", "archive",
    }
    assert required <= {path.name for path in ROOT.iterdir()}
    assert not (ROOT / "DATA_POLICY.md").exists()
    assert not (ROOT / "MIGRATION_PLAN.md").exists()
    assert not list(ROOT.rglob(".gitkeep"))
    for obsolete in ("data_preprocessing.py", "graph_construction.py", "models.py", "train.py", "evaluate.py"):
        assert not (ROOT / "src" / obsolete).exists()


def test_readme_required_story_and_local_links():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert text.startswith("# ESG Analysis with Graph Neural Networks\n")
    for heading in (
        "Project overview", "Project at a glance", "Key results", "Visual summary",
        "Data and graph construction", "Models compared", "Evaluation protocol",
        "Interpretability/anomaly analysis", "How to run", "Repository structure",
        "Limitations", "Research lineage and archive",
    ):
        assert f"## {heading}" in text
    for warning in ("not scientific validation", "not causal", "provenance"):
        assert warning in text.lower()
    local_links = re.findall(r"!?\[[^]]*\]\((?!https?://|#)([^)]+)\)", text)
    assert local_links
    for link in local_links:
        assert (ROOT / link.split("#", 1)[0]).exists(), link


def test_svg_assets_are_valid_and_accessibly_labelled():
    for name in ("model-comparison.svg", "pipeline-overview.svg"):
        path = ROOT / "assets" / name
        tree = ET.parse(path)
        root = tree.getroot()
        assert root.tag.endswith("svg")
        assert root.attrib.get("role") == "img"
        assert any(child.tag.endswith("title") for child in root)
        assert any(child.tag.endswith("desc") for child in root)


def test_csv_json_summary_agree_and_remain_synthetic():
    payload = json.loads((ROOT / "results" / "summary_metrics.json").read_text(encoding="utf-8"))
    with (ROOT / "results" / "summary_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert payload["scope"] == "deterministic_synthetic_smoke"
    assert "not scientific" in payload["warning"].lower()
    assert {model["model"] for model in payload["models"]} == {"mlp", "weighted_gat", "gpr_gnn", "train_mean"}
    assert len(rows) == 4 * 2 * 3
    lookup = {(row["model"], row["split"], row["metric"]): float(row["value"]) for row in rows}
    for model in payload["models"]:
        for split in ("validation", "test"):
            for metric in ("macro_mae", "macro_rmse", "macro_r2"):
                assert lookup[(model["model"], split, metric)] == round(model["metrics"][split][metric], 9)
