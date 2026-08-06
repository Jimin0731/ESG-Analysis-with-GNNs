#!/usr/bin/env python3
"""Regenerate or verify the committed deterministic synthetic result summary."""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.notebook_support import (  # noqa: E402
    build_synthetic_prepared_data,
    run_example_baseline,
    run_example_training,
)

SEED = 17
EPOCHS = 2
MODELS = (
    ("mlp", "non-graph neural baseline", False),
    ("weighted_gat", "directed weighted graph model", True),
    ("gpr_gnn", "directed graph propagation model", True),
    ("train_mean", "non-neural train-mean baseline", False),
)
CSV_PATH = ROOT / "results" / "summary_metrics.csv"
JSON_PATH = ROOT / "results" / "summary_metrics.json"
SVG_PATH = ROOT / "assets" / "model-comparison.svg"
README_PATH = ROOT / "README.md"
TABLE_START = "<!-- RESULTS_TABLE_START -->"
TABLE_END = "<!-- RESULTS_TABLE_END -->"


def build_summary() -> dict:
    """Run the same fixed comparison as notebook 02 and return plain data."""
    data = build_synthetic_prepared_data(SEED)
    evaluations = {}
    for name, _, _ in MODELS[:-1]:
        result = run_example_training(name, epochs=EPOCHS, seed=SEED, prepared_data=data)
        evaluations[name] = result.evaluations
    evaluations["train_mean"] = run_example_baseline(data).evaluations

    models = []
    for name, family, uses_graph in MODELS:
        split_payload = {}
        for split in ("validation", "test"):
            result = evaluations[name][split]
            split_payload[split] = {
                "macro_mae": result.macro_mae,
                "macro_rmse": result.macro_rmse,
                "macro_r2": result.macro_r2,
                "targets": {
                    metric.target_name: {
                        "mae": metric.mae,
                        "rmse": metric.rmse,
                        "r2": metric.r2,
                        "observed_count": metric.observed_count,
                    }
                    for metric in result.metrics
                },
            }
        models.append({
            "model": name,
            "family": family,
            "uses_graph": uses_graph,
            "metrics": split_payload,
        })
    return {
        "schema_version": 1,
        "scope": "deterministic_synthetic_smoke",
        "source": "examples.notebook_support.build_synthetic_prepared_data",
        "protocol": {
            "seed": SEED,
            "epochs": EPOCHS,
            "nodes": len(data.train[0].node_ids),
            "directed_edges_per_snapshot": int(data.train[0].edge_index.shape[1]),
            "train_periods": [item.period for item in data.train],
            "validation_periods": [item.period for item in data.validation],
            "test_periods": [item.period for item in data.test],
            "targets": list(data.target_names),
            "selection_rule": "validation metrics only; test metrics are descriptive",
        },
        "models": models,
        "warning": (
            "Synthetic smoke results validate deterministic execution only. They are not "
            "scientific evidence, real-data reproduction, hyperparameter optimization, or causal proof."
        ),
    }


def render_json(summary: dict) -> str:
    return json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_csv(summary: dict) -> str:
    output = io.StringIO(newline="")
    fieldnames = (
        "scope", "model", "family", "uses_graph", "split", "metric", "value",
        "selection_role", "seed", "epochs",
    )
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for model in summary["models"]:
        for split in ("validation", "test"):
            for metric in ("macro_mae", "macro_rmse", "macro_r2"):
                writer.writerow({
                    "scope": summary["scope"],
                    "model": model["model"],
                    "family": model["family"],
                    "uses_graph": str(model["uses_graph"]).lower(),
                    "split": split,
                    "metric": metric,
                    "value": f"{model['metrics'][split][metric]:.9f}",
                    "selection_role": "workflow comparison" if split == "validation" else "descriptive only",
                    "seed": summary["protocol"]["seed"],
                    "epochs": summary["protocol"]["epochs"],
                })
    return output.getvalue()


def render_readme_table(summary: dict) -> str:
    lines = [
        "| Model | Uses graph? | Validation MAE | Validation RMSE | Test RMSE* |",
        "|---|:---:|---:|---:|---:|",
    ]
    for model in summary["models"]:
        validation = model["metrics"]["validation"]
        test = model["metrics"]["test"]
        lines.append(
            f"| `{model['model']}` | {'Yes' if model['uses_graph'] else 'No'} | "
            f"{validation['macro_mae']:.3f} | {validation['macro_rmse']:.3f} | {test['macro_rmse']:.3f} |"
        )
    lines.append("")
    lines.append("\\* Test values are descriptive only and are never used for model selection.")
    return "\n".join(lines)


def render_model_comparison_svg(summary: dict) -> str:
    values = [(model["model"], model["metrics"]["validation"]["macro_rmse"], model["uses_graph"])
              for model in summary["models"]]
    maximum = 1.30
    chart_x, chart_width = 245, 580
    colors = {True: "#25b7a4", False: "#73839c"}
    rows = []
    for index, (name, value, uses_graph) in enumerate(values):
        y = 165 + index * 70
        width = chart_width * value / maximum
        rows.append(
            f'<text x="220" y="{y + 22}" text-anchor="end" class="label">{name}</text>'
            f'<rect x="{chart_x}" y="{y}" width="{width:.1f}" height="34" rx="7" fill="{colors[uses_graph]}"/>'
            f'<text x="{chart_x + width + 12:.1f}" y="{y + 23}" class="value">{value:.3f}</text>'
        )
    ticks = []
    for tick in (0.0, 0.25, 0.50, 0.75, 1.00, 1.25):
        x = chart_x + chart_width * tick / maximum
        ticks.append(
            f'<line x1="{x:.1f}" y1="145" x2="{x:.1f}" y2="440" class="grid"/>'
            f'<text x="{x:.1f}" y="463" text-anchor="middle" class="tick">{tick:.2f}</text>'
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540" role="img" aria-labelledby="title desc">
  <title id="title">Deterministic synthetic validation RMSE by model</title>
  <desc id="desc">Train mean has RMSE 0.286, weighted GAT 0.937, MLP 1.016, and GPR GNN 1.210. Lower is better. These are smoke results, not scientific findings.</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0b1630"/><stop offset="1" stop-color="#14294a"/></linearGradient>
    <style>.title{{font:700 28px system-ui,sans-serif;fill:#f5f8ff}}.sub{{font:15px system-ui,sans-serif;fill:#a9bad5}}.label{{font:600 16px ui-monospace,monospace;fill:#e7eefb}}.value{{font:700 16px system-ui,sans-serif;fill:#f5f8ff}}.tick{{font:13px system-ui,sans-serif;fill:#96a9c7}}.grid{{stroke:#527099;stroke-width:1;opacity:.32}}.note{{font:14px system-ui,sans-serif;fill:#f5d982}}.legend{{font:13px system-ui,sans-serif;fill:#becbe0}}</style>
  </defs>
  <rect width="960" height="540" rx="24" fill="url(#bg)"/>
  <text x="56" y="62" class="title">Synthetic smoke comparison</text>
  <text x="56" y="91" class="sub">Validation macro RMSE · fixed seed 17 · 2 epochs · lower is better</text>
  {''.join(ticks)}
  {''.join(rows)}
  <rect x="56" y="488" width="14" height="14" rx="3" fill="#25b7a4"/><text x="78" y="500" class="legend">graph model</text>
  <rect x="188" y="488" width="14" height="14" rx="3" fill="#73839c"/><text x="210" y="500" class="legend">baseline</text>
  <text x="914" y="500" text-anchor="end" class="note">Deterministic synthetic fixture—not scientific validation</text>
</svg>'''


def update_readme(text: str, table: str) -> str:
    if text.count(TABLE_START) != 1 or text.count(TABLE_END) != 1:
        raise ValueError("README must contain exactly one generated-results marker pair")
    before, remainder = text.split(TABLE_START, 1)
    _, after = remainder.split(TABLE_END, 1)
    return before + TABLE_START + "\n" + table + "\n" + TABLE_END + after


def expected_files(summary: dict) -> dict[Path, str]:
    readme = update_readme(README_PATH.read_text(encoding="utf-8"), render_readme_table(summary))
    return {
        CSV_PATH: render_csv(summary),
        JSON_PATH: render_json(summary),
        SVG_PATH: render_model_comparison_svg(summary),
        README_PATH: readme,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if committed outputs are stale")
    args = parser.parse_args()
    summary = build_summary()
    files = expected_files(summary)
    stale = [path for path, content in files.items() if not path.exists() or path.read_text(encoding="utf-8") != content]
    if args.check:
        if stale:
            print("stale generated files: " + ", ".join(str(path.relative_to(ROOT)) for path in stale), file=sys.stderr)
            return 1
        print("README/results/assets synchronization passed")
        return 0
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print("updated " + ", ".join(str(path.relative_to(ROOT)) for path in files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
