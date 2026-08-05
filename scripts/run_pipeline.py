#!/usr/bin/env python
"""Run the extracted ESG GNN pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.data.preprocessing import build_graph_dataset, make_smoke_dataset, read_io_matrix, targets_from_esg_frame
from src.training import evaluate_model, train_model

DEFAULT_METRICS_OUTPUT = REPO_ROOT / "results" / "metrics" / "smoke_test_metrics.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke-test", action="store_true", help="run on deterministic built-in sample data")
    p.add_argument("--io-matrix", type=Path, help="CSV/XLSX industry I/O matrix")
    p.add_argument("--esg-data", type=Path, help="CSV with ESG scores or E/S/G pillar columns")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_OUTPUT, help="JSON metrics output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise SystemExit("epochs must be a positive integer")
    if args.smoke_test:
        dataset = make_smoke_dataset()
    else:
        if not args.io_matrix or not args.esg_data:
            raise SystemExit("Provide --io-matrix and --esg-data, or use --smoke-test")
        io = read_io_matrix(args.io_matrix)
        targets = targets_from_esg_frame(pd.read_csv(args.esg_data), io.index.astype(str))
        dataset = build_graph_dataset(io, targets=targets)
    model, history = train_model(dataset, epochs=args.epochs, hidden_dim=args.hidden_dim)
    metrics = evaluate_model(model, dataset)
    metrics_payload = {**metrics, "nodes": len(dataset.node_labels), "edges": int(dataset.edge_index.shape[1]), "final_loss": history[-1]}
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n")
    print(f"nodes={metrics_payload['nodes']} edges={metrics_payload['edges']} final_loss={metrics_payload['final_loss']:.4f}")
    print("metrics=" + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    print(f"metrics_json={args.metrics_output}")


if __name__ == "__main__":
    main()
