# ESG Analysis with GNNs

A migrated, configuration-driven research codebase for directed economic graphs, multi-target ESG experiments, anomaly analysis, and non-causal model interpretation.

## Start here

```bash
python scripts/run_pipeline.py --smoke-test --epochs 2 --hidden-dim 8
```

`src/` public APIs and `scripts/` are the canonical executable interfaces. Repository layout:

- `src/` — typed data, graphs, features, targets, models, experiments, evaluation, and visualization.
- `scripts/` — canonical smoke and local utility entry points.
- `configs/` — typed example configuration without datasets or credentials.
- `notebooks/` — five thin, output-free synthetic examples; run in order `00` through `04`.
- `archive/research/` — historical provenance only; **never import or execute it**.
- `docs/` — [research lineage](docs/RESEARCH_LINEAGE.md), [machine-readable lineage](docs/research_lineage.yaml), and [migration completion](docs/MIGRATION_COMPLETION.md).

Launch notebooks with the canonical cross-platform command:

```bash
python scripts/launch_notebooks.py
```

The launcher starts Jupyter from the repository root and makes `src` and `examples` importable in notebook kernels. It does not execute archived notebooks, insert paths inside notebook cells, or edit notebook files. Then follow `00_end_to_end_overview.ipynb`, `01_data_pipeline.ipynb`, `02_model_comparison.ipynb`, `03_temporal_evaluation.ipynb`, and `04_interpretability.ipynb`.

## Active checks

```bash
python -m pytest tests
python scripts/security_inventory_check.py
python scripts/run_pipeline.py --smoke-test --epochs 2 --hidden-dim 8
python scripts/run_experiment_smoke.py
python scripts/run_interpretability_smoke.py
python scripts/run_notebook_smoke.py
```

Real datasets must be separately obtained, licensed, and configured outside Git. Synthetic success is not scientific reproduction, validation, or evidence of causality. The previously exposed NewsAPI credential still requires manual revocation or rotation; archival does not erase Git history. Archived notebooks are unsupported historical material, not usage documentation.
