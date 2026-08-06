# Active reproducible notebooks

Run these thin synthetic examples in order:

1. `00_end_to_end_overview.ipynb` — architecture, one supervised run, and baseline.
2. `01_data_pipeline.ipynb` — typed snapshots, ordering, masks, and directed weights.
3. `02_model_comparison.ipynb` — fixed three-model smoke comparison.
4. `03_temporal_evaluation.ipynb` — chronological training history and split metrics.
5. `04_interpretability.ipynb` — directed attention and local sensitivity.

From the repository root, with Python 3.11+ and `requirements.txt` installed, run:

```bash
python scripts/launch_notebooks.py
```

This launcher starts Jupyter from the repository root and makes `src` and `examples` importable in notebook kernels. It does not execute archived notebooks, insert paths inside notebook cells, or edit notebook files. Committed notebooks contain no outputs. CI runs fresh-kernel, temporary in-memory copies using synthetic data only. Supply real datasets outside Git through typed configuration paths. Do not commit datasets, executed/generated notebooks, figures, or metrics; the reviewed deterministic `results/summary_metrics.*` and README SVG are the only documentation exception. Historical notebooks are preserved under `archive/research/` and are unsupported, non-executable provenance artifacts. Never place credentials in notebook cells.
