# ESG Analysis with GNNs

This repository contains notebook experiments and reusable Python modules for
Environmental, Social, and Governance (ESG) analysis with graph neural networks.
The reusable pipeline is extracted from the most complete end-to-end notebook,
`ESG Analysis final ver.ipynb`, which combines data preprocessing, economic
input-output graph construction, GNN training, evaluation, and visualization.

## Project structure

```text
ESG-Analysis-with-GNNs/
├── ESG Analysis final ver.ipynb        # canonical source notebook
├── Data Preprocess.ipynb               # preprocessing exploration
├── Real GDP data + real env data ver3.ipynb
├── Heterophily ver6.ipynb
├── Attention_weight_visualization_FULL.py
├── scripts/
│   ├── run_pipeline.py                 # reusable CLI pipeline
│   ├── export_news_db.py
│   └── inspect_bea_headers.py
├── src/
│   ├── data/preprocessing.py           # I/O matrix, ESG target, graph features
│   ├── models/gnn_models.py            # EconomicESGGNN and fallback graph conv
│   ├── training.py                     # train/evaluate helpers
│   └── visualization/attention.py
├── tests/test_pipeline.py
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`torch-geometric` is optional. If it is installed, `EconomicESGGNN` uses PyG GCN
layers. If it is not installed, the model automatically uses a dense weighted
message-passing fallback so the pipeline and tests can run in constrained
environments.

## Smoke-test execution

Run the full extracted workflow on deterministic in-repository sample data:

```bash
python scripts/run_pipeline.py --smoke-test --epochs 2 --hidden-dim 8
# writes results/metrics/smoke_test_metrics.json by default
```

This mode does not use external data, API credentials, raw datasets, databases,
or generated output files.

## Running with local research data

Keep private datasets and credentials outside git. Then provide local file paths:

```bash
python scripts/run_pipeline.py \
  --io-matrix /path/to/REAL_USE.xlsx \
  --esg-data /path/to/esg_scores.csv \
  --epochs 300 \
  --hidden-dim 64 \
  --metrics-output results/metrics/local_metrics.json
```

The I/O matrix may be CSV, XLS, or XLSX with industries in both rows and columns.
The ESG CSV should include either an `esg_score`, `esg`, or `score` column, or
all three pillar columns: `environmental`, `social`, and `governance`. If it also
contains `sector`, `industry`, `node`, or `code`, targets are aligned to graph
nodes by that column; otherwise scores are resized to the node count for local
experimentation.

## Tests

```bash
python -m pytest tests
```

## Notes for future migration

The original notebooks are intentionally preserved. Remaining migration work is
to move additional notebook-only visualizations, full temporal validation reports,
NewsAPI sentiment ingestion, and large-dataset experiment configuration into
versioned modules without committing credentials or raw data.
