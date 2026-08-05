# ESG Analysis with GNNs

A simplified scaffold for ESG analysis with graph neural networks. The repository
is organized around data ingestion, model components, evaluation utilities,
visualization helpers, scripts, notebooks, and generated outputs.

## Project structure

```text
ESG-Analysis-with-GNNs/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   ├── external_data.py
│   │   └── news_sentiment.py
│   ├── models/
│   │   ├── gnn_models.py
│   │   └── autoencoder.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── interpretability.py
│   └── visualization/
│       └── attention.py
├── scripts/
│   ├── export_news_db.py
│   └── inspect_bea_headers.py
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_model_comparison.ipynb
│   ├── 03_temporal_evaluation.ipynb
│   └── 04_interpretability.ipynb
├── results/
│   ├── figures/
│   └── metrics/
└── reports/
    └── final_presentation.pdf
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/export_news_db.py
python scripts/inspect_bea_headers.py
```

The modules use standard-library baselines where possible so the scaffold can be
smoke-tested before full GNN dependencies and production ESG datasets are added.
