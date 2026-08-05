# ESG Analysis with GNNs

This repository provides a lightweight project scaffold for experimenting with
Environmental, Social, and Governance (ESG) analysis using graph neural networks.

## Project structure

```text
ESG-Analysis-with-GNNs/
├── README.md
├── requirements.txt
├── src/
│   ├── data_preprocessing.py
│   ├── graph_construction.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── esg_gnn_analysis.ipynb
└── .gitignore
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
```

The default scripts run on a small synthetic dataset so the scaffold can be
executed before real ESG data is added.


Binary outputs such as PNG figures and PDF reports should be generated locally and are not tracked in this scaffold.
