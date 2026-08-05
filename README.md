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


## Security, data, and local configuration

Copy `.env.example` to `.env` only on your local machine and fill in secrets or private paths there. `.env` and local credential files are intentionally ignored; never commit API keys, passwords, tokens, database credentials, private URLs, raw data, caches, SQLite databases, checkpoints, or generated run artifacts.

The optional NewsAPI sentiment path reads `NEWS_API_KEY` from the environment and should skip cleanly when the variable is absent. Dataset paths can also be supplied with environment variables documented in `.env.example`, or by CLI arguments where available. See `DATA_POLICY.md` for the full repository policy and `configs/datasets.yaml` for the machine-readable external dataset inventory.

Credential rotation remains mandatory for any previously exposed secret. Removing a value from the current tree does not remove it from git history; any history rewrite with `git filter-repo` or BFG must be planned separately because it changes commit hashes.

## Smoke-test execution

Run the full extracted workflow on deterministic in-repository sample data:

```bash
python scripts/run_pipeline.py --smoke-test --epochs 2 --hidden-dim 8 --metrics-output pipeline_metrics.json
```

This mode does not use external data, API credentials, raw datasets, or databases,
and writes metrics to `pipeline_metrics.json` unless `--metrics-output` is provided.

## Running with local research data

Keep private datasets and credentials outside git. Then provide local file paths:

```bash
python scripts/run_pipeline.py \
  --io-matrix /path/to/REAL_USE.xlsx \
  --esg-data /path/to/esg_scores.csv \
  --epochs 300 \
  --hidden-dim 64
```

The I/O matrix may be CSV, XLS, or XLSX with industries in both rows and columns; XLS/XLSX reading uses `openpyxl`.
The ESG CSV should include either an `esg_score`, `esg`, or `score` column, or
all three pillar columns: `environmental`, `social`, and `governance`. If it also
contains `sector`, `industry`, `node`, or `code`, targets are aligned to graph
nodes by that column; otherwise the score row count must match the graph node count.

## Tests

```bash
python -m pytest tests
```

## Notes for future migration

The original notebooks are intentionally preserved. Remaining migration work is
to move additional notebook-only visualizations, full temporal validation reports,
NewsAPI sentiment ingestion, and large-dataset experiment configuration into
versioned modules without committing credentials or raw data.

## Typed data loaders (Migration PR 2)

This repository now includes dependency-light, dataclass-based contracts for validating dataset inventory entries and local data-loader configuration. The loaders are intended to fail loudly on invalid schemas instead of inventing synthetic replacement values or silently imputing provider data.

Real ESG, BEA, emissions, news, or I/O datasets must be placed locally according to `DATA_POLICY.md`; do not commit licensed provider files, credentials, API keys, personal absolute paths, or downloaded raw datasets.

Supported loader formats are CSV, XLS, and XLSX for BEA-style tables, and CSV, XLS, and XLSX for ESG score files. `configs/pipeline.example.yaml` shows repository-relative synthetic example paths only.

### BEA loader example

```python
from src.data.bea import load_bea_table

result = load_bea_table("tests/fixtures/bea/header_first.csv", header_row=0)
print(result.schema.time_value_columns)
print(result.data.head())
```

The BEA loader can use an explicit header row or detect one from a bounded range. It normalizes column names, preserves source industry codes as strings, detects year/period columns, strips footnote markers, converts comma-formatted numeric cells, and treats suppressed values such as `(D)` or `--` as missing. Ambiguous industry columns, missing industry columns, empty files, and files without usable time/value columns raise validation errors.

### ESG loader example

```python
from src.data.contracts import MissingValuePolicy
from src.data.esg import load_esg_scores

result = load_esg_scores(
    "tests/fixtures/esg/valid_esg.csv",
    missing_score_policy=MissingValuePolicy.PRESERVE,
)
print(result.column_mapping)
print(result.row_count, result.duplicate_count)
```

The ESG loader validates required entity, industry, and overall-score columns through configurable aliases. It rejects empty files, missing required columns, duplicate entity-period records unless an explicit supported duplicate policy is provided, non-numeric or infinite scores, scores outside the configured range, and missing industry identifiers. Missing scores are handled explicitly with `error`, `drop`, or `preserve` policies; the loader does not impute ESG values or construct model targets.

### Mapping coverage example

```python
from src.data.mapping import map_industries

mapped, report = map_industries(frame, "industry", {"Utilities": "UTIL"}, min_coverage=0.8)
print(report.row_coverage_ratio, report.unmatched_values)
```

The mapping utility maps source industry names or codes to canonical IDs while preserving unmatched values and row counts. Its coverage report includes total rows, unique source industries, matched and unmatched row counts, matched and unmatched unique industry counts, coverage ratios, and sorted unmatched values. Strict mode raises a validation error when coverage is below the configured threshold. Large production mapping dictionaries are intentionally deferred to a later industry-mapping PR.
