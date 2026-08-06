# ESG Analysis with GNNs

## Post-training interpretability (Migration PR 9)

The focused `src.evaluation` API extracts final- or per-layer GAT attention, runs additive node-feature perturbations in the already processed feature space, performs one-at-a-time feature ablation and frozen-model edge ablation, and traces bounded descriptive attention paths. Stored graph columns always retain their `source → target` direction. Canonical GAT coefficients use **incoming-target-per-head** normalization; the older `normalize_attention` utility remains a generic absolute-global helper and is not used by extraction. Reports explicitly record whether the fitted model uses economic edge weights: weighted GAT incorporates them in its existing normalization, while unweighted GAT preserves snapshot weights only as graph context and does not use them. Zero-weight ablation is rejected for unweighted models.

GPR-GNN reports softmax propagation-step coefficients separately: they are not edge attention. Attention-path scores are products of aggregated attention coefficients along directed, cycle-free paths, whereas perturbation results are prediction deltas; these quantities must be examined separately. Attention is not an explanation by itself and is not a causal effect. Perturbations and ablations measure local model sensitivity, not real-world interventions, and processed-space units may be standardized or transformed.

`export_explanation_bundle` writes applicable CSV/JSON reports and static PNG/SVG Matplotlib/NetworkX figures only when an output directory is explicitly provided. Run `python scripts/run_interpretability_smoke.py` for a synthetic validation-only check. Reports and figures identify model association/local sensitivity rather than a causal estimate. Derived targets retain the PR 5 construct-validity caveat; test explanations must not be used retrospectively for model selection or tuning. Real-data scientific conclusions remain outside synthetic smoke validation.

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

## Economic graph backends (Migration PR 3)

The repository now separates notebook-derived economic graph construction into two framework-neutral backends under `src.graphs`:

* `use_make` consumes already loaded USE and MAKE tables. It explicitly treats USE as commodity-by-sector and MAKE as sector-by-commodity by default, aligns common commodity and sector labels, computes notebook-derived `B` and `D` components, and builds the technical-coefficient matrix `A = B @ D` before thresholding retained directed sector-to-sector edges.
* `icio` consumes OECD/ICIO-style transaction data either as a square labelled transaction matrix or a wide table with an explicit source-industry column and destination-industry columns. ICIO industry codes are preserved as strings, including leading zeros, with only surrounding whitespace normalized.

Synthetic USE/MAKE example:

```python
import pandas as pd
from src.graphs import UseMakeGraphConfig, build_economic_graph

use = pd.DataFrame([[10, 2], [1, 8]], index=["C1", "C2"], columns=["S1", "S2"])
make = pd.DataFrame([[9, 1], [2, 7]], index=["S1", "S2"], columns=["C1", "C2"])
snapshot = build_economic_graph(
    "use_make",
    use,
    make,
    config=UseMakeGraphConfig(threshold_policy="absolute", threshold_value=0.0),
)
```

Synthetic ICIO example:

```python
import pandas as pd
from src.graphs import ICIOGraphConfig, build_economic_graph

icio = pd.DataFrame({"source_industry": ["01", "02"], "01": [5, 0], "02": [7, 3]})
snapshot = build_economic_graph(
    "icio",
    icio,
    config=ICIOGraphConfig(source_column="source_industry", include_self_loops=False),
)
```

Edges are directed from the source row entity to the destination column entity. Threshold configuration is explicit (`absolute` or `percentile`), and self-loops are included only when requested. Every `GraphSnapshot` keeps raw unmodified economic flows separately from model-ready `edge_weight`; configured model transforms include raw, `log1p`, row-normalized and globally standardized weights. Leontief computation uses an exact inverse when numerically appropriate and records any configured pseudo-inverse or regularized fallback in `LeontiefComputationReport`; a successful fallback is diagnostic only, not proof that the economic system is valid.

Real datasets must remain local and untracked under `DATA_POLICY.md`. Do not commit licensed provider files, generated graph outputs, credentials, or binary fixtures.

## Migration PR 4 feature-block research panel

This repository now includes a framework-neutral feature layer under `src/features/` and chronological split utilities under `src/splits/`. Feature construction is organized as typed blocks for structural graph statistics, macroeconomic series, environmental observations, ESG score aggregation, and leakage-resistant temporal history. Blocks emit rows keyed only by `(node_id, period)` for integer annual periods and are assembled deterministically by period and canonical node order.

Every feature carries `FeatureProvenance` metadata recording its final namespaced name, block, source dataset or graph backend, source statistic or column, transformation, period semantics, missing-value policy, units/notes, and whether preprocessing parameters were fitted. Assembly rejects feature-name collisions and duplicate keys rather than overwriting or multiplying rows.

Chronological splitting never shuffles rows. It assigns all rows from a given annual period to exactly one train, validation, or test split using either explicit period boundaries or ratios over sorted unique periods. The train periods precede validation periods, which precede test periods.

Train-only preprocessing is implemented by `TrainOnlyPreprocessor`. Missing-value policies are `error`, `preserve`, `train_mean`, `train_median`, and `constant`; scaling policies are `none` and `standard`. Imputation and scaling statistics are fitted only on the training mask, then reused unchanged for validation and test rows. The original missing-value mask is retained, zero-variance training features are recorded, and fitted state is JSON-serializable.

A small synthetic configuration is available at `configs/features.example.yaml`. For example, it references invented CSV fixtures under `tests/fixtures/features/`, enables macro/environmental/ESG/temporal blocks, configures chronological ratios, and selects train-median imputation with standard scaling.

The legacy `features_from_io_matrix()` helper in `src/data/preprocessing.py` remains for backward-compatible smoke tests and fits a scaler to one smoke matrix. It is not the train-only research preprocessing path. Migration PR 4 intentionally does not construct targets, labels, outcomes, or leakage-derived target proxies; target work is deferred to a later migration PR.

## Migration PR 5 target construction and direct leakage audit

Research target construction now lives in the framework-neutral `src/targets/` package. Observed targets are explicitly configured long-form outcome columns keyed by `(node_id, period)`, while derived targets are opt-in legacy proxy formulas reconstructed from named input features. The existing `targets_from_esg_frame()` helper in `src/data/preprocessing.py` remains only a backward-compatible single-target smoke helper; it is not the research target-construction or leakage-audit path.

Forecast horizons are annual integers. With horizon `0`, feature period `t` aligns to an observation at `t`; with horizon `1`, feature period `t` aligns to the observed outcome at `t+1`, while the output key remains the feature row key. Target provenance records the target name, observed/derived status, source columns or source features, formula or transformation, horizon semantics, missing-value policy, units when known, and construct-validity limitations. No target is automatically treated as independently valid ground truth.

Notebook-derived proxies such as ESG risk, economic impact, volatility, transition cost, and compliance probability are mechanically constructed from model input features. That direct lineage creates leakage if those source features are also used for supervised modelling, so derived proxies are disabled by default and require `allow_derived_targets=True` for explicit reproduction. The leakage auditor supports `error`, which rejects direct findings, and `drop_declared_sources`, which removes only direct target columns and the union of declared source features for selected derived targets; unrelated features are preserved.

Minimal synthetic examples are provided under `tests/fixtures/targets/`, with a text-only configuration at `configs/targets.example.yaml`:

```python
import pandas as pd
from src.targets import build_observed_target_block, audit_and_filter_features

observed = pd.read_csv("tests/fixtures/targets/observed_esg_outcomes.csv")
block = build_observed_target_block(
    observed,
    target_columns=["esg_score"],
    forecast_horizon=0,
    missing_policy="preserve",
)
safe_features, removed, audit_report, reasons = audit_and_filter_features(
    ["environmental_score", "safe_feature"],
    block,
    policy="drop_declared_sources",
)
```

## Migration PR 6 model registry

Migration PR 6 adds five canonical, pure-PyTorch model definitions behind a deterministic registry: `mlp`, `gcn`, `gat`, `weighted_gat`, and `bidirectional_gnn`.  All new registry models share the same forward interface:

```python
forward(x, edge_index=None, edge_weight=None, *, return_aux=False)
```

The graph convention is directed and source-to-target: `edge_index[0]` stores source nodes and `edge_index[1]` stores target nodes, so a message on `(source, target)` moves information from the source node to the target node.  Directed economic graphs are never silently symmetrized.  The feature matrix supplied to a model must already have passed the Migration PR 5 direct-lineage leakage policy; models do not construct targets or filter leakage internally.

Every registry model returns a `ModelOutput` with two-dimensional `predictions`, ordered `target_names`, two-dimensional `node_embeddings`, and optional `auxiliary` tensors.  A shared `NodeRegressionHead` preserves the configured target order and ends in a linear layer with identity output transformation by default.  The `mlp` model is the required non-graph baseline for fair graph-model comparison.  The `gcn` model uses direction-preserving incoming-neighbor mean aggregation with a self path.  The unweighted `gat` computes destination-wise, per-head attention over stored directed edges only and ignores economic edge weights.  The `weighted_gat` uses the same attention logits but multiplies each edge's unnormalized attention by the explicit non-negative `edge_weight` before normalizing incoming mass for each destination and head:

```text
unnormalized_attention(edge, head) = exp(stabilized_attention_logit(edge, head)) * edge_weight(edge)
normalized_attention = unnormalized_attention / sum_incoming_unnormalized_attention_for_target_head
```

The `bidirectional_gnn` uses separately parameterized stored-direction and `edge_index.flip(0)` reverse-direction branches, then combines `forward_embedding` and `reverse_embedding` by concatenation followed by projection.  Training comparison, orchestration, checkpointing, and experiment tracking remain deferred to Migration PR 8.  The legacy smoke wrapper remains available through `src.models.gnn_models.EconomicESGGNN` and still returns the historical dictionary keys used by `scripts/run_pipeline.py`.

```python
import torch
from src.models import ModelConfig, build_model

x = torch.randn(4, 8)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
edge_weight = torch.tensor([1.0, 0.5, 2.0])

config = ModelConfig(
    name="weighted_gat",
    input_dim=8,
    hidden_dim=16,
    num_layers=2,
    dropout=0.1,
    attention_heads=2,
    target_names=(
        "target__observed_esg_score",
        "target__observed_real_output_growth",
    ),
    seed=7,
)

model = build_model(config)
output = model(
    x,
    edge_index,
    edge_weight,
    return_aux=True,
)
```

Registry helpers are exposed from `src.models`:

```python
from src.models import available_models, build_model, get_model_capabilities
```


## Migration PR 7 heterophily and anomaly models

The supervised registry now appends `gpr_gnn`. For incoming weighted propagation `P`, its retained states satisfy `h_k = (1-alpha) P(h_(k-1)) + alpha h_0`, and the final embedding is `sum_k softmax(gamma)[k] h_k`. These learned propagation-step coefficients are **not attention**. `stored` follows supplied directed edges; `reverse` flips endpoints while preserving weight-column alignment. No mode silently symmetrizes the graph. `build_gpr_ablation_configs` produces the configuration-only alpha/depth/direction Cartesian grid in caller-supplied order; it neither trains models nor chooses a winner.

`EconomicGAE` is a separate unsupervised workflow, not a supervised registry model. Its two-stage directed weighted encoder feeds a linear-ended feature decoder and an asymmetric structure decoder with separate source and target projections. Training loss is weighted feature MSE plus directed-edge binary cross entropy on explicitly supplied positives and negatives. Raw per-node scores are `feature_error + coefficient * embedding_consistency_error` (`0.5` by default). They are reconstruction indicators, not verified ground truth; no threshold or binary label is fitted. Training orchestration, comparison, and threshold evaluation remain deferred to Migration PR 8.
# Reproducible chronological experiments (Migration PR 8)

Research training now lives in `src.experiments` and operates on one directed graph snapshot per annual period. Parameters are updated from train periods only; validation loss alone controls early stopping and plateau scheduling; the detached in-memory best checkpoint is restored before a single final train/validation/test evaluation. Test data must never be used for model, epoch, scheduler, threshold, or configuration selection.

Targets retain missing values and a boolean observation mask. Masked per-target MSE values are combined with normalized positive target weights (equal by default). Reports provide per-target MAE, RMSE, and R²; R² is `null` with a reason for fewer than two observations or zero variance, and macro R² excludes undefined targets. `TrainMeanBaseline` fits each target mean solely from observed train labels and is distinct from the registry MLP.

Preprocessing must already have been fit on train periods by PR 4, and selected features must already pass PR 5 direct-lineage leakage auditing. Derived targets are not independently observed ground truth. EconomicGAE remains a separate unsupervised workflow; optional validation-quantile flags are heuristic reconstruction flags unless independent anomaly labels exist.

Load strict, text-only configuration from `configs/experiment.example.yaml`. Run the synthetic CPU check with `python scripts/run_experiment_smoke.py`. The legacy `python scripts/run_pipeline.py --smoke-test` workflow and `src.training` API remain supported unchanged. Checkpoints and predictions are held in memory and no output is written unless an explicit runtime path is supplied; only load PyTorch checkpoint files from trusted sources.
