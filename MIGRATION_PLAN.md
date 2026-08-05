# Research Notebook Migration Plan

## Scope and method

This document is based on static inspection of the 14 research files currently tracked on `main` at commit `718f0a1da580aa41939b8fcc0ed18d892a573988`. No external API requests, dataset downloads, or long model-training runs were performed. The goal is to identify the strongest source for each component, document the research lineage, and split migration into reviewable pull requests.

The existing `src/` pipeline should remain available as a lightweight smoke-test harness until the notebook-derived implementation is migrated and validated. It should not yet be treated as a complete reproduction of the research notebooks.

## File-by-file inventory

| File | Purpose and status | Data sources | Preprocessing and graph construction | Model architecture and training | Evaluation and visualization | Key dependencies | Hard-coded paths, credentials, or other risks |
|---|---|---|---|---|---|---|---|
| `API added ver5.ipynb` | Integrated ESG/economic GNN experiment that adds news sentiment. Substantial and executable-looking, but experimental and superseded by later heterophily/time-series variants. | ESG CSV; USE/MAKE Excel tables; BEA GDP/output files; environmental-tax data; emissions data; NewsAPI. | Industry mapping, ESG normalization, USE/MAKE alignment, technical-coefficient matrix, Leontief inverse, percentile-filtered weighted graph, macro/environment/risk/sentiment feature fusion. | GCN/GAT baselines, weighted GAT, hybrid GNN, improved economic GNN, `EconomicESGGNN`, bidirectional GNN, ESG channel gating, economic-prior fusion and multi-task prediction heads. | Multi-model comparisons and task metrics; limited temporal validation. | PyTorch, PyG, pandas, NumPy, scikit-learn, openpyxl, NLTK/VADER, newsapi-python, matplotlib/seaborn. | Contains a literal NewsAPI credential and local filenames. Credential must be revoked and removed from history. Dummy-data fallbacks can hide failed real-data ingestion. |
| `Attention weight + visualization ver8.ipynb` | Nearly empty shell containing imports only. Incomplete and not a source of implementation. | None beyond imported libraries. | None. | None. | None. | PyTorch, PyG, plotting, NLTK/NewsAPI imports. | Treat as superseded by `Attention_weight_visualization_FULL.py`; remove or archive after verification. |
| `Attention_weight_visualization_FULL.py` | Exported notebook/report text containing attention extraction and shock-propagation visualization. Valuable as a reference, but not valid clean Python in its current form. | Same integrated ESG, I/O, macro, environment and sentiment sources as the main experiment branch. | Repeats the integrated preprocessor and weighted directed graph construction. | Weighted GAT and other model definitions expose layer attention; includes global attention and attention-returning forwards. | Attention-weight inspection and `visualize_shock_propagation`, directed subgraph tracing, top-k affected sectors and network plots. | PyTorch, PyG, NetworkX, matplotlib, seaborn, pandas, NumPy, NLTK/NewsAPI. | Notebook-export artifacts, page breaks, malformed spacing and execution output make it non-importable. Local paths remain. A prior credential was removed from the current file, but repository history still requires credential rotation. |
| `Data Preprocess.ipynb` | Data-engineering prototype for OECD/ICIO and BEA series. Strong source for mappings and exploratory feature generation; not an end-to-end canonical model. | OECD/ICIO `USA2020ttl.csv`; BEA real value added and related industry series. | Wide I/O matrix to edge list, positive-flow filtering, ICIO code cleaning, extensive BEA-industry-to-ICIO mapping, growth/volatility/recent-performance features, ESG shock heuristics and clustering exploration. | Mostly preprocessing and exploratory analysis; not the authoritative integrated GNN. | Distribution plots, clustering, industry performance and shock ranking. | pandas, NumPy, scikit-learn, matplotlib, seaborn. | Multiple absolute absolute personal paths paths. Mapping rules are hand-written and need versioned tests. Heuristic shock labels risk becoming targets without independent ground truth. |
| `ESG Analysis final ver.ipynb` | Most complete single integrated notebook. Best orchestration and model-comparison reference, but not automatically the best source for every component. | ESG CSV; multi-year USE/MAKE tables; real GDP/gross-output data; environmental taxes; emissions; NewsAPI sentiment. | Chronological split, temporal trend/volatility features, USE/MAKE-derived A matrix, Leontief features, percentile-weighted edges, ESG/macro/environment/risk/sentiment fusion. | `GPRGNNLayer`/`GPRGNN`, graph autoencoder (`EconomicGAE`), GCN/GAT baselines, weighted GAT, hybrid GNN, improved economic GNN, `EconomicESGGNN`, ESG gating, economic-prior fusion and multi-task heads. Trains multiple models, generally up to 300 epochs, with AdamW, cosine scheduling and validation tracking. | MAE/RMSE/R², accuracy for thresholded tasks, train/validation histories, model comparisons, anomaly detection. | Full scientific Python/PyTorch/PyG stack plus NewsAPI/NLTK and Excel support. | Contains a literal NewsAPI credential in notebook cells. Target construction derives several labels from input features, creating leakage/construct-validity risk. Duplicate training paths and extensive mutable notebook state. |
| `Epoch 300 + IO data ver2.ipynb` | Earlier integrated pipeline emphasizing corrected USE/MAKE alignment and 300-epoch experiments. Useful predecessor and graph-construction reference. | ESG CSV; USE/MAKE Excel tables; related macro/environment files. | Intersects and sorts common USE/MAKE sectors, computes B and D matrices, forms A = B·D, Leontief inverse, median/percentile edge threshold and normalized edge weights. | Earlier versions of economic GCN/GAT, weighted GAT, hybrid and integrated ESG models; long fixed-epoch comparison. | Loss curves and comparative regression metrics. | PyTorch, PyG, pandas, NumPy, scikit-learn, openpyxl, plotting. | Local filenames and dummy fallbacks. Uses the latest selected year for much of the graph even when several years are listed; temporal semantics are weaker than later versions. |
| `Heterophily ver6.ipynb` | Adds heterophily-oriented propagation to the integrated pipeline. Best source for the GPR-GNN experiment. | Same ESG, I/O, macro, environment and sentiment sources as the main integrated branch. | Reuses real-data preprocessor and weighted economic graph. | Introduces `GPRGNNLayer` and `GPRGNN` with generalized PageRank propagation parameters (`alpha`, `K`) alongside GCN/GAT/weighted/hybrid/integrated models. | Comparative model training and task metrics. | PyTorch, PyG message passing utilities, scientific Python stack, NewsAPI/NLTK. | Contains a literal NewsAPI credential. GPR parameters and initialization need configuration and ablation tests; no clean isolated module. |
| `Real GDP data + real env data ver3.ipynb` | Intermediate version that replaces more synthetic macro/environment inputs with real BEA and emissions/tax data. Best source for the first real-data integration attempt. | ESG CSV; USE/MAKE tables; BEA GDP/gross-output files; emissions workbook; environmental-tax CSV. | Industry mapping, BEA header handling, macro/environment feature alignment, I/O graph and Leontief features. | Integrated GCN/GAT family and multi-task prediction; earlier than sentiment, heterophily and temporal additions. | Standard regression/task metrics and plots. | PyTorch, PyG, pandas, NumPy, scikit-learn, openpyxl, matplotlib/seaborn. | Fragile header assumptions and local filenames. Broad exception handling can silently replace unavailable data with demo values. |
| `db_to_csv.ipynb` | Small utility for exporting a SQLite `articles` table to CSV. Auxiliary and currently demonstrated as failing because the table is absent. | `news_monitoring.db`, table `articles`. | SQL query to pandas DataFrame, UTF-8 CSV export. | None. | Row count and success/error logging only. | sqlite3, pandas. | Assumes fixed database/table/output names. Should become a tested CLI utility with schema validation or be removed if unused. |
| `differentiation top 10.ipynb` | Parallel late-stage US-focused pipeline using OECD/ICIO plus six BEA series and a differentiated integrated architecture. Strong source for robust US data ingestion and feature differentiation. | OECD/ICIO USA I/O table; six BEA series including real value added, real intermediate input, price indexes and gross output. | Robust multi-format BEA readers, dynamic time-column detection, extensive name-to-ICIO mapping, latest-period selection, feature aggregation, directed edge construction. | `IntegratedESGProcessor`, bidirectional message passing, ESG channel gating and `IntegratedESGGNN`. Intended to rank/differentiate top sectors. | Top-sector differentiation/ranking and model outputs; less complete reproducible validation than the final integrated notebook. | pandas, NumPy, scikit-learn, PyTorch, PyG, openpyxl. | Many absolute absolute personal paths paths. Large hand-maintained mapping dictionary. This is a parallel US/ICIO branch and should not be merged blindly with the USE/MAKE branch. |
| `graphsage_ppi.py` | Standalone GraphSAGE/PPI tutorial or benchmark. Auxiliary and unrelated to the ESG economic graph except as a sampling example. | PyG PPI dataset downloaded under `./data/PPI`. | Neighbor sampling with fixed fanouts. | Two-layer GraphSAGE, binary cross-entropy, Adam, 10 epochs. | Micro-F1 on validation/test. | PyTorch, PyG, scikit-learn. | Executes data download and training at import time. Different task (multi-label protein graph), so it must not be treated as an ESG baseline without explicit justification. |
| `header file check.ipynb` | Diagnostic notebook used to discover BEA Excel header rows and column structure. Auxiliary but useful for fixture design. | BEA Excel workbooks. | Tries multiple header offsets and inspects shapes, columns and sample rows. | None. | Console diagnostics only. | pandas, openpyxl. | Contains local file assumptions and large stored outputs/warnings. Convert useful cases into automated parser tests, then archive. |
| `time series ver9.ipynb` | Advanced temporal version and likely immediate predecessor/sibling of the final notebook. Best source for chronological splitting and temporal features. | Multi-year USE/MAKE tables plus ESG, macro, environment and sentiment inputs. | `TimeSeriesDataSplitter`, chronological train/validation/test years, trend/volatility/mean temporal features, split-specific graph/features, Leontief and integrated features. | Includes GPR-GNN, `EconomicGAE`, GCN/GAT/weighted/hybrid/integrated models and multi-task training. | Split-aware validation/test evaluation, histories and anomaly analysis. | Same full PyTorch/PyG/data stack as the final notebook. | Contains a literal NewsAPI credential. Some scalers appear fit separately on split data, which must be checked for train/validation consistency. Large duplication with `ESG Analysis final ver.ipynb`. |
| `top 4 update ver.ipynb` | Earlier Korean-industry-mapping experiment intended to compare a smaller set of models. Incomplete/broken and not canonical. | ESG CSV and Korean USE/MAKE tables. | Very extensive ESG-industry-to-Korean-I/O mapping and direct I/O normalization. | Economic GNN, weighted GAT, hybrid, bidirectional, gating, prior fusion, improved/economic ESG variants. | Intended top-four model comparison. | PyTorch, PyG, pandas, NumPy, openpyxl, plotting. | Contains a recorded `SyntaxError` (`[` never closed). Mapping and preprocessing differ from the later US/ICIO branch. Keep only as a source of mapping ideas until repaired and tested. |

## Likely development lineage

The filenames and code deltas suggest two partly overlapping research branches rather than one perfectly linear sequence.

### Integrated USE/MAKE branch

1. `top 4 update ver.ipynb` — early integrated Korean/USE-MAKE model comparison; currently syntactically broken.
2. `Epoch 300 + IO data ver2.ipynb` — fixes USE/MAKE sector alignment and formalizes longer model training.
3. `Real GDP data + real env data ver3.ipynb` — adds real macroeconomic and environmental inputs.
4. `API added ver5.ipynb` — adds NewsAPI/VADER sentiment and bidirectional modeling.
5. `Heterophily ver6.ipynb` — adds GPR-GNN generalized PageRank propagation.
6. `Attention_weight_visualization_FULL.py` / `Attention weight + visualization ver8.ipynb` — adds attention extraction and shock-propagation visualization; the notebook itself is only a shell, while the exported file contains the useful logic.
7. `time series ver9.ipynb` — adds chronological splits, temporal features and graph-autoencoder anomaly detection.
8. `ESG Analysis final ver.ipynb` — consolidates temporal, heterophily, anomaly, multi-model and multi-task components.

### US OECD/ICIO and BEA branch

1. `Data Preprocess.ipynb` — explores OECD/ICIO edges, BEA mappings and industry shock features.
2. `header file check.ipynb` — diagnoses BEA workbook structures.
3. `differentiation top 10.ipynb` — integrates the ICIO graph with six BEA indicators and a bidirectional/gated ESG model for sector differentiation.

### Auxiliary branch

- `db_to_csv.ipynb` supports a news database workflow.
- `graphsage_ppi.py` is a generic GraphSAGE benchmark/tutorial and not part of the ESG research lineage.

## Best source by component

| Component | Recommended source | Rationale |
|---|---|---|
| Data-ingestion framework | `differentiation top 10.ipynb` plus `header file check.ipynb` | Most defensive BEA reader and strongest evidence for variable header/time-column handling. Convert diagnostic cases into fixtures. |
| ESG ingestion and normalization | `ESG Analysis final ver.ipynb`, audited against `API added ver5.ipynb` | Most integrated schema, but score/grade normalization and missing-value policy require explicit validation. |
| USE/MAKE economic graph | `Epoch 300 + IO data ver2.ipynb` and later final/time-series copies | Common-sector alignment, B/D matrices, A = B·D and Leontief inverse are clearer than the early direct-normalization approach. |
| OECD/ICIO graph backend | `Data Preprocess.ipynb` / `differentiation top 10.ipynb` | Clean wide-table-to-edge-list logic and ICIO code normalization. This should be a separate configurable backend, not mixed implicitly with USE/MAKE. |
| ESG target construction | `ESG Analysis final ver.ipynb`, but redesign required | It has the broadest multi-task target set; however, targets derived directly from input features need a leakage and construct-validity audit before reuse. |
| Baseline model comparison | `ESG Analysis final ver.ipynb` | Widest registry: GCN, GAT, weighted GAT, hybrid, improved economic GNN, integrated ESG GNN and GPR-GNN. |
| Bidirectional upstream/downstream modeling | `differentiation top 10.ipynb`, cross-checked with `API added ver5.ipynb` | Clearest differentiated architecture for directed economic propagation. |
| Heterophily handling | `Heterophily ver6.ipynb` | Explicit GPR-GNN implementation with `alpha` and propagation depth `K`. |
| Temporal evaluation | `time series ver9.ipynb`, then final notebook | Explicit chronological split and temporal trend/volatility features. |
| Anomaly detection | `time series ver9.ipynb` / final notebook | `EconomicGAE` is present as a distinct unsupervised graph-autoencoder path. |
| Attention and shock propagation | `Attention_weight_visualization_FULL.py` | Contains attention-returning forwards and shock-propagation plotting logic, but must be manually extracted into valid Python. |
| Feature differentiation/top-sector ranking | `differentiation top 10.ipynb` | Strongest BEA/ICIO feature set and explicit differentiation objective. |
| Industry mappings | `differentiation top 10.ipynb` and `Data Preprocess.ipynb` | Broad US mappings; store in versioned data files with tests rather than Python literals. Korean mappings from `top 4 update ver.ipynb` should remain a separate profile. |
| End-to-end orchestration | `ESG Analysis final ver.ipynb` | Most complete research runner, while individual modules should come from the stronger sources above. |

## Recommended canonical pipeline

The canonical implementation should be assembled from components rather than copied wholesale from one notebook.

1. **Configuration and data contracts**
   - Use typed configuration for dataset paths, country/profile, years, graph threshold, random seed, model and training settings.
   - Define a common `IndustryPanel`/`GraphSnapshot` contract with node IDs, period, feature provenance, target provenance, edges and masks.
   - Keep `use_make` and `icio` graph builders as explicit interchangeable backends.

2. **Data ingestion**
   - Port robust BEA parsing and time-column detection from `differentiation top 10.ipynb`.
   - Convert `header file check.ipynb` examples into small test fixtures.
   - Load ESG, GDP/output, environmental tax/emissions and optional sentiment through independent adapters.
   - Make NewsAPI enrichment optional and cache raw responses outside git.

3. **Industry mapping**
   - Move US ICIO and Korean I/O mappings into CSV/YAML files with source, version and one-to-many mapping rules.
   - Emit coverage reports and fail on unexpectedly low mapping coverage instead of silently filling all unmatched sectors.

4. **Graph construction**
   - For USE/MAKE: align common sectors; compute B, D and A; validate dimensions/non-negativity; compute Leontief inverse with documented pseudo-inverse fallback.
   - For ICIO: normalize codes and create a directed weighted edge list from positive flows.
   - Make percentile thresholding a configuration and report retained-edge density. Preserve raw economic weights separately from transformed model weights.

5. **Feature engineering**
   - Separate structural I/O features, macro features, environmental features, ESG features, temporal features and sentiment features.
   - Fit all scalers on training periods only and reuse them for validation/test.
   - Attach provenance metadata to every feature block.

6. **Targets and leakage controls**
   - Prefer externally observed ESG or economic outcomes.
   - Treat heuristic targets such as transition cost, compliance probability and shock score as derived research indices, not independent ground truth.
   - Prevent any target-generating column from reappearing unchanged in model inputs.
   - Add masks for missing labels rather than mean-filling labels.

7. **Model registry**
   - Baselines: linear/MLP, GCN, GAT and weighted GAT.
   - Directed model: bidirectional upstream/downstream GNN.
   - Heterophily model: GPR-GNN.
   - Integrated model: ESG channel gating plus optional economic-prior fusion.
   - Auxiliary anomaly model: EconomicGAE.
   - Standardize every model output behind a common embedding and task-head interface.

8. **Training and evaluation**
   - Chronological train/validation/test splits from `time series ver9.ipynb`.
   - Seeded training, early stopping, checkpointing and configuration snapshots.
   - Report MAE, RMSE and R² for regression; ROC-AUC/F1/accuracy only for genuinely classified targets; compare against non-graph baselines.
   - Save per-period and per-sector predictions, not only aggregate metrics.

9. **Interpretability**
   - Extract valid attention-returning layers and shock-propagation logic from `Attention_weight_visualization_FULL.py`.
   - Label attention visualizations as model explanations, not causal economic effects.
   - Add perturbation-based sensitivity and ablation tests alongside attention plots.

10. **Reproducible execution**
    - Keep the existing smoke-test CLI, then add research configs such as `configs/use_make.yaml` and `configs/icio_bea.yaml`.
    - Store generated metrics/figures under ignored run directories with a manifest.
    - Keep original notebooks read-only during migration; later replace them with thin notebooks that import tested modules.

## Duplicated, contradictory, incomplete or potentially broken areas

- `RealDataPreprocessor`, model classes and training loops are copied across many notebooks with small undocumented differences.
- The same model names sometimes have different forwards or output types, including attention tuples versus plain embeddings.
- `BidirectionalGNN`/`BiDirectionalGNN` naming and implementation differ between branches.
- USE/MAKE graph construction and direct USE-row normalization are both present; they are not equivalent.
- US ICIO codes and Korean I/O sector mappings are mixed across versions and must become explicit dataset profiles.
- Several notebooks list multiple years but build the final graph from only the latest year.
- Temporal scalers and feature transformations need verification to ensure no validation/test fitting.
- Some task heads apply sigmoid/softplus while training code may also transform logits, creating possible double-transform or loss/output mismatches.
- Multi-task labels such as transition cost and compliance probability are partly derived from the same engineered features used as inputs.
- Broad `except Exception` blocks frequently substitute dummy data, which can make an apparently successful run use synthetic inputs.
- `top 4 update ver.ipynb` contains a stored syntax error and cannot be considered runnable.
- `Attention weight + visualization ver8.ipynb` is effectively empty.
- `Attention_weight_visualization_FULL.py` is a report-style export rather than valid maintained Python.
- `db_to_csv.ipynb` shows a missing-table failure.
- `graphsage_ppi.py` trains and downloads at import time and is unrelated to the ESG data contract.
- Notebooks contain large execution outputs, environment-specific warnings and mutable execution-order dependencies.

## Security review

1. Literal NewsAPI credentials are present in multiple tracked notebooks, including the API, heterophily, time-series and final variants. Do not copy the value into documentation, issues or new commits.
2. Revoke and rotate the exposed NewsAPI credential immediately. Removing it only from the latest commit does not remove it from git history.
3. Replace notebook literals with environment-variable lookups and optional dependency injection.
4. Consider a history rewrite with `git filter-repo` or BFG after rotation, coordinated carefully because it rewrites commit hashes.
5. Add automated secret scanning (for example, GitHub secret scanning where available and a CI tool such as gitleaks).
6. Absolute local paths disclose usernames and workstation layout. Replace them with configuration and repository-relative examples.
7. Do not commit raw ESG datasets, API response caches, SQLite databases, trained weights or generated metrics containing sensitive company-level data.

## Phased migration plan

### PR 1 — Security and repository inventory

- Remove credentials from every notebook/script without deleting research content.
- Add environment-variable examples, secret scanning and a data/artifact policy.
- Add a machine-readable inventory of required external datasets.

### PR 2 — Data contracts and BEA/ESG loaders

- Introduce typed dataset schemas and configuration.
- Port robust BEA parsing from `differentiation top 10.ipynb`.
- Add fixtures based on `header file check.ipynb`.
- Add ESG loader validation and mapping-coverage reports.

### PR 3 — Graph backends — implemented

- Implemented separate `use_make` and `icio` graph builders with framework-neutral contracts.
- Preserved sector/industry labels and raw economic flows separately from model weights.
- Added focused tests for alignment, edge density, Leontief fallback, thresholding, self-loop policy and invalid inputs.

### PR 4 — Feature blocks and temporal splitting

- Port structural, macro, environmental, ESG and temporal feature blocks.
- Implement train-only scaler fitting and chronological masks.
- Add provenance metadata and missing-data policies.

### PR 5 — Targets and leakage audit

- Port target generation behind explicit functions.
- Mark observed versus derived targets.
- Add leakage tests and remove input columns that directly construct targets.

### PR 6 — Baseline and directed model registry

- Add MLP/GCN/GAT/weighted-GAT baselines.
- Add the bidirectional upstream/downstream model.
- Standardize model outputs and task heads.

### PR 7 — Heterophily and anomaly models

- Add GPR-GNN from `Heterophily ver6.ipynb`.
- Add EconomicGAE as a separate anomaly workflow.
- Add ablations for propagation depth, alpha and graph direction.

### PR 8 — Training, evaluation and experiment configs

- Add seeded training, early stopping, checkpoints and chronological evaluation.
- Add non-graph baselines and task-appropriate metrics.
- Add small deterministic integration fixtures and CI smoke runs.

### PR 9 — Interpretability and visualization

- Extract valid attention/shock functions from `Attention_weight_visualization_FULL.py`.
- Add saved figures and tabular explanation outputs.
- Add perturbation/ablation checks and warnings against causal interpretation.

### PR 10 — Notebook consolidation

- Preserve original notebooks under an archive/reference location.
- Replace active notebooks with thin, reproducible examples importing `src/` modules.
- Document which historical notebook each module was derived from and any intentional behavioral change.

## Acceptance criteria for the completed migration

- Both graph backends run from configuration without source edits.
- No tracked credentials or absolute user paths remain.
- All data mappings report coverage and are versioned outside Python code.
- Training uses chronological splits with train-only preprocessing.
- At least one non-graph baseline and four graph variants are compared under the same data split.
- Observed and derived targets are clearly distinguished, with leakage tests.
- Attention/shock visualizations are reproducible from saved predictions/checkpoints.
- CI runs unit tests and a deterministic end-to-end smoke experiment.
- Original research notebooks remain available for traceability but are no longer the production execution path.


- Migration PR 2 implemented typed data contracts, BEA/ESG loaders, and synthetic mapping coverage utilities without adding real datasets.

### Migration PR 4 — implemented

Implemented typed, framework-neutral feature blocks, deterministic feature assembly, chronological train/validation/test splitting, and train-only preprocessing. Design decisions are intentionally narrow: feature rows are keyed by `(node_id, period)` with integer annual periods; feature names are namespaced; provenance/report objects are JSON-serializable; temporal features use only prior periods by default; preprocessing state stores deterministic statistics instead of serialized estimator objects. This PR does not implement targets or claim that target leakage is fully resolved; target construction remains Migration PR 5 work.
