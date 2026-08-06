# Migration completion report

> **The software migration is complete, but real-data scientific reproduction and validation are not complete.**

## Ten-PR scope

| PR | Scope |
|---:|---|
| 1 | security inventory and active-code boundary |
| 2 | typed loaders, configuration, and mapping |
| 3 | explicit USE/MAKE and OECD/ICIO graph backends |
| 4 | ordered feature construction and train-only preprocessing |
| 5 | chronological splits, targets, provenance, and leakage audit |
| 6 | canonical model registry, directed models, and baselines |
| 7 | GPR-GNN and reconstruction-based EconomicGAE anomaly workflow |
| 8 | deterministic chronological supervised/anomaly experiments |
| 9 | corrected training reconstruction and non-causal interpretability |
| 10 | byte-preserved archive, thin notebooks, and lineage documentation |

Canonical executable entry points are `src/` public APIs and `scripts/run_pipeline.py`, `scripts/run_experiment_smoke.py`, and `scripts/run_interpretability_smoke.py`. Active synthetic notebooks are under `notebooks/`; historical artifacts are under `archive/research/` and must not be imported or executed. Unit, security-inventory, legacy, chronological, interpretability, checksum, lineage, static notebook, and fresh-kernel notebook smoke checks define current software coverage.

Security controls include no active embedded credential, configuration-driven external paths, current-tree and history secret scans, and a narrowly scoped non-reversible fingerprint allowance. Moving files has not purged Git history.

The reproducibility boundary is deterministic synthetic execution. External datasets are not included, and archived results have not been independently reproduced.

## Remaining non-code work

- **Revoke or rotate the previously exposed NewsAPI credential.**
- Obtain and license the real external datasets.
- Configure real dataset paths outside Git.
- Run and review a complete real-data experiment.
- Review industry mappings with domain expertise.
- Validate derived targets and scientific assumptions independently.
- Do not infer causality from attention or perturbation outputs.
