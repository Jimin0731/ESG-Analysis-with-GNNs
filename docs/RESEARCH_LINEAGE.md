# Research lineage and intentional redesigns

The machine-readable source is [`research_lineage.yaml`](research_lineage.yaml). Migration preserves provenance; it does **not** prove research correctness or exact behavioral equivalence.

| Module group | Primary source | Secondary source | Retained | Intentionally changed | Scientifically unverified |
|---|---|---|---|---|---|
| Data/configuration | `Data Preprocess.ipynb` | final notebook | source concepts and ordered identifiers | typed contracts; environment variables instead of embedded keys; no silent real-data download | source coverage and licensing |
| Loaders/mapping | header/preprocess notebooks | early ranking notebook | BEA/ESG/industry intent | explicit validation and configured paths; broken Korean mapping remains reference only | mapping correctness requires experts |
| Graphs | IO iteration | final notebook | directed economic relationships | explicit USE/MAKE versus ICIO backends and directed edge-weight semantics | empirical graph validity |
| Features/targets | real GDP/environment notebook | differentiation notebook | feature and derived-target concepts | train-only preprocessing, explicit target provenance, direct-lineage leakage audit | construct validity and equations in real settings |
| Models/baselines | final notebook | epoch iteration | graph and non-graph comparison | pure-PyTorch canonical models, ordered registry, no import-time training | comparative scientific performance |
| Supervised experiments | time-series notebook | epoch iteration | chronological evaluation intent | validation-only early stopping/scheduling and in-memory state | real-data generalization |
| Anomaly | heterophily notebook | final notebook | anomaly exploration | reconstruction-based anomaly scoring | anomaly threshold meaning |
| Interpretation | attention notebook | final notebook | directed attention and sensitivity questions | corrected incoming-target attention semantics and non-causal attention/perturbation language | causal or economic interpretation |
| Utilities | database/header notebooks | API notebook | local export and inspection roles | no credentials, network activity, or automatic execution | source-data fitness |
| Archive | all 14 artifacts | Git history | exact bytes and checksum identity | historical notebook archival; no supported execution | all archived outputs remain unreproduced |

## Major intentional changes across Migration PRs 1–10

Credentials were removed from active code and configuration uses environment variables rather than embedded keys. Active code performs no silent real-data download and no import-time training. Typed data contracts, explicit USE/MAKE versus ICIO graph backends, train-only preprocessing, chronological splitting, explicit target provenance, and direct-lineage leakage auditing replaced implicit notebook state. Canonical models are pure PyTorch and preserve directed edge-weight semantics. Training uses validation-only early stopping. Anomaly scores are reconstruction based. Attention and perturbation are described as model association/local sensitivity—not causal estimates. PR 10 preserves historical notebooks in a non-imported archive.

The malformed historical attention export and unrelated `graphsage_ppi.py` are reference only. The early broken Korean-mapping notebook is also reference only; no equivalence is claimed. See the YAML for component-level PR attribution, status, changes, and caveats.
