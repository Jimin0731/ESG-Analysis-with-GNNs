# Historical research archive

These files are historical research artifacts preserved byte-for-byte for provenance and comparison. They are **not canonical executable entry points**. Several contain broken, incomplete, duplicated, or environment-specific code; they may retain old local paths, notebook output, and unsafe import-time behavior. **Do not import or automatically execute them.** Active reusable code lives under `src/`; active reproducible synthetic examples live under `notebooks/`.

The previously exposed NewsAPI credential must still be manually revoked or rotated. No value is reproduced here. Moving files does not erase Git history, and archived results are not independently reproduced scientific results.

| Archived path | Research branch | Historical role | Current replacement | Status | Important caveat |
|---|---|---|---|---|---|
| `integrated_use_make/top 4 update ver.ipynb` | integrated USE/MAKE | early ranking exploration | `src/features/`, `src/targets/` | reference only | early mapping is broken/unverified |
| `integrated_use_make/Epoch 300 + IO data ver2.ipynb` | integrated USE/MAKE | graph/model iteration | `src/graphs/use_make.py`, `src/models/` | partially migrated | behavior intentionally redesigned |
| `integrated_use_make/Real GDP data + real env data ver3.ipynb` | integrated USE/MAKE | macro/environment features | `src/features/` | partially migrated | scientific constructs unverified |
| `integrated_use_make/API added ver5.ipynb` | integrated USE/MAKE | news/API experiment | `src/data/news_sentiment.py` | intentionally excluded | old credential requires rotation |
| `integrated_use_make/Heterophily ver6.ipynb` | integrated USE/MAKE | anomaly exploration | `src/anomaly/` | partially migrated | canonical scoring is reconstruction based |
| `integrated_use_make/Attention weight + visualization ver8.ipynb` | integrated USE/MAKE | attention exploration | `src/evaluation/`, `src/visualization/` | partially migrated | association, not causation |
| `integrated_use_make/Attention_weight_visualization_FULL.py` | integrated USE/MAKE | malformed attention export | none | reference only | malformed and never imported |
| `integrated_use_make/time series ver9.ipynb` | integrated USE/MAKE | temporal experiment | `src/experiments/`, `src/splits/` | partially migrated | validation-only selection now enforced |
| `integrated_use_make/ESG Analysis final ver.ipynb` | integrated USE/MAKE | consolidated prototype | `src/`, `scripts/` | partially migrated | no claim of exact equivalence |
| `us_icio_bea/Data Preprocess.ipynb` | US OECD/ICIO + BEA | preprocessing/ICIO prototype | `src/data/`, `src/graphs/icio.py` | partially migrated | mappings need expert review |
| `us_icio_bea/header file check.ipynb` | US OECD/ICIO + BEA | BEA header inspection | `scripts/inspect_bea_headers.py` | migrated | requires separately supplied data |
| `us_icio_bea/differentiation top 10.ipynb` | US OECD/ICIO + BEA | differentiation/ranking | `src/targets/` | reference only | target validity unverified |
| `auxiliary/db_to_csv.ipynb` | auxiliary | news database export | `scripts/export_news_db.py` | migrated | local utility; no network access |
| `auxiliary/graphsage_ppi.py` | auxiliary | unrelated GraphSAGE/PPI reference | none | reference only | unrelated material, not migrated |
