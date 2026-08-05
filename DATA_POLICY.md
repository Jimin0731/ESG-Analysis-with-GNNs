# Data and artifact policy

Do not commit private, licensed, or generated research artifacts to this repository.

## Never commit

- Raw or licensed datasets, including downloaded ESG, BEA, OECD/ICIO, EPA/GHGP, environmental-tax, or NewsAPI response data.
- Company-level sensitive or proprietary data.
- `.env` files or any local credential/configuration file containing secrets.
- API response caches and downloaded news payloads.
- SQLite databases, including local news-monitoring databases.
- Trained model weights, checkpoints, and serialized experiments.
- Generated metrics, reports, plots, figures, and run outputs, except deliberately selected documentation assets reviewed for publication.

## Intended local directories

- `data/external/` — raw datasets supplied by the user or downloaded under the provider's terms.
- `data/cache/` — API responses and temporary intermediate caches.
- `data/local/` — local databases or sensitive working files.
- `models/` — trained model weights and checkpoints.
- `runs/` — experiment logs, generated metrics, figures, and reports.

Keep credentials in an untracked `.env` file or a local secret manager. Users must rotate any exposed credential; deleting it from the current tree does not remove it from git history. Any history rewrite with `git filter-repo` or BFG must be coordinated separately because it changes commit hashes.
