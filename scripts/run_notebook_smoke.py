"""Execute exactly the five active notebooks in fresh, temporary kernels."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_NOTEBOOKS = (
    "00_end_to_end_overview.ipynb",
    "01_data_pipeline.ipynb",
    "02_model_comparison.ipynb",
    "03_temporal_evaluation.ipynb",
    "04_interpretability.ipynb",
)


def main() -> None:
    old_pythonpath = os.environ.get("PYTHONPATH")
    old_backend = os.environ.get("MPLBACKEND")
    os.environ["PYTHONPATH"] = str(ROOT) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    os.environ["MPLBACKEND"] = "Agg"
    try:
        with tempfile.TemporaryDirectory(prefix="esg-notebooks-") as directory:
            for name in ACTIVE_NOTEBOOKS:
                path = ROOT / "notebooks" / name
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                NotebookClient(
                    notebook, timeout=120, kernel_name="python3",
                    resources={"metadata": {"path": directory}},
                    allow_errors=False,
                ).execute()
                print(f"notebook_smoke ok {name}")
    finally:
        if old_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_pythonpath
        if old_backend is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = old_backend
    print(f"notebook_smoke {len(ACTIVE_NOTEBOOKS)}/{len(ACTIVE_NOTEBOOKS)} passed")


if __name__ == "__main__":
    main()
