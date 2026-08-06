"""Launch the active notebooks with repository imports configured."""
from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path


def build_notebook_environment(
    repository_root: Path,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a copied environment with the repository first on ``PYTHONPATH``."""
    environment = dict(os.environ if base_environment is None else base_environment)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(repository_root) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    return environment


def build_notebook_command() -> tuple[str, ...]:
    """Return the cross-platform command for this interpreter's Jupyter server."""
    return (sys.executable, "-m", "notebook")


def main() -> int:
    """Launch Jupyter from the repository root and propagate its exit code."""
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        build_notebook_command(),
        cwd=repository_root,
        env=build_notebook_environment(repository_root),
        shell=False,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
