"""Reusable preprocessing and graph construction for ESG GNN experiments.

The implementation is extracted from the project notebooks, primarily
``ESG Analysis final ver.ipynb``.  It keeps the notebook workflow's core ideas:
time-ordered I/O matrices, Leontief-style economic features, percentile-based
weighted edges, and ESG target preparation, while avoiding hard-coded local
paths or credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ESGRecord:
    company: str
    sector: str
    environmental: float
    social: float
    governance: float

    @property
    def esg_score(self) -> float:
        return float(np.mean([self.environmental, self.social, self.governance]))


@dataclass(frozen=True)
class GraphDataset:
    """Framework-neutral graph tensors used by training and tests."""

    features: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    targets: np.ndarray
    node_labels: list[str]
    years_used: list[int]


class TimeSeriesDataSplitter:
    """Chronological train/validation/test split used by the source notebook."""

    def __init__(self, train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2):
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError("split ratios must sum to 1.0")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split_by_years(self, yearly_data: dict[int, pd.DataFrame | np.ndarray]) -> dict[str, dict[int, pd.DataFrame | np.ndarray]]:
        years = sorted(int(y) for y in yearly_data)
        if len(years) < 3:
            raise ValueError("at least three years are required for chronological splitting")
        train_end = max(1, int(len(years) * self.train_ratio))
        val_end = max(train_end + 1, int(len(years) * (self.train_ratio + self.val_ratio)))
        val_end = min(val_end, len(years) - 1)
        return {
            "train": {y: yearly_data[y] for y in years[:train_end]},
            "val": {y: yearly_data[y] for y in years[train_end:val_end]},
            "test": {y: yearly_data[y] for y in years[val_end:]},
        }


def load_sample_esg_data() -> list[ESGRecord]:
    """Small deterministic ESG sample for smoke tests without private data."""
    return [
        ESGRecord("Alpha Energy", "Energy", 62.0, 58.0, 64.0),
        ESGRecord("Beta Foods", "Consumer", 74.0, 79.0, 71.0),
        ESGRecord("Gamma Tech", "Technology", 81.0, 77.0, 83.0),
        ESGRecord("Delta Bank", "Financials", 69.0, 72.0, 75.0),
    ]


def add_composite_scores(records: list[ESGRecord]) -> list[dict[str, float | str]]:
    return [{**record.__dict__, "esg_score": record.esg_score} for record in records]


def validate_square_matrix(matrix: pd.DataFrame | np.ndarray) -> None:
    """Validate that an I/O matrix is non-empty and square."""
    shape = np.asarray(matrix).shape
    if len(shape) != 2 or shape[0] == 0 or shape[1] == 0:
        raise ValueError("I/O matrix must be a non-empty two-dimensional matrix")
    if shape[0] != shape[1]:
        raise ValueError("I/O matrix must be square with the same number of rows and columns")


def read_io_matrix(path: str | Path) -> pd.DataFrame:
    """Read an industry-by-industry I/O matrix from CSV or Excel."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, index_col=0)
    else:
        df = pd.read_csv(path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    validate_square_matrix(df)
    return df


def graph_from_io_matrix(io_matrix: pd.DataFrame | np.ndarray, percentile: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Create weighted edges from positive I/O flows above a percentile threshold."""
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    validate_square_matrix(io_matrix)
    matrix = np.asarray(io_matrix, dtype=float)
    positive = matrix[matrix > 0]
    if positive.size == 0:
        raise ValueError("I/O matrix must contain positive flows")
    threshold = np.percentile(positive, percentile)
    sources, targets = np.where(matrix > threshold)
    # Canonical weighted graph models require non-negative economic weights.
    # Log scaling compresses large flows; max scaling preserves ordering and
    # direction without introducing the negative values created by z-scores.
    weights = np.log1p(matrix[sources, targets])
    if weights.size == 0:
        raise ValueError("percentile threshold retained no directed edges")
    maximum = float(weights.max())
    weights = weights / maximum if maximum > 0.0 else np.ones_like(weights, dtype=float)
    return np.vstack([sources, targets]).astype(np.int64), weights.astype(np.float32)


def features_from_io_matrix(io_matrix: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Build notebook-inspired Leontief and network features for each industry."""
    validate_square_matrix(io_matrix)
    matrix = np.asarray(io_matrix, dtype=float)
    row_sum = matrix.sum(axis=1)
    col_sum = matrix.sum(axis=0)
    technical = matrix / (col_sum.reshape(1, -1) + 1e-9)
    identity = np.eye(matrix.shape[0])
    leontief = np.linalg.pinv(identity - technical)
    raw = np.column_stack([
        row_sum,
        col_sum,
        np.diag(leontief),
        leontief.sum(axis=1),
        leontief.sum(axis=0),
        (row_sum - col_sum),
    ])
    return StandardScaler().fit_transform(raw).astype(np.float32)


def targets_from_esg_frame(esg_df: pd.DataFrame, node_labels: Iterable[str]) -> np.ndarray:
    """Align ESG targets to graph nodes using sector/industry labels when present."""
    df = esg_df.copy()
    labels = list(node_labels)
    if df.empty:
        raise ValueError("ESG data must contain at least one row")
    score_cols = [c for c in df.columns if c.lower() in {"esg_score", "esg", "score"}]
    if not score_cols:
        pillars = [c for c in df.columns if c.lower() in {"environmental", "social", "governance"}]
        if len(pillars) >= 3:
            df["esg_score"] = df[pillars].mean(axis=1)
            score_col = "esg_score"
        else:
            raise ValueError("ESG data needs esg_score or environmental/social/governance columns")
    else:
        score_col = score_cols[0]
    key_col = next((c for c in df.columns if c.lower() in {"sector", "industry", "node", "code"}), None)
    if key_col is None:
        values = df[score_col].to_numpy(dtype=float)
        if values.shape != (len(labels),):
            raise ValueError("ESG target length must match the number of graph nodes when no alignment key is present")
    else:
        mapping = df.groupby(key_col)[score_col].mean().to_dict()
        values = np.array([mapping.get(label, np.nan) for label in labels], dtype=float)
        values = np.where(np.isnan(values), np.nanmean(df[score_col].to_numpy(dtype=float)), values)
    return values.astype(np.float32)


def build_graph_dataset(io_matrix: pd.DataFrame | np.ndarray, targets: np.ndarray | None = None, years_used: list[int] | None = None) -> GraphDataset:
    validate_square_matrix(io_matrix)
    labels = list(io_matrix.index.astype(str)) if isinstance(io_matrix, pd.DataFrame) else [str(i) for i in range(np.asarray(io_matrix).shape[0])]
    features = features_from_io_matrix(io_matrix)
    if targets is None:
        targets = features[:, 0] * 10 + 70
    targets = np.asarray(targets, dtype=np.float32)
    if targets.shape != (features.shape[0],):
        raise ValueError("targets must be a one-dimensional array matching the number of graph nodes")
    edge_index, edge_weight = graph_from_io_matrix(io_matrix)
    return GraphDataset(features, edge_index, edge_weight, targets, labels, years_used or [])


def make_smoke_dataset() -> GraphDataset:
    matrix = pd.DataFrame(
        [[0, 12, 4, 1], [3, 0, 7, 2], [2, 9, 0, 8], [5, 1, 6, 0]],
        index=[r.sector for r in load_sample_esg_data()],
        columns=[r.sector for r in load_sample_esg_data()],
    )
    targets = np.array([r.esg_score for r in load_sample_esg_data()], dtype=np.float32)
    return build_graph_dataset(matrix, targets=targets, years_used=[2021, 2022, 2023])
