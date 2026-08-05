"""Shared economic graph utilities."""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd


class GraphInputError(ValueError):
    """Raised when source economic graph inputs violate public contracts."""


SUPPORTED_BACKENDS = {"use_make", "icio"}
SUPPORTED_THRESHOLD_POLICIES = {"absolute", "percentile"}
SUPPORTED_DUPLICATE_EDGE_POLICIES = {"reject", "sum", "mean"}
SUPPORTED_WEIGHT_TRANSFORMS = {"raw", "log1p", "row_normalized", "standardized"}
SUPPORTED_ORIENTATIONS = {"commodity_by_sector", "sector_by_commodity"}
SUPPORTED_MISSING_POLICIES = {"skip", "error"}
SUPPORTED_LEONTIEF_FALLBACKS = {"error", "pinv", "regularized"}


def ordered_labels(labels: Iterable[object]) -> list[str]:
    return [str(x).strip() for x in labels]


def reject_empty_labels(labels: Iterable[str], axis: str = "labels") -> None:
    bad = [i for i, label in enumerate(labels) if not str(label).strip()]
    if bad:
        raise GraphInputError(f"empty or whitespace-only {axis}: positions {bad}")


def duplicate_labels(labels: Iterable[object]) -> list[str]:
    cleaned = ordered_labels(labels)
    return sorted([label for label, count in Counter(cleaned).items() if count > 1])


def reject_duplicate_labels(labels: Iterable[object], axis: str = "labels") -> None:
    cleaned = ordered_labels(labels)
    reject_empty_labels(cleaned, axis)
    duplicates = duplicate_labels(cleaned)
    if duplicates:
        raise GraphInputError(f"duplicate {axis}: {duplicates}")


def validate_config_common(
    *,
    backend: str,
    threshold_policy: str,
    threshold_value: float,
    duplicate_edge_policy: str,
    weight_transform: str,
) -> None:
    if backend not in SUPPORTED_BACKENDS:
        raise GraphInputError(f"unsupported backend: {backend}")
    if threshold_policy not in SUPPORTED_THRESHOLD_POLICIES:
        raise GraphInputError("threshold policy must be 'absolute' or 'percentile'")
    value = float(threshold_value)
    if not np.isfinite(value):
        raise GraphInputError("threshold value must be finite")
    if threshold_policy == "absolute" and value < 0:
        raise GraphInputError("absolute threshold must be non-negative")
    if threshold_policy == "percentile" and not 0 <= value <= 100:
        raise GraphInputError("percentile threshold must be within 0..100")
    if duplicate_edge_policy not in SUPPORTED_DUPLICATE_EDGE_POLICIES:
        raise GraphInputError("unsupported duplicate-edge policy")
    if weight_transform not in SUPPORTED_WEIGHT_TRANSFORMS:
        raise GraphInputError("unsupported weight transform")


def validate_square_labelled_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.shape[0] != frame.shape[1]:
        raise GraphInputError("matrix must be a non-empty square DataFrame")
    frame = frame.copy()
    frame.index = ordered_labels(frame.index)
    frame.columns = ordered_labels(frame.columns)
    reject_duplicate_labels(frame.index, "row labels")
    reject_duplicate_labels(frame.columns, "column labels")
    if list(frame.index) != list(frame.columns):
        raise GraphInputError("square matrix row and column labels must match in order")
    return frame.apply(pd.to_numeric, errors="raise")


def sort_edges(edges: list[tuple[int, int, float, float]]) -> list[tuple[int, int, float, float]]:
    return sorted(edges, key=lambda e: (int(e[0]), int(e[1]), float(e[2])))


def apply_self_loop_policy(edges: list[tuple[int, int, float, float]], include: bool) -> list[tuple[int, int, float, float]]:
    if include:
        return edges
    return [edge for edge in edges if edge[0] != edge[1]]


def aggregate_duplicate_edges(edges: list[tuple[int, int, float, float]], policy: str = "reject") -> list[tuple[int, int, float, float]]:
    if policy not in SUPPORTED_DUPLICATE_EDGE_POLICIES:
        raise GraphInputError("unsupported duplicate-edge policy")
    buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for source, target, weight, raw in edges:
        key = (int(source), int(target))
        buckets.setdefault(key, []).append((float(weight), float(raw)))
    if policy == "reject" and any(len(values) > 1 for values in buckets.values()):
        raise GraphInputError("duplicate edges found")
    out = []
    for (source, target), values in buckets.items():
        arr = np.asarray(values, dtype=float)
        value = arr.sum(axis=0) if policy == "sum" else arr.mean(axis=0)
        out.append((source, target, float(value[0]), float(value[1])))
    return sort_edges(out)


def threshold_mask(values: Iterable[float], policy: str = "absolute", value: float = 0.0) -> np.ndarray:
    vals = np.asarray(list(values), dtype=float)
    if not np.isfinite(vals).all():
        raise GraphInputError("threshold candidates must be finite")
    if (vals < 0).any():
        raise GraphInputError("threshold candidates must be non-negative")
    threshold_value = float(value)
    if policy == "absolute":
        if not np.isfinite(threshold_value) or threshold_value < 0:
            raise GraphInputError("absolute threshold must be finite and non-negative")
        return vals > threshold_value
    if policy == "percentile":
        if not np.isfinite(threshold_value) or not 0 <= threshold_value <= 100:
            raise GraphInputError("percentile threshold must be 0..100")
        threshold = float(np.percentile(vals, threshold_value)) if vals.size else np.inf
        # Documented rule: flows equal to the threshold are excluded in all backends.
        return vals > threshold
    raise GraphInputError("unsupported threshold policy")


def density(node_count: int, edge_count: int, include_self_loops: bool = False) -> float:
    denom = node_count * node_count if include_self_loops else node_count * (node_count - 1)
    return 0.0 if denom <= 0 else float(edge_count) / float(denom)


def degree_summary(edge_index: np.ndarray, node_count: int) -> dict[str, list[int]]:
    ei = np.asarray(edge_index, dtype=int)
    out = np.bincount(ei[0], minlength=node_count) if ei.size else np.zeros(node_count, dtype=int)
    inn = np.bincount(ei[1], minlength=node_count) if ei.size else np.zeros(node_count, dtype=int)
    return {"in_degree": inn.tolist(), "out_degree": out.tolist()}


def to_edge_index(edges: list[tuple[int, int, float, float]]) -> np.ndarray:
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray([[edge[0] for edge in edges], [edge[1] for edge in edges]], dtype=np.int64)


def model_weights(raw: Iterable[float], edge_index: np.ndarray | None = None, node_count: int | None = None, transform: str = "raw") -> np.ndarray:
    raw_arr = np.asarray(list(raw), dtype=float)
    if transform == "raw":
        out = raw_arr.copy()
    elif transform == "log1p":
        out = np.log1p(raw_arr)
    elif transform == "row_normalized":
        if edge_index is None or node_count is None:
            raise GraphInputError("row_normalized requires edge_index and node_count")
        out = np.zeros_like(raw_arr)
        sums = np.zeros(node_count, dtype=float)
        if raw_arr.size:
            np.add.at(sums, np.asarray(edge_index)[0], raw_arr)
        for i, (source, _) in enumerate(np.asarray(edge_index).T):
            out[i] = 0.0 if sums[source] == 0 else raw_arr[i] / sums[source]
    elif transform == "standardized":
        if raw_arr.size <= 1:
            out = np.zeros_like(raw_arr)
        else:
            std = float(raw_arr.std())
            out = np.zeros_like(raw_arr) if std == 0 else (raw_arr - raw_arr.mean()) / std
    else:
        raise GraphInputError("unsupported weight transform")
    if not np.isfinite(out).all():
        raise GraphInputError("model weights must be finite")
    return out.astype(float)
