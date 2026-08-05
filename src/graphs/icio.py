"""OECD/ICIO transaction graph backend."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import (
    GraphInputError,
    SUPPORTED_MISSING_POLICIES,
    aggregate_duplicate_edges,
    density,
    model_weights,
    ordered_labels,
    reject_duplicate_labels,
    threshold_mask,
    to_edge_index,
)
from .contracts import GraphConstructionConfig, GraphConstructionReport, GraphNode, GraphSnapshot


@dataclass(frozen=True)
class ICIOGraphConfig(GraphConstructionConfig):
    backend: str = "icio"
    source_column: str | None = None
    missing_tokens: tuple[str, ...] = ("", "..", "--", "NA", "N/A", "SUPPRESSED")
    missing_cell_policy: str = "skip"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.missing_cell_policy not in SUPPORTED_MISSING_POLICIES:
            raise GraphInputError("unsupported ICIO missing-cell policy")


def _clean(x: object) -> str:
    return str(x).strip()


def _numeric(x: object, missing: tuple[str, ...]) -> float | None:
    if pd.isna(x) or _clean(x) in missing:
        return None
    try:
        return float(str(x).replace(",", ""))
    except ValueError as exc:
        raise GraphInputError(f"invalid numeric transaction cell: {x}") from exc


def _ordered_union(left: list[str], right: list[str]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for label in left + right:
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def _wide(df: pd.DataFrame, cfg: ICIOGraphConfig) -> tuple[pd.DataFrame, list[str], list[str]]:
    cols = ordered_labels(df.columns)
    reject_duplicate_labels(cols, "ICIO columns")
    wide = df.copy(deep=True)
    wide.columns = cols
    if cfg.source_column is None:
        candidates = [col for col in cols if col.lower() in {"source", "source_industry", "source industry", "industry"}]
        if len(candidates) != 1:
            raise GraphInputError("ambiguous source column")
        source_column = candidates[0]
    elif cfg.source_column not in cols:
        raise GraphInputError("configured source column not found")
    else:
        source_column = cfg.source_column
    source_ids = ordered_labels(wide[source_column])
    destination_ids = [col for col in cols if col != source_column]
    reject_duplicate_labels(source_ids, "source industries")
    reject_duplicate_labels(destination_ids, "destination industries")
    labels = _ordered_union(source_ids, destination_ids)
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=object)
    for _, row in wide.iterrows():
        source = _clean(row[source_column])
        for destination in destination_ids:
            matrix.at[source, destination] = row[destination]
    return matrix, source_ids, destination_ids


def build_icio_graph(transactions: pd.DataFrame, *, config: ICIOGraphConfig | None = None, period=None) -> GraphSnapshot:
    cfg = config or ICIOGraphConfig()
    df = transactions.copy(deep=True)
    clean_index = ordered_labels(df.index)
    clean_cols = ordered_labels(df.columns)
    if cfg.source_column is not None or df.shape[0] != df.shape[1] or set(clean_index) != set(clean_cols):
        matrix, source_ids, destination_ids = _wide(df, cfg)
    else:
        matrix = df.copy(deep=True)
        matrix.index = clean_index
        matrix.columns = clean_cols
        reject_duplicate_labels(matrix.index, "source industries")
        reject_duplicate_labels(matrix.columns, "destination industries")
        if set(matrix.index) != set(matrix.columns):
            raise GraphInputError("ambiguous square ICIO schema")
        matrix = matrix.loc[list(matrix.index), list(matrix.index)]
        source_ids = list(matrix.index)
        destination_ids = list(matrix.columns)
    labels = list(matrix.index)
    if not labels:
        raise GraphInputError("no usable industries")

    input_cell_count = int(matrix.size)
    edges: list[tuple[int, int, float, float]] = []
    missing_count = 0
    zero_flow_count = 0
    positive_before_self_loops = 0
    self_loops_dropped = 0
    for source_pos, source in enumerate(labels):
        for target_pos, destination in enumerate(labels):
            value = _numeric(matrix.at[source, destination], cfg.missing_tokens)
            if value is None:
                missing_count += 1
                continue
            if not np.isfinite(value):
                raise GraphInputError("non-finite transaction value")
            if value < 0:
                raise GraphInputError("negative transaction value")
            if value == 0:
                zero_flow_count += 1
                continue
            positive_before_self_loops += 1
            if source_pos == target_pos and not cfg.include_self_loops:
                self_loops_dropped += 1
                continue
            edges.append((source_pos, target_pos, value, value))
    if cfg.missing_cell_policy == "error" and missing_count:
        raise GraphInputError(f"ICIO table contains {missing_count} missing transaction cells")
    if not edges:
        raise GraphInputError("no usable positive flows")
    threshold = threshold_mask([edge[3] for edge in edges], cfg.threshold_policy, cfg.threshold_value)
    kept = [edge for edge, keep in zip(edges, threshold) if keep]
    flows_dropped_by_threshold = len(edges) - len(kept)
    if not kept:
        raise GraphInputError("no flows retained after threshold")
    kept = aggregate_duplicate_edges(kept, cfg.duplicate_edge_policy)
    edge_index = to_edge_index(kept)
    flows = np.asarray([edge[3] for edge in kept], dtype=float)
    weights = model_weights(flows, edge_index, len(labels), cfg.weight_transform)
    source_set = set(source_ids)
    destination_set = set(destination_ids)
    diagnostics = {
        "input_cell_count": input_cell_count,
        "missing_cell_count": missing_count,
        "missing_count": missing_count,
        "zero_flow_count": zero_flow_count,
        "zero_count": zero_flow_count,
        "positive_flow_count_before_self_loop_filtering": positive_before_self_loops,
        "self_loops_dropped": self_loops_dropped,
        "dropped_self_loops": self_loops_dropped,
        "flows_dropped_by_threshold": flows_dropped_by_threshold,
        "retained_flow_count": len(kept),
        "source_only_industry_ids": [label for label in labels if label in source_set and label not in destination_set],
        "destination_only_industry_ids": [label for label in labels if label in destination_set and label not in source_set],
        "final_node_count": len(labels),
    }
    report = GraphConstructionReport(
        "icio",
        len(labels),
        len(kept),
        density(len(labels), len(kept), cfg.include_self_loops),
        cfg.threshold_policy,
        cfg.threshold_value,
        diagnostics,
    )
    return GraphSnapshot(tuple(GraphNode(label, label) for label in labels), edge_index, weights, flows, "icio", period, {}, report)
