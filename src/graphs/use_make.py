"""USE/MAKE economic graph backend."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import (
    GraphInputError,
    SUPPORTED_LEONTIEF_FALLBACKS,
    SUPPORTED_ORIENTATIONS,
    aggregate_duplicate_edges,
    apply_self_loop_policy,
    density,
    model_weights,
    ordered_labels,
    reject_duplicate_labels,
    threshold_mask,
    to_edge_index,
)
from .contracts import GraphConstructionConfig, GraphConstructionReport, GraphNode, GraphSnapshot, LeontiefComputationReport


@dataclass(frozen=True)
class UseMakeGraphConfig(GraphConstructionConfig):
    backend: str = "use_make"
    use_orientation: str = "commodity_by_sector"
    make_orientation: str = "sector_by_commodity"
    min_alignment_coverage: float = 0.5
    negative_policy: str = "reject"
    leontief_fallback: str = "error"
    leontief_regularization: float = 1e-8

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.use_orientation not in SUPPORTED_ORIENTATIONS:
            raise GraphInputError("unsupported USE orientation")
        if self.make_orientation not in SUPPORTED_ORIENTATIONS:
            raise GraphInputError("unsupported MAKE orientation")
        if not 0 <= float(self.min_alignment_coverage) <= 1:
            raise GraphInputError("minimum alignment coverage must be within 0..1")
        if self.negative_policy != "reject":
            raise GraphInputError("only reject negative policy is supported")
        if self.leontief_fallback not in SUPPORTED_LEONTIEF_FALLBACKS:
            raise GraphInputError("invalid Leontief fallback policy")
        if not np.isfinite(float(self.leontief_regularization)) or float(self.leontief_regularization) <= 0:
            raise GraphInputError("Leontief regularization value must be finite and positive")


def _orient(df: pd.DataFrame, orientation: str, name: str) -> pd.DataFrame:
    if orientation not in SUPPORTED_ORIENTATIONS:
        raise GraphInputError(f"unsupported {name} orientation")
    oriented = df.copy(deep=True)
    oriented.index = ordered_labels(oriented.index)
    oriented.columns = ordered_labels(oriented.columns)
    reject_duplicate_labels(oriented.index, f"{name} rows")
    reject_duplicate_labels(oriented.columns, f"{name} columns")
    values = oriented.apply(pd.to_numeric, errors="raise")
    if name == "USE":
        canonical = values if orientation == "commodity_by_sector" else values.T
    elif name == "MAKE":
        canonical = values if orientation == "sector_by_commodity" else values.T
    else:
        raise GraphInputError("unknown USE/MAKE table name")
    return canonical.copy()


def compute_leontief_inverse(A: np.ndarray, fallback_policy: str = "error", regularization: float = 1e-8) -> tuple[np.ndarray, LeontiefComputationReport]:
    if fallback_policy not in SUPPORTED_LEONTIEF_FALLBACKS:
        raise GraphInputError("invalid Leontief fallback policy")
    if not np.isfinite(float(regularization)) or float(regularization) <= 0:
        raise GraphInputError("Leontief regularization value must be finite and positive")
    coefficients = np.asarray(A, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[0] != coefficients.shape[1] or not np.isfinite(coefficients).all():
        raise GraphInputError("A must be square and finite")
    system_matrix = np.eye(coefficients.shape[0]) - coefficients
    condition_number = float(np.linalg.cond(system_matrix))
    method = "inverse"
    used_fallback = False
    try:
        if not np.isfinite(condition_number) or condition_number > 1e12:
            raise np.linalg.LinAlgError("Leontief system is singular or ill-conditioned")
        inverse = np.linalg.inv(system_matrix)
    except np.linalg.LinAlgError as exc:
        if fallback_policy == "error":
            raise GraphInputError("Leontief system is singular or ill-conditioned") from exc
        used_fallback = True
        if fallback_policy == "pinv":
            method = "pinv"
            inverse = np.linalg.pinv(system_matrix)
        else:
            method = "regularized"
            inverse = np.linalg.inv(system_matrix + np.eye(system_matrix.shape[0]) * float(regularization))
    report = LeontiefComputationReport(
        shape=coefficients.shape,
        condition_number=condition_number,
        fallback_policy=fallback_policy,
        method=method,
        used_fallback=used_fallback,
        regularization_value=float(regularization) if method == "regularized" else None,
    )
    return inverse, report


def build_use_make_graph(use_table: pd.DataFrame, make_table: pd.DataFrame, *, config: UseMakeGraphConfig | None = None, period=None) -> GraphSnapshot:
    cfg = config or UseMakeGraphConfig()
    canonical_use = _orient(use_table, cfg.use_orientation, "USE")
    canonical_make = _orient(make_table, cfg.make_orientation, "MAKE")
    if not np.isfinite(canonical_use.to_numpy(float)).all() or not np.isfinite(canonical_make.to_numpy(float)).all():
        raise GraphInputError("economic values must be finite")
    if (canonical_use.to_numpy(float) < -1e-12).any() or (canonical_make.to_numpy(float) < -1e-12).any():
        raise GraphInputError("negative economic values are not allowed")

    common_commodities = [label for label in canonical_use.index if label in set(canonical_make.columns)]
    common_sectors = [label for label in canonical_use.columns if label in set(canonical_make.index)]
    coverage = min(
        len(common_commodities) / max(len(canonical_use.index), 1),
        len(common_sectors) / max(len(canonical_use.columns), 1),
        len(common_sectors) / max(len(canonical_make.index), 1),
        len(common_commodities) / max(len(canonical_make.columns), 1),
    )
    if coverage < cfg.min_alignment_coverage:
        raise GraphInputError("alignment coverage below configured minimum")
    use = canonical_use.loc[common_commodities, common_sectors]
    make = canonical_make.loc[common_sectors, common_commodities]
    if use.empty or make.empty:
        raise GraphInputError("no aligned USE/MAKE labels")

    # Notebook-derived industry technology formula. USE is normalized by total commodity
    # output to commodity-by-sector use_requirements; MAKE is normalized by sector output
    # to sector-by-commodity market_shares; A[source, target] is source sector input
    # required per unit of target sector output.
    commodity_output = use.sum(axis=1).to_numpy(float) + make.sum(axis=0).to_numpy(float)
    sector_output = make.sum(axis=1).to_numpy(float)
    if (commodity_output <= 0).any() or (sector_output <= 0).any():
        raise GraphInputError("commodity and sector outputs must be positive")
    use_requirements_by_commodity_sector = use.to_numpy(float) / commodity_output[:, None]
    market_shares_by_sector_commodity = make.to_numpy(float) / sector_output[:, None]
    technical_coefficients = use_requirements_by_commodity_sector.T @ market_shares_by_sector_commodity.T
    if not np.isfinite(technical_coefficients).all():
        raise GraphInputError("technical coefficients must be finite")
    _, leontief_report = compute_leontief_inverse(
        technical_coefficients,
        fallback_policy=cfg.leontief_fallback,
        regularization=cfg.leontief_regularization,
    )

    raw_edges = [
        (source, target, float(technical_coefficients[source, target]), float(technical_coefficients[source, target]))
        for source in range(technical_coefficients.shape[0])
        for target in range(technical_coefficients.shape[1])
    ]
    candidate_count = len(raw_edges)
    no_self_loop_edges = apply_self_loop_policy(raw_edges, cfg.include_self_loops)
    threshold = threshold_mask([edge[3] for edge in no_self_loop_edges], cfg.threshold_policy, cfg.threshold_value)
    kept = [edge for edge, keep in zip(no_self_loop_edges, threshold) if keep]
    dropped_by_threshold = len(no_self_loop_edges) - len(kept)
    kept = aggregate_duplicate_edges(kept, cfg.duplicate_edge_policy)
    edge_index = to_edge_index(kept)
    flows = np.asarray([edge[3] for edge in kept], dtype=float)
    weights = model_weights(flows, edge_index, len(common_sectors), cfg.weight_transform)
    diagnostics = {
        "alignment_coverage": float(coverage),
        "removed_use_commodities": [label for label in ordered_labels(use_table.index) if label not in common_commodities],
        "removed_sectors": [label for label in ordered_labels(use_table.columns) if label not in common_sectors],
        "use_requirements_shape": tuple(use_requirements_by_commodity_sector.shape),
        "market_shares_shape": tuple(market_shares_by_sector_commodity.shape),
        "technical_coefficients_shape": tuple(technical_coefficients.shape),
        "B_shape": tuple(use_requirements_by_commodity_sector.shape),
        "D_shape": tuple(market_shares_by_sector_commodity.shape),
        "candidate_flow_count": candidate_count,
        "self_loops_dropped": candidate_count - len(no_self_loop_edges),
        "flows_dropped_by_threshold": dropped_by_threshold,
    }
    report = GraphConstructionReport(
        "use_make",
        len(common_sectors),
        len(kept),
        density(len(common_sectors), len(kept), cfg.include_self_loops),
        cfg.threshold_policy,
        cfg.threshold_value,
        diagnostics,
    )
    nodes = tuple(GraphNode(label, label) for label in common_sectors)
    return GraphSnapshot(
        nodes,
        edge_index,
        weights,
        flows,
        "use_make",
        period,
        {"unthresholded_A": technical_coefficients},
        report,
        leontief_report,
    )
