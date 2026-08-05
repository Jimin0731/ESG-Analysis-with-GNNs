from __future__ import annotations

import numpy as np
import pandas as pd

from src.graphs.contracts import GraphSnapshot
from .contracts import FeatureBlock, FeatureProvenance, FeatureValidationError, sort_feature_frame, validate_annual_period

FEATURES = (
    "structural__retained_in_degree",
    "structural__retained_out_degree",
    "structural__raw_in_strength",
    "structural__raw_out_strength",
    "structural__model_weight_in_strength",
    "structural__model_weight_out_strength",
    "structural__raw_in_share",
    "structural__raw_out_share",
)


def build_structural_feature_block(snapshot: GraphSnapshot, *, period: int | None = None, canonical_node_order=None, include_technical_coefficients: bool = True) -> FeatureBlock:
    raw_period = period if period is not None else snapshot.period
    p = validate_annual_period(raw_period)
    n = len(snapshot.nodes)
    sources = snapshot.edge_index[0]
    targets = snapshot.edge_index[1]
    raw = np.asarray(snapshot.raw_flow, dtype=float)
    weights = np.asarray(snapshot.edge_weight, dtype=float)
    data = {"node_id": snapshot.node_ids, "period": [p] * n}
    stats = {
        FEATURES[0]: np.bincount(targets, minlength=n),
        FEATURES[1]: np.bincount(sources, minlength=n),
        FEATURES[2]: np.bincount(targets, weights=raw, minlength=n),
        FEATURES[3]: np.bincount(sources, weights=raw, minlength=n),
        FEATURES[4]: np.bincount(targets, weights=weights, minlength=n),
        FEATURES[5]: np.bincount(sources, weights=weights, minlength=n),
    }
    for name, values in stats.items():
        data[name] = values.astype(float)
    total_in = float(np.sum(data[FEATURES[2]]))
    total_out = float(np.sum(data[FEATURES[3]]))
    data[FEATURES[6]] = data[FEATURES[2]] / total_in if total_in else np.full(n, np.nan)
    data[FEATURES[7]] = data[FEATURES[3]] / total_out if total_out else np.full(n, np.nan)
    names = list(FEATURES)
    matrix = snapshot.source_metadata.get("unthresholded_A")
    if include_technical_coefficients and matrix is not None:
        try:
            A = np.asarray(matrix, dtype=float)
        except (TypeError, ValueError) as exc:
            raise FeatureValidationError("unthresholded_A must be numeric") from exc
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise FeatureValidationError("unthresholded_A must be a square two-dimensional matrix")
        if A.shape != (n, n):
            raise FeatureValidationError("unthresholded_A shape must match node count")
        if not np.isfinite(A).all():
            raise FeatureValidationError("unthresholded_A must contain only finite values")
        technical = {
            "structural__technical_coefficient_row_sum": A.sum(axis=1),
            "structural__technical_coefficient_column_sum": A.sum(axis=0),
            "structural__technical_coefficient_diagonal": np.diag(A),
        }
        for name, values in technical.items():
            data[name] = values
            names.append(name)
    frame = sort_feature_frame(pd.DataFrame(data), canonical_node_order or snapshot.node_ids)
    provenance = tuple(
        FeatureProvenance(name, "structural", snapshot.backend, name.removeprefix("structural__"), "graph_statistic", "snapshot annual", False, (), "preserve", None, "raw flows are distinct from model edge weights")
        for name in names
    )
    return FeatureBlock("structural", frame, tuple(names), provenance)
