from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import FeatureBlock, FeatureProvenance, FeatureValidationError, sort_feature_frame, validate_period_series, validate_node_id


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        raise FeatureValidationError(f"missing environmental column {column}")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise FeatureValidationError(f"non-numeric environmental column {column}")
    values = df[column].astype(float)
    if np.isinf(values.to_numpy()).any():
        raise FeatureValidationError(f"infinite values in environmental column {column}")
    return values


def build_environmental_feature_block(df, *, levels=(), log1p=(), intensities=None, canonical_node_order=None, invalid_denominator_policy="error"):
    if invalid_denominator_policy not in {"error", "missing"}:
        raise FeatureValidationError("unsupported invalid_denominator_policy")
    intensities = intensities or {}
    df["node_id"].map(validate_node_id)
    validate_period_series(df["period"])
    out = df[["node_id", "period"]].copy()
    names: list[str] = []
    for column in levels:
        values = _numeric_column(df, column)
        name = f"environmental__{column}_level"
        out[name] = values
        names.append(name)
    for column in log1p:
        values = _numeric_column(df, column)
        if (values.dropna() < 0).any():
            raise FeatureValidationError("log1p requires non-negative environmental values")
        name = f"environmental__{column}_log1p"
        out[name] = np.log1p(values)
        names.append(name)
    for suffix, cfg in intensities.items():
        numerator = cfg.get("numerator")
        denominator = cfg.get("denominator")
        if not isinstance(numerator, str) or not isinstance(denominator, str) or not numerator or not denominator:
            raise FeatureValidationError("intensity requires explicit numerator and denominator")
        if numerator == denominator:
            raise FeatureValidationError("intensity numerator and denominator must be distinct")
        num = _numeric_column(df, numerator)
        den = _numeric_column(df, denominator)
        invalid = den.isna() | (den <= 0)
        if invalid.any() and invalid_denominator_policy == "error":
            raise FeatureValidationError("intensity denominator must be positive and non-missing")
        name = f"environmental__{suffix}"
        out[name] = num / den.mask(invalid)
        names.append(name)
    out = sort_feature_frame(out, canonical_node_order or sorted(out["node_id"].unique()))
    provenance = tuple(
        FeatureProvenance(name, "environmental", "normalized_environmental", name.removeprefix("environmental__"), "configured", "annual observation", False, (), "preserve", None, "intensity ratios use explicit numerator and denominator")
        for name in names
    )
    return FeatureBlock("environmental", out, tuple(names), provenance)
