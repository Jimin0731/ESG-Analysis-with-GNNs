from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import FeatureBlock, FeatureProvenance, FeatureValidationError, sort_feature_frame, validate_period_series


def build_temporal_feature_block(df, *, source_features, lags=(1,), rolling_windows=(), min_history=1, include_current=False, changes=True, slopes=False, canonical_node_order=None):
    if min_history <= 0:
        raise FeatureValidationError("min_history must be positive")
    lag_values = tuple(lags)
    window_values = tuple(rolling_windows)
    if any(not isinstance(lag, int) or isinstance(lag, bool) or lag <= 0 for lag in lag_values):
        raise FeatureValidationError("lags must be positive integers")
    if any(not isinstance(window, int) or isinstance(window, bool) or window <= 0 for window in window_values):
        raise FeatureValidationError("rolling windows must be positive integers")
    validate_period_series(df["period"])
    data = df.sort_values(["node_id", "period"]).copy()
    out = data[["node_id", "period"]].copy()
    names: list[str] = []
    for feature in source_features:
        if feature not in data:
            raise FeatureValidationError(f"missing temporal source feature {feature}")
        if not pd.api.types.is_numeric_dtype(data[feature]):
            raise FeatureValidationError(f"non-numeric temporal source feature {feature}")
        if np.isinf(data[feature].to_numpy(dtype=float)).any():
            raise FeatureValidationError(f"infinite temporal source feature {feature}")
        group = data.groupby("node_id", sort=False)[feature]
        history_series = data[feature] if include_current else group.shift(1)
        history = history_series.groupby(data["node_id"], sort=False)
        for lag in lag_values:
            name = f"temporal__{feature}_lag_{lag}"
            out[name] = group.shift(lag - 1 if include_current else lag).to_numpy(dtype=float)
            names.append(name)
        if changes:
            name = f"temporal__{feature}_change_1"
            current_change = group.diff(1)
            out[name] = current_change.to_numpy(dtype=float) if include_current else current_change.groupby(data["node_id"], sort=False).shift(1).to_numpy(dtype=float)
            names.append(name)
        for window in window_values:
            rolling = history.rolling(window, min_periods=min_history)
            for stat, values in {
                "rolling_mean": rolling.mean().reset_index(level=0, drop=True),
                "rolling_std": rolling.std(ddof=0).reset_index(level=0, drop=True),
            }.items():
                name = f"temporal__{feature}_{stat}_{window}"
                out[name] = values.to_numpy(dtype=float)
                names.append(name)
            if slopes:
                def slope(values):
                    if len(values) < min_history:
                        return np.nan
                    return float(np.polyfit(np.arange(len(values)), values, 1)[0])
                name = f"temporal__{feature}_slope_{window}"
                out[name] = rolling.apply(slope, raw=True).reset_index(level=0, drop=True).to_numpy(dtype=float)
                names.append(name)
    out = sort_feature_frame(out, canonical_node_order or sorted(out["node_id"].unique()))
    provenance = tuple(FeatureProvenance(name, "temporal", "assembled_features", name, "lag/rolling/change", "prior periods only unless include_current=True", False, (), "preserve") for name in names)
    return FeatureBlock("temporal", out, tuple(names), provenance)
