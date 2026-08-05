from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import FeatureBlock, FeatureProvenance, FeatureValidationError, sort_feature_frame


def build_macro_feature_block(df: pd.DataFrame, *, metrics: dict[str, list[str]], canonical_node_order=None, zero_previous_policy: str = "missing") -> FeatureBlock:
    if zero_previous_policy not in {"missing", "error", "zero"}:
        raise FeatureValidationError("unsupported zero_previous_policy")
    base = df.copy()
    base = base.sort_values(["node_id", "period"])
    out = base[["node_id", "period"]].copy()
    names: list[str] = []
    for col, transforms in metrics.items():
        if col not in base:
            raise FeatureValidationError(f"missing macro column {col}")
        if not pd.api.types.is_numeric_dtype(base[col]):
            raise FeatureValidationError(f"non-numeric macro column {col}")
        if np.isinf(base[col].to_numpy(dtype=float)).any():
            raise FeatureValidationError(f"infinite values in macro column {col}")
        for transform in transforms:
            name = f"macro__{col}_{transform}"
            if transform == "level":
                values = base[col].astype(float)
            elif transform == "log1p":
                values = base[col].astype(float)
                if (values.dropna() < 0).any():
                    raise FeatureValidationError("log1p requires non-negative values")
                values = np.log1p(values)
            elif transform in {"change", "pct_change"}:
                prev = base.groupby("node_id", sort=False)[col].shift(1)
                cur = base[col].astype(float)
                if transform == "change":
                    values = cur - prev
                else:
                    zero_prev = prev == 0
                    if zero_prev.any() and zero_previous_policy == "error":
                        raise FeatureValidationError("previous value is zero for percentage change")
                    denominator = prev.mask(zero_prev)
                    values = (cur - prev) / denominator
                    if zero_previous_policy == "zero":
                        values = values.mask(zero_prev, 0.0)
            else:
                raise FeatureValidationError(f"unsupported macro transform {transform}")
            out[name] = values.to_numpy(dtype=float)
            names.append(name)
    out = sort_feature_frame(out, canonical_node_order or sorted(out["node_id"].unique()))
    provenance = tuple(
        FeatureProvenance(name, "macro", "normalized_macro", name.removeprefix("macro__"), name.rsplit("_", 1)[-1], "annual observation", False, (), "preserve")
        for name in names
    )
    return FeatureBlock("macro", out, tuple(names), provenance)
