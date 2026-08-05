from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.contracts import ChronologicalSplit, FeatureValidationError, validate_annual_period, validate_period_series

RATIO_TOLERANCE = 1e-9


def chronological_split(df: pd.DataFrame, *, train_end=None, validation_end=None, ratios=None, period_col: str = "period") -> ChronologicalSplit:
    if ratios is not None and (train_end is not None or validation_end is not None):
        raise FeatureValidationError("cannot supply ratio mode and boundary mode simultaneously")
    observed = sorted(set(validate_period_series(df[period_col], field_name=period_col).tolist()))
    if len(observed) < 3:
        raise FeatureValidationError("at least three unique periods are required")
    if ratios is not None:
        values = tuple(ratios)
        if len(values) != 3:
            raise FeatureValidationError("ratio mode requires exactly three ratios")
        if any(not np.isfinite(value) for value in values):
            raise FeatureValidationError("split ratios must be finite")
        if any(value < 0 for value in values):
            raise FeatureValidationError("split ratios must be non-negative")
        if not np.isclose(sum(values), 1.0, atol=RATIO_TOLERANCE):
            raise FeatureValidationError("split ratios must sum to 1.0")
        n_periods = len(observed)
        train_count = int(np.floor(n_periods * values[0]))
        validation_count = int(np.floor(n_periods * values[1]))
        test_count = n_periods - train_count - validation_count
        if min(train_count, validation_count, test_count) <= 0:
            raise FeatureValidationError("split ratios produce an empty split")
        train = observed[:train_count]
        validation = observed[train_count : train_count + validation_count]
        test = observed[train_count + validation_count :]
        mode = "ratios"
    else:
        if train_end is None or validation_end is None:
            raise FeatureValidationError("boundary mode requires train_end and validation_end")
        train_boundary = validate_annual_period(train_end, field_name="train_end")
        validation_boundary = validate_annual_period(validation_end, field_name="validation_end")
        if train_boundary >= validation_boundary:
            raise FeatureValidationError("train boundary must precede validation boundary")
        train = [p for p in observed if p <= train_boundary]
        validation = [p for p in observed if train_boundary < p <= validation_boundary]
        test = [p for p in observed if p > validation_boundary]
        if not train or not validation or not test:
            raise FeatureValidationError("boundaries must create three non-empty observed splits")
        mode = "boundaries"
    series = validate_period_series(df[period_col], field_name=period_col)
    train_mask = series.isin(train).to_numpy(dtype=bool)
    validation_mask = series.isin(validation).to_numpy(dtype=bool)
    test_mask = series.isin(test).to_numpy(dtype=bool)
    coverage = train_mask.astype(int) + validation_mask.astype(int) + test_mask.astype(int)
    if not np.all(coverage == 1):
        raise FeatureValidationError("period split omitted or duplicated rows")
    return ChronologicalSplit(
        tuple(train),
        tuple(validation),
        tuple(test),
        train_mask,
        validation_mask,
        test_mask,
        {"train": int(train_mask.sum()), "validation": int(validation_mask.sum()), "test": int(test_mask.sum())},
        {"periods": observed, "mode": mode, "ratio_tolerance": RATIO_TOLERANCE},
    )
