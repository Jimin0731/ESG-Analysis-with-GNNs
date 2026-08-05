from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import FeatureValidationError, FittedPreprocessingState, PreprocessingConfig, validate_feature_name, validate_period_series


class TrainOnlyPreprocessor:
    """Fit imputation and scaling statistics on training rows only."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.state_: FittedPreprocessingState | None = None

    def _validate_fit_inputs(self, frame: pd.DataFrame, feature_names, train_mask, periods):
        names = tuple(feature_names)
        if not names:
            raise FeatureValidationError("feature_names must be non-empty")
        names = tuple(validate_feature_name(name) for name in names)
        if len(names) != len(set(names)):
            raise FeatureValidationError("duplicate feature names")
        missing = [name for name in names if name not in frame.columns]
        if missing:
            raise FeatureValidationError(f"missing feature columns: {missing}")
        mask = np.asarray(train_mask)
        if mask.ndim != 1 or mask.dtype != bool:
            raise FeatureValidationError("train_mask must be a one-dimensional boolean mask")
        if mask.shape[0] != len(frame):
            raise FeatureValidationError("train_mask length must equal frame length")
        if not mask.any():
            raise FeatureValidationError("train_mask must select at least one row")
        period_series = validate_period_series(periods, field_name="periods")
        if len(period_series) != len(frame):
            raise FeatureValidationError("periods length must equal frame length")
        X = frame.loc[:, names].to_numpy(dtype=float).copy()
        if np.isinf(X).any():
            raise FeatureValidationError("input features must not contain infinity")
        return names, mask, period_series, X

    def fit(self, frame: pd.DataFrame, *, feature_names, train_mask, periods):
        names, mask, period_series, X = self._validate_fit_inputs(frame, feature_names, train_mask, periods)
        train = X[mask]
        imputation: dict[str, float | None] = {}
        filled = X.copy()
        original_missing = {name: int(np.isnan(X[:, i]).sum()) for i, name in enumerate(names)}
        for i, name in enumerate(names):
            train_col = train[:, i]
            observed_train = train_col[~np.isnan(train_col)]
            any_missing_all_rows = np.isnan(X[:, i]).any()
            if self.config.missing_policy == "error":
                if any_missing_all_rows:
                    raise FeatureValidationError("missing values present")
                imputation[name] = None
            elif self.config.missing_policy == "preserve":
                imputation[name] = None
            elif self.config.missing_policy == "train_mean":
                if observed_train.size == 0:
                    raise FeatureValidationError(f"all training values missing for {name}")
                imputation[name] = float(observed_train.mean())
                filled[np.isnan(filled[:, i]), i] = imputation[name]
            elif self.config.missing_policy == "train_median":
                if observed_train.size == 0:
                    raise FeatureValidationError(f"all training values missing for {name}")
                imputation[name] = float(np.median(observed_train))
                filled[np.isnan(filled[:, i]), i] = imputation[name]
            elif self.config.missing_policy == "constant":
                imputation[name] = float(self.config.constant_value)  # PreprocessingConfig validates finite value.
                filled[np.isnan(filled[:, i]), i] = imputation[name]
            else:  # pragma: no cover - guarded by PreprocessingConfig
                raise FeatureValidationError("unsupported missing policy")
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        zero_variance: list[str] = []
        for i, name in enumerate(names):
            train_col = filled[mask, i]
            if self.config.scaling_policy == "none":
                means[name] = 0.0
                scales[name] = 1.0
            elif self.config.scaling_policy == "standard":
                finite_train = train_col[np.isfinite(train_col)]
                if finite_train.size == 0:
                    raise FeatureValidationError(f"cannot standard-scale entirely missing preserved training feature {name}")
                mean = float(finite_train.mean())
                scale = float(finite_train.std(ddof=0))
                if scale == 0.0:
                    zero_variance.append(name)
                    scale = 1.0
                means[name] = mean
                scales[name] = scale
            else:  # pragma: no cover - guarded by PreprocessingConfig
                raise FeatureValidationError("unsupported scaling policy")
        self.state_ = FittedPreprocessingState(
            names,
            tuple(sorted(set(period_series[mask].tolist()))),
            self.config.missing_policy,
            self.config.scaling_policy,
            imputation,
            means,
            scales,
            tuple(zero_variance),
            original_missing,
        )
        return self

    def transform(self, frame: pd.DataFrame):
        if self.state_ is None:
            raise FeatureValidationError("transform before fit")
        names = self.state_.feature_names
        if list(frame.columns) != list(names):
            raise FeatureValidationError("feature schema mismatch")
        missing = [name for name in names if name not in frame.columns]
        if missing:
            raise FeatureValidationError(f"missing fitted feature columns: {missing}")
        X = frame.loc[:, names].to_numpy(dtype=float).copy()
        if np.isinf(X).any():
            raise FeatureValidationError("input features must not contain infinity")
        original_missing_mask = np.isnan(X.copy())
        for i, name in enumerate(names):
            impute_value = self.state_.imputation_values[name]
            if impute_value is not None:
                X[np.isnan(X[:, i]), i] = impute_value
            X[:, i] = (X[:, i] - self.state_.means[name]) / self.state_.scales[name]
        if self.state_.missing_policy != "preserve" and not np.isfinite(X).all():
            raise FeatureValidationError("processed values must be finite")
        return X, original_missing_mask

    def fit_transform(self, frame: pd.DataFrame, *, feature_names, train_mask, periods):
        return self.fit(frame, feature_names=feature_names, train_mask=train_mask, periods=periods).transform(frame)
