"""Typed contracts and validation helpers for feature research panels."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping
import json
import numpy as np
import pandas as pd


class FeatureValidationError(ValueError):
    """Raised when a feature block, split, panel, or preprocessing state is invalid."""


def _jsonable(value: Any, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FeatureValidationError(f"{label} must be JSON-serializable") from exc


def validate_node_id(node_id: Any, *, field_name: str = "node_id") -> str:
    if not isinstance(node_id, str):
        raise FeatureValidationError(f"{field_name} must be a string")
    if node_id == "" or node_id.strip() == "":
        raise FeatureValidationError(f"{field_name} must be non-empty after trimming")
    if node_id != node_id.strip():
        raise FeatureValidationError(f"{field_name} must not contain surrounding whitespace")
    return node_id


def validate_annual_period(period: Any, *, field_name: str = "period") -> int:
    if isinstance(period, bool) or period is None:
        raise FeatureValidationError(f"{field_name} must be an integer annual period")
    if isinstance(period, (np.integer, int)):
        return int(period)
    raise FeatureValidationError(f"{field_name} must be an integer annual period")


def validate_period_series(periods: Any, *, field_name: str = "period") -> pd.Series:
    s = pd.Series(periods)
    if s.isna().any():
        raise FeatureValidationError(f"{field_name} contains missing periods")
    return s.map(lambda value: validate_annual_period(value, field_name=field_name))


def validate_feature_name(name: Any, *, field_name: str = "feature_name") -> str:
    if not isinstance(name, str) or name == "" or name.strip() == "" or name != name.strip():
        raise FeatureValidationError(f"{field_name} must be a non-empty trimmed string")
    if "__" not in name:
        raise FeatureValidationError(f"{field_name} must be namespaced with '__'")
    return name


def validate_metadata_jsonable(value: Mapping[str, Any] | None, *, field_name: str) -> dict[str, Any]:
    data = dict(value or {})
    _jsonable(data, field_name)
    return data


@dataclass(frozen=True, order=True)
class FeatureKey:
    node_id: str
    period: int

    def __post_init__(self) -> None:
        validate_node_id(self.node_id)
        validate_annual_period(self.period)


@dataclass(frozen=True)
class FeatureProvenance:
    feature_name: str
    block_name: str
    source: str
    source_column: str
    transformation: str
    period_semantics: str
    preprocessing_fitted: bool = False
    fit_periods: tuple[int, ...] = ()
    missing_value_policy: str = "preserve"
    units: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        validate_feature_name(self.feature_name)
        for field_name in ("block_name", "source", "source_column", "transformation", "period_semantics", "missing_value_policy"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or value.strip() == "":
                raise FeatureValidationError(f"{field_name} must be a non-empty string")
        object.__setattr__(self, "fit_periods", tuple(validate_annual_period(p, field_name="fit_periods") for p in self.fit_periods))
        self.to_json_dict()

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        _jsonable(data, "feature provenance")
        return data


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    provenance: tuple[FeatureProvenance, ...]
    report: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name.strip() == "":
            raise FeatureValidationError("feature block name must be non-empty")
        names = tuple(validate_feature_name(n) for n in self.feature_names)
        if len(names) != len(set(names)):
            raise FeatureValidationError("duplicate feature names")
        if len(self.provenance) != len(names):
            raise FeatureValidationError("provenance count must equal feature count")
        for expected, provenance in zip(names, self.provenance, strict=True):
            if provenance.feature_name != expected:
                raise FeatureValidationError("provenance feature names must match output features in order")
        validate_feature_frame(self.frame, names)
        validate_metadata_jsonable(self.report, field_name="feature block report")


@dataclass(frozen=True)
class FeatureAssemblyReport:
    total_final_rows: int
    total_final_features: int
    feature_names_by_block: dict[str, list[str]]
    missing_values_by_feature: dict[str, int]
    missing_keys_by_block: dict[str, list[dict[str, Any]]]
    extra_keys_by_block: dict[str, list[dict[str, Any]]]
    node_ids: list[str]
    periods: list[int]
    node_count: int
    period_count: int
    final_node_order: list[str]
    final_period_order: list[int]

    def __post_init__(self) -> None:
        if self.total_final_rows < 0 or self.total_final_features < 0:
            raise FeatureValidationError("assembly report counts must be non-negative")
        for node_id in self.node_ids + self.final_node_order:
            validate_node_id(node_id)
        validate_unique_periods(self.periods, field_name="periods")
        validate_unique_periods(self.final_period_order, field_name="final_period_order")
        if self.node_count != len(self.node_ids) or self.period_count != len(self.periods):
            raise FeatureValidationError("assembly report coverage counts must match node/period lists")
        self.to_json_dict()

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        _jsonable(data, "assembly report")
        return data


@dataclass(frozen=True)
class ChronologicalSplit:
    train_periods: tuple[int, ...]
    validation_periods: tuple[int, ...]
    test_periods: tuple[int, ...]
    train_mask: np.ndarray
    validation_mask: np.ndarray
    test_mask: np.ndarray
    split_counts: dict[str, int]
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        train = validate_unique_periods(self.train_periods, field_name="train_periods")
        validation = validate_unique_periods(self.validation_periods, field_name="validation_periods")
        test = validate_unique_periods(self.test_periods, field_name="test_periods")
        if not train or not validation or not test:
            raise FeatureValidationError("each split must contain at least one period")
        if not (max(train) < min(validation) and max(validation) < min(test)):
            raise FeatureValidationError("split periods must be strictly chronological")
        masks = [_validate_mask(self.train_mask, "train_mask"), _validate_mask(self.validation_mask, "validation_mask"), _validate_mask(self.test_mask, "test_mask")]
        lengths = {m.shape[0] for m in masks}
        if len(lengths) != 1:
            raise FeatureValidationError("split masks must have matching lengths")
        coverage = masks[0].astype(int) + masks[1].astype(int) + masks[2].astype(int)
        if not np.all(coverage == 1):
            raise FeatureValidationError("split masks must be mutually exclusive and cover every row exactly once")
        _jsonable(self.split_counts, "split counts")
        _jsonable(self.diagnostics, "split diagnostics")


def _validate_mask(mask: Any, label: str) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 1 or arr.dtype != bool:
        raise FeatureValidationError(f"{label} must be a one-dimensional boolean array")
    return arr


@dataclass(frozen=True)
class PreprocessingConfig:
    missing_policy: str = "error"
    scaling_policy: str = "none"
    constant_value: float | None = None

    def __post_init__(self) -> None:
        if self.missing_policy not in {"error", "preserve", "train_mean", "train_median", "constant"}:
            raise FeatureValidationError("unsupported missing policy")
        if self.scaling_policy not in {"none", "standard"}:
            raise FeatureValidationError("unsupported scaling policy")
        if self.missing_policy == "constant" and (self.constant_value is None or not np.isfinite(self.constant_value)):
            raise FeatureValidationError("constant missing policy requires a finite constant")


@dataclass(frozen=True)
class FittedPreprocessingState:
    feature_names: tuple[str, ...]
    fit_periods: tuple[int, ...]
    missing_policy: str
    scaling_policy: str
    imputation_values: dict[str, float | None]
    means: dict[str, float]
    scales: dict[str, float]
    zero_variance_features: tuple[str, ...]
    original_missing_by_feature: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(validate_feature_name(n) for n in self.feature_names)
        if not names or len(names) != len(set(names)):
            raise FeatureValidationError("fitted state feature names must be non-empty and unique")
        validate_unique_periods(self.fit_periods, field_name="fit_periods")
        for mapping_name, mapping in (("means", self.means), ("scales", self.scales)):
            if set(mapping) != set(names):
                raise FeatureValidationError(f"{mapping_name} must contain exactly fitted feature names")
            for value in mapping.values():
                if not np.isfinite(value):
                    raise FeatureValidationError(f"{mapping_name} values must be finite")
        if set(self.imputation_values) != set(names):
            raise FeatureValidationError("imputation values must contain exactly fitted feature names")
        for value in self.imputation_values.values():
            if value is not None and not np.isfinite(value):
                raise FeatureValidationError("numeric imputation values must be finite")
        if any(name not in names for name in self.zero_variance_features):
            raise FeatureValidationError("zero-variance features must be fitted feature names")
        self.to_json_dict()

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        _jsonable(data, "fitted preprocessing state")
        return data


@dataclass(frozen=True)
class FeaturePanel:
    node_ids: tuple[str, ...]
    periods: tuple[int, ...]
    raw_features: np.ndarray
    processed_features: np.ndarray
    feature_names: tuple[str, ...]
    original_missing_mask: np.ndarray
    train_mask: np.ndarray
    validation_mask: np.ndarray
    test_mask: np.ndarray
    provenance: tuple[FeatureProvenance, ...]
    assembly_report: FeatureAssemblyReport
    preprocessing_state: FittedPreprocessingState

    def __post_init__(self) -> None:
        nodes = tuple(validate_node_id(n) for n in self.node_ids)
        if len(nodes) != len(set(nodes)):
            raise FeatureValidationError("panel node IDs must be unique")
        validate_unique_periods(self.periods, field_name="periods")
        names = tuple(validate_feature_name(n) for n in self.feature_names)
        if len(names) != len(set(names)):
            raise FeatureValidationError("panel feature names must be unique")
        raw = _validate_2d_float_array(self.raw_features, "raw_features")
        processed = _validate_2d_float_array(self.processed_features, "processed_features")
        if raw.shape != processed.shape or raw.shape[1] != len(names):
            raise FeatureValidationError("panel feature arrays must align with feature names")
        missing = np.asarray(self.original_missing_mask)
        if missing.shape != raw.shape or missing.dtype != bool:
            raise FeatureValidationError("original missing mask must be boolean and match feature array shape")
        split = ChronologicalSplit((), (), (), np.array([], dtype=bool), np.array([], dtype=bool), np.array([], dtype=bool), {}, {}) if False else None
        masks = [_validate_mask(self.train_mask, "train_mask"), _validate_mask(self.validation_mask, "validation_mask"), _validate_mask(self.test_mask, "test_mask")]
        if {m.shape[0] for m in masks} != {raw.shape[0]}:
            raise FeatureValidationError("panel split masks must match feature rows")
        if not np.all(masks[0].astype(int) + masks[1].astype(int) + masks[2].astype(int) == 1):
            raise FeatureValidationError("panel split masks must cover each row exactly once")
        if len(self.provenance) != len(names):
            raise FeatureValidationError("panel provenance must match feature names")


def _validate_2d_float_array(array: Any, label: str) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if arr.ndim != 2:
        raise FeatureValidationError(f"{label} must be two-dimensional")
    if np.isinf(arr).any():
        raise FeatureValidationError(f"{label} must not contain infinity")
    return arr


def validate_unique_periods(periods: Any, *, field_name: str = "periods") -> list[int]:
    values = list(validate_period_series(periods, field_name=field_name))
    if len(values) != len(set(values)):
        raise FeatureValidationError(f"{field_name} must be unique")
    if values != sorted(values):
        raise FeatureValidationError(f"{field_name} must be sorted ascending")
    return values


def validate_feature_frame(df: pd.DataFrame, feature_names: tuple[str, ...]) -> pd.DataFrame:
    if df.empty:
        raise FeatureValidationError("empty feature blocks are not allowed")
    if "node_id" not in df or "period" not in df:
        raise FeatureValidationError("feature frames require node_id and period")
    df["node_id"].map(validate_node_id)
    validate_period_series(df["period"])
    period_types = {type(v) for v in df["period"].tolist()}
    if len(period_types) > 1:
        raise FeatureValidationError("periods must not mix types")
    if df.duplicated(["node_id", "period"]).any():
        raise FeatureValidationError("duplicate (node_id, period) keys")
    names = tuple(validate_feature_name(n) for n in feature_names)
    if not names:
        raise FeatureValidationError("empty feature blocks are not allowed")
    if len(names) != len(set(names)):
        raise FeatureValidationError("duplicate feature names")
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise FeatureValidationError(f"missing feature columns: {missing}")
    for col in names:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise FeatureValidationError(f"non-numeric feature column: {col}")
        values = df[col].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise FeatureValidationError(f"infinite values in feature column: {col}")
    return df


def sort_feature_frame(df: pd.DataFrame, canonical_node_order: list[str] | tuple[str, ...]) -> pd.DataFrame:
    canonical = list(canonical_node_order)
    for node_id in canonical:
        validate_node_id(node_id)
    if len(canonical) != len(set(canonical)):
        raise FeatureValidationError("canonical node order must be unique")
    order = {node_id: i for i, node_id in enumerate(canonical)}
    extra = sorted(set(df["node_id"]) - set(order))
    order.update({node_id: len(order) + i for i, node_id in enumerate(extra)})
    out = df.copy()
    out["_node_order"] = out["node_id"].map(order)
    return out.sort_values(["period", "_node_order"]).drop(columns="_node_order").reset_index(drop=True)
