from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np, pandas as pd

class FeatureValidationError(ValueError): pass

@dataclass(frozen=True, order=True)
class FeatureKey: node_id: str; period: int
@dataclass(frozen=True)
class FeatureProvenance:
    feature_name: str; block_name: str; source: str; source_column: str; transformation: str; period_semantics: str
    preprocessing_fitted: bool=False; fit_periods: tuple[int,...]=(); missing_value_policy: str="preserve"; units: str|None=None; notes: str=""
    def to_json_dict(self): return asdict(self)
@dataclass(frozen=True)
class FeatureBlock:
    name: str; frame: pd.DataFrame; feature_names: tuple[str,...]; provenance: tuple[FeatureProvenance,...]
    def __post_init__(self): validate_feature_frame(self.frame, self.feature_names)
@dataclass(frozen=True)
class FeatureAssemblyReport:
    total_final_rows:int; total_final_features:int; feature_names_by_block:dict[str,list[str]]; missing_values_by_feature:dict[str,int]
    missing_keys_by_block:dict[str,list[dict[str,Any]]]; extra_keys_by_block:dict[str,list[dict[str,Any]]]; node_coverage:list[str]; period_coverage:list[int]; final_node_order:list[str]; final_period_order:list[int]
@dataclass(frozen=True)
class ChronologicalSplit:
    train_periods: tuple[int,...]; validation_periods: tuple[int,...]; test_periods: tuple[int,...]
    train_mask: np.ndarray; validation_mask: np.ndarray; test_mask: np.ndarray; split_counts: dict[str,int]; diagnostics: dict[str,Any]
@dataclass(frozen=True)
class PreprocessingConfig:
    missing_policy: str="error"; scaling_policy: str="none"; constant_value: float|None=None
@dataclass(frozen=True)
class FittedPreprocessingState:
    feature_names: tuple[str,...]; fit_periods: tuple[int,...]; missing_policy: str; scaling_policy: str
    imputation_values: dict[str,float|None]; means: dict[str,float]; scales: dict[str,float]; zero_variance_features: tuple[str,...]
    original_missing_by_feature: dict[str,int]=field(default_factory=dict)
    def to_json_dict(self): return asdict(self)
@dataclass(frozen=True)
class FeaturePanel:
    node_ids: tuple[str,...]; periods: tuple[int,...]; raw_features: np.ndarray; processed_features: np.ndarray; feature_names: tuple[str,...]
    original_missing_mask: np.ndarray; train_mask: np.ndarray; validation_mask: np.ndarray; test_mask: np.ndarray; provenance: tuple[FeatureProvenance,...]
    assembly_report: FeatureAssemblyReport; preprocessing_state: FittedPreprocessingState

def _validate_periods(s: pd.Series):
    vals=s.tolist()
    if any(isinstance(v,bool) or not float(v).is_integer() for v in vals if pd.notna(v)): raise FeatureValidationError("periods must be integer annual values")

def validate_feature_frame(df: pd.DataFrame, feature_names) -> pd.DataFrame:
    if df.empty: raise FeatureValidationError("empty feature blocks are not allowed")
    if "node_id" not in df or "period" not in df: raise FeatureValidationError("feature frames require node_id and period")
    if df["node_id"].map(lambda x: not isinstance(x,str) or not x.strip()).any(): raise FeatureValidationError("node IDs must be non-empty strings")
    _validate_periods(df["period"])
    if df.duplicated(["node_id","period"]).any(): raise FeatureValidationError("duplicate (node_id, period) keys")
    names=list(feature_names)
    if not names: raise FeatureValidationError("empty feature blocks are not allowed")
    if len(names)!=len(set(names)): raise FeatureValidationError("duplicate feature names")
    miss=[c for c in names if c not in df.columns]
    if miss: raise FeatureValidationError(f"missing feature columns: {miss}")
    for c in names:
        if not pd.api.types.is_numeric_dtype(df[c]): raise FeatureValidationError(f"non-numeric feature column: {c}")
        if np.isinf(df[c].to_numpy(dtype=float)).any(): raise FeatureValidationError(f"infinite values in feature column: {c}")
    return df

def sort_feature_frame(df, canonical_node_order):
    order={n:i for i,n in enumerate(canonical_node_order)}
    extra=sorted(set(df.node_id)-set(order)); order.update({n:len(order)+i for i,n in enumerate(extra)})
    out=df.copy(); out["_node_order"]=out.node_id.map(order)
    return out.sort_values(["period","_node_order"]).drop(columns="_node_order").reset_index(drop=True)
