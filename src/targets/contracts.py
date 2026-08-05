from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np
import pandas as pd

KEY_COLUMNS=("node_id","period")
class TargetValidationError(ValueError): pass

def _json(v):
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, tuple): return [_json(x) for x in v]
    if isinstance(v, list): return [_json(x) for x in v]
    if isinstance(v, dict): return {str(k): _json(x) for k,x in v.items()}
    return v

@dataclass(frozen=True)
class TargetProvenance:
    target_name: str
    target_kind: str
    source_dataset_or_formula: str
    source_column_names: tuple[str,...]=()
    source_feature_names: tuple[str,...]=()
    transformation: str=""
    target_period_semantics: str="feature period t predicts observation period t+horizon"
    forecast_horizon: int=0
    units: str|None=None
    missing_value_policy: str="preserve"
    suitable_as_independent_ground_truth: bool=False
    limitations: str=""
    metadata: dict[str,Any]=field(default_factory=dict)
    def __post_init__(self):
        if not self.target_name.startswith('target__'): raise TargetValidationError('target names must be namespaced with target__')
        if self.target_kind not in {'observed','derived'}: raise TargetValidationError('target_kind must be observed or derived')
        if not isinstance(self.forecast_horizon,int) or self.forecast_horizon<0: raise TargetValidationError('forecast horizon must be a non-negative integer')
    def to_dict(self): return _json(asdict(self))

@dataclass(frozen=True)
class TargetBlock:
    frame: pd.DataFrame
    target_names: tuple[str,...]
    provenance: tuple[TargetProvenance,...]
    metadata: dict[str,Any]=field(default_factory=dict)
    def __post_init__(self): validate_target_frame(self.frame,self.target_names,self.provenance)
    def to_dict(self): return {'frame': self.frame.to_dict(orient='records'), 'target_names': list(self.target_names), 'provenance':[p.to_dict() for p in self.provenance], 'metadata':_json(self.metadata)}

@dataclass(frozen=True)
class TargetAssemblyReport:
    strict: bool
    total_feature_rows: int
    total_target_rows: int
    missing_keys_by_block: dict[str,list[dict[str,Any]]]
    extra_keys_by_block: dict[str,list[dict[str,Any]]]
    removed_keys_by_block: dict[str,list[dict[str,Any]]]=field(default_factory=dict)
    def to_dict(self): return _json(asdict(self))

@dataclass(frozen=True)
class LeakageFinding:
    target_name: str
    finding_type: str
    feature_name: str|None=None
    message: str=""
    severity: str="error"
    metadata: dict[str,Any]=field(default_factory=dict)
    def to_dict(self): return _json(asdict(self))

@dataclass(frozen=True)
class LeakageAuditReport:
    findings: tuple[LeakageFinding,...]
    policy: str="error"
    removed_feature_names: tuple[str,...]=()
    removal_reasons: dict[str,list[str]]=field(default_factory=dict)
    def to_dict(self): return _json(asdict(self))

@dataclass(frozen=True)
class TargetPanel:
    frame: pd.DataFrame
    target_names: tuple[str,...]
    provenance: tuple[TargetProvenance,...]
    assembly_report: TargetAssemblyReport|None=None
    def __post_init__(self): validate_target_frame(self.frame,self.target_names,self.provenance)
    def to_dict(self): return {'frame': self.frame.to_dict(orient='records'), 'target_names':list(self.target_names), 'provenance':[p.to_dict() for p in self.provenance], 'assembly_report': None if self.assembly_report is None else self.assembly_report.to_dict()}

def validate_keys(df: pd.DataFrame):
    for c in KEY_COLUMNS:
        if c not in df.columns: raise TargetValidationError(f'missing key column {c}')
    s=df['node_id']
    if s.isna().any() or (s.astype(str).str.strip()=='').any(): raise TargetValidationError('node_id must be non-empty after trimming')
    if not s.map(lambda x: isinstance(x,str)).all(): raise TargetValidationError('node_id must be strings')
    if not df['period'].map(lambda x: isinstance(x,(int,np.integer)) and not isinstance(x,bool)).all(): raise TargetValidationError('periods must be integer annual periods')
    if df.duplicated(list(KEY_COLUMNS)).any(): raise TargetValidationError('duplicate (node_id, period) keys are not allowed')

def validate_target_frame(df: pd.DataFrame, target_names, provenance):
    if len(target_names)==0: raise TargetValidationError('at least one target is required')
    if len(set(target_names))!=len(target_names): raise TargetValidationError('target names must be unique')
    validate_keys(df)
    prov_names=[p.target_name for p in provenance]
    if tuple(prov_names)!=tuple(target_names): raise TargetValidationError('provenance target names must match target columns')
    for name in target_names:
        if name not in df.columns: raise TargetValidationError(f'missing target column {name}')
        if not pd.api.types.is_numeric_dtype(df[name]): raise TargetValidationError(f'target {name} must be numeric')
        vals=df[name].to_numpy(dtype=float)
        if np.isinf(vals).any(): raise TargetValidationError(f'target {name} must not contain infinity')
