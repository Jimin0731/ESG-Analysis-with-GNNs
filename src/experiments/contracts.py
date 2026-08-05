"""Typed, JSON-safe contracts for chronological experiments."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any
import math
import torch

class ExperimentValidationError(ValueError): pass

def _finite(name, value, *, positive=False, nonnegative=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(float(value)): raise ExperimentValidationError(f"{name} must be finite numeric")
    if positive and value<=0: raise ExperimentValidationError(f"{name} must be positive")
    if nonnegative and value<0: raise ExperimentValidationError(f"{name} must be non-negative")

@dataclass(frozen=True)
class TrainingConfig:
    epochs:int; learning_rate:float=1e-3; weight_decay:float=0.; optimizer:str="adamw"; gradient_clip_norm:float|None=None; seed:int=0; device:str="cpu"; target_loss_weights:dict[str,float]|None=None
    def __post_init__(self):
        if isinstance(self.epochs,bool) or not isinstance(self.epochs,int) or self.epochs<=0: raise ExperimentValidationError("epochs must be a positive integer")
        _finite("learning_rate",self.learning_rate,positive=True); _finite("weight_decay",self.weight_decay,nonnegative=True)
        if self.optimizer not in {"adam","adamw"}: raise ExperimentValidationError("optimizer must be adam or adamw")
        if self.gradient_clip_norm is not None: _finite("gradient_clip_norm",self.gradient_clip_norm,positive=True)
        if isinstance(self.seed,bool) or not isinstance(self.seed,int) or self.seed<0: raise ExperimentValidationError("seed must be a non-negative integer")
        if self.device!="cpu": raise ExperimentValidationError("canonical experiments support device=cpu")
        for v in (self.target_loss_weights or {}).values(): _finite("target loss weight",v,positive=True)
    def to_report(self): return asdict(self)

@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled:bool=True; patience:int=10; minimum_delta:float=0.; monitor:str="validation_loss"; restore_best:bool=True
    def __post_init__(self):
        if self.enabled and (isinstance(self.patience,bool) or not isinstance(self.patience,int) or self.patience<=0): raise ExperimentValidationError("patience must be positive when enabled")
        _finite("minimum_delta",self.minimum_delta,nonnegative=True)
        if self.monitor!="validation_loss": raise ExperimentValidationError("only validation_loss is supported")

@dataclass(frozen=True)
class SchedulerConfig:
    name:str="none"; cosine_min_lr:float=0.; plateau_factor:float=.5; plateau_patience:int=2
    def __post_init__(self):
        if self.name not in {"none","cosine","reduce_on_plateau"}: raise ExperimentValidationError("unsupported scheduler")
        _finite("cosine_min_lr",self.cosine_min_lr,nonnegative=True); _finite("plateau_factor",self.plateau_factor,positive=True)
        if self.plateau_factor>=1: raise ExperimentValidationError("plateau_factor must be below one")
        if isinstance(self.plateau_patience,bool) or not isinstance(self.plateau_patience,int) or self.plateau_patience<0: raise ExperimentValidationError("plateau_patience must be non-negative")

@dataclass(frozen=True)
class EvaluationConfig: include_predictions:bool=True; include_timing:bool=False
@dataclass(frozen=True)
class CheckpointConfig: path:str|None=None

@dataclass(frozen=True)
class SupervisedSnapshot:
    period:int; split:str; node_ids:tuple[str,...]; x:torch.Tensor; y:torch.Tensor; target_observed_mask:torch.Tensor; edge_index:torch.Tensor; edge_weight:torch.Tensor; feature_names:tuple[str,...]; target_names:tuple[str,...]
    def __post_init__(self):
        if isinstance(self.period,bool) or not isinstance(self.period,int): raise ExperimentValidationError("period must be integer annual period")
        if self.split not in {"train","validation","test"}: raise ExperimentValidationError("invalid split")
        n=len(self.node_ids)
        if not n or len(set(self.node_ids))!=n: raise ExperimentValidationError("node IDs must be non-empty and unique")
        if not self.feature_names or len(set(self.feature_names))!=len(self.feature_names): raise ExperimentValidationError("feature names must be ordered and unique")
        if not self.target_names or len(set(self.target_names))!=len(self.target_names) or any(not x.startswith("target__") for x in self.target_names): raise ExperimentValidationError("invalid target names")
        if self.x.shape!=(n,len(self.feature_names)) or self.y.shape!=(n,len(self.target_names)) or self.target_observed_mask.shape!=self.y.shape: raise ExperimentValidationError("snapshot tensor shapes mismatch")
        if self.target_observed_mask.dtype!=torch.bool: raise ExperimentValidationError("observed mask must be boolean")
        if not torch.isfinite(self.x).all() or not torch.isfinite(self.y[self.target_observed_mask]).all(): raise ExperimentValidationError("observed data must be finite")
        if not torch.isnan(self.y[~self.target_observed_mask]).all(): raise ExperimentValidationError("missing targets must correspond exactly to false mask")
        if self.edge_index.shape[0]!=2 or self.edge_weight.shape!=(self.edge_index.shape[1],): raise ExperimentValidationError("graph shapes mismatch")
        if self.edge_index.numel() and (self.edge_index.min()<0 or self.edge_index.max()>=n): raise ExperimentValidationError("invalid graph endpoint")
        if not torch.isfinite(self.edge_weight).all() or (self.edge_weight<0).any(): raise ExperimentValidationError("edge weights must be finite and non-negative")
    def metadata(self): return {"period":self.period,"split":self.split,"node_ids":list(self.node_ids),"feature_names":list(self.feature_names),"target_names":list(self.target_names)}

@dataclass(frozen=True)
class PreparedSupervisedData:
    train:tuple[SupervisedSnapshot,...]; validation:tuple[SupervisedSnapshot,...]; test:tuple[SupervisedSnapshot,...]; feature_names:tuple[str,...]; target_names:tuple[str,...]
@dataclass(frozen=True)
class EpochRecord:
    epoch:int; train_total_loss:float; validation_total_loss:float; train_per_target:dict[str,float]; validation_per_target:dict[str,float]; train_observed_counts:dict[str,int]; validation_observed_counts:dict[str,int]; learning_rate:float; became_best:bool
@dataclass(frozen=True)
class MetricResult: target_name:str; mae:float; rmse:float; r2:float|None; observed_count:int; r2_undefined_reason:str|None=None
@dataclass(frozen=True)
class SplitEvaluation: split:str; metrics:tuple[MetricResult,...]; macro_mae:float; macro_rmse:float; macro_r2:float|None; predictions:tuple[dict[str,Any],...]
@dataclass(frozen=True)
class TrainingResult:
    model_name:str; seed:int; requested_epochs:int; completed_epochs:int; best_epoch:int; stopped_early:bool; stop_reason:str; best_monitored_value:float; restored_best:bool; normalized_target_weights:dict[str,float]; history:tuple[EpochRecord,...]; evaluations:dict[str,SplitEvaluation]; best_state:dict[str,torch.Tensor]=field(repr=False,compare=False)
    def to_report(self):
        d=asdict(self); d.pop("best_state"); return d
@dataclass(frozen=True)
class BaselineResult: means:dict[str,float]; evaluations:dict[str,SplitEvaluation]
@dataclass(frozen=True)
class AnomalyTrainingResult: seed:int; best_epoch:int; best_validation_loss:float; restored_best:bool; scores:dict[str,tuple[dict[str,Any],...]]; threshold:dict[str,Any]|None
@dataclass(frozen=True)
class ExperimentReport:
    experiment_name:str; seed:int; model_capabilities:dict[str,Any]; feature_names:tuple[str,...]; target_names:tuple[str,...]; split_periods:dict[str,tuple[int,...]]; preprocessing_fit_periods:tuple[int,...]; target_provenance:tuple[dict[str,Any],...]; leakage_audit:dict[str,Any]; training:dict[str,Any]; baseline:dict[str,Any]; software_versions:dict[str,str]
    def to_dict(self): return asdict(self)

__all__=["ExperimentValidationError","TrainingConfig","EarlyStoppingConfig","SchedulerConfig","EvaluationConfig","CheckpointConfig","SupervisedSnapshot","PreparedSupervisedData","EpochRecord","MetricResult","SplitEvaluation","TrainingResult","BaselineResult","AnomalyTrainingResult","ExperimentReport"]
