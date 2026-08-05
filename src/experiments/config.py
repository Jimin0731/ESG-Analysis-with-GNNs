from dataclasses import dataclass
from pathlib import Path
import yaml
from src.models import ModelConfig
from .contracts import CheckpointConfig,EarlyStoppingConfig,EvaluationConfig,ExperimentValidationError,NegativeSamplingConfig,SchedulerConfig,TrainingConfig
@dataclass(frozen=True)
class ExperimentConfig:
    name:str; seed:int; safe_feature_names:tuple[str,...]; selected_target_names:tuple[str,...]; model:ModelConfig; training:TrainingConfig; early_stopping:EarlyStoppingConfig; scheduler:SchedulerConfig; checkpoint:CheckpointConfig; evaluation:EvaluationConfig; train_mean_baseline:bool=True; economic_gae:dict|None=None; negative_sampling:NegativeSamplingConfig=NegativeSamplingConfig(); anomaly_threshold:dict|None=None; runtime_output_directory:str|None=None
def _section(data,name,allowed,required=()):
    value=data.get(name,{})
    bad=set(value)-set(allowed)
    if bad: raise ExperimentValidationError(f"unknown keys in {name}: {sorted(bad)}")
    missing=set(required)-set(value)
    if missing: raise ExperimentValidationError(f"missing keys in {name}: {sorted(missing)}")
    return value
def load_experiment_config(path):
    data=yaml.safe_load(Path(path).read_text())
    allowed={"name","seed","safe_feature_names","selected_target_names","model","training","early_stopping","scheduler","checkpoint","evaluation","train_mean_baseline","economic_gae","negative_sampling","anomaly_threshold","runtime_output_directory"}
    if not isinstance(data,dict) or set(data)-allowed: raise ExperimentValidationError(f"unknown top-level sections: {sorted(set(data or {})-allowed)}")
    for key in ("name","seed","safe_feature_names","selected_target_names","model","training"): 
        if key not in data: raise ExperimentValidationError(f"missing required value: {key}")
    model=_section(data,"model",{"name","input_dim","hidden_dim","num_layers","dropout","target_names","attention_heads","activation","residual","seed","options","alpha","propagation_steps","graph_direction","self_loop_weight"},{"name","input_dim","hidden_dim","num_layers","dropout","target_names"})
    training=_section(data,"training",{"epochs","learning_rate","weight_decay","optimizer","gradient_clip_norm","seed","device","target_loss_weights"},{"epochs"})
    early=_section(data,"early_stopping",{"enabled","patience","minimum_delta","monitor","restore_best"})
    scheduler=_section(data,"scheduler",{"name","cosine_min_lr","plateau_factor","plateau_patience"})
    checkpoint=_section(data,"checkpoint",{"path"}); evaluation=_section(data,"evaluation",{"include_predictions","include_timing"})
    economic=_section(data,"economic_gae",{"enabled","input_dim","hidden_dim","latent_dim","dropout","graph_direction","self_loop_weight","feature_loss_weight","structure_loss_weight","anomaly_structural_coefficient"})
    negative=_section(data,"negative_sampling",{"negative_ratio","exclude_self_loops"}); threshold=_section(data,"anomaly_threshold",{"policy","quantile"})
    selected=tuple(data["selected_target_names"])
    if not selected or len(selected)!=len(set(selected)) or any(not n.startswith("target__") for n in selected): raise ExperimentValidationError("selected_target_names must be non-empty, unique, and begin with target__")
    if tuple(model["target_names"])!=selected: raise ExperimentValidationError("model target_names must exactly match selected_target_names")
    if training.get("target_loss_weights") is not None and tuple(training["target_loss_weights"])!=selected: raise ExperimentValidationError("target loss weights must exactly match selected_target_names")
    seed=data["seed"]; training={**training,"seed":seed}; model={**model,"target_names":selected,"seed":seed}
    return ExperimentConfig(data["name"],seed,tuple(data["safe_feature_names"]),selected,ModelConfig(**model),TrainingConfig(**training),EarlyStoppingConfig(**early),SchedulerConfig(**scheduler),CheckpointConfig(**checkpoint),EvaluationConfig(**evaluation),data.get("train_mean_baseline",True),economic or None,NegativeSamplingConfig(**negative),threshold or None,data.get("runtime_output_directory"))
__all__=["ExperimentConfig","load_experiment_config"]
