from dataclasses import dataclass
from pathlib import Path
import yaml
from src.models import ModelConfig
from .contracts import CheckpointConfig,EarlyStoppingConfig,EvaluationConfig,ExperimentValidationError,SchedulerConfig,TrainingConfig
@dataclass(frozen=True)
class ExperimentConfig:
    name:str; seed:int; safe_feature_names:tuple[str,...]; selected_target_names:tuple[str,...]; model:ModelConfig; training:TrainingConfig; early_stopping:EarlyStoppingConfig; scheduler:SchedulerConfig; checkpoint:CheckpointConfig; evaluation:EvaluationConfig; train_mean_baseline:bool=True; economic_gae:dict|None=None; negative_sampling:dict|None=None; anomaly_threshold:dict|None=None; runtime_output_directory:str|None=None
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
    seed=data["seed"]; training={**training,"seed":seed}; model={**model,"target_names":tuple(model["target_names"]),"seed":seed}
    return ExperimentConfig(data["name"],seed,tuple(data["safe_feature_names"]),tuple(data["selected_target_names"]),ModelConfig(**model),TrainingConfig(**training),EarlyStoppingConfig(**early),SchedulerConfig(**scheduler),CheckpointConfig(**checkpoint),EvaluationConfig(**evaluation),data.get("train_mean_baseline",True),data.get("economic_gae"),data.get("negative_sampling"),data.get("anomaly_threshold"),data.get("runtime_output_directory"))
__all__=["ExperimentConfig","load_experiment_config"]
