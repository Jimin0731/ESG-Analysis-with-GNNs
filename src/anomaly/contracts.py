"""Typed contracts for reconstruction-based economic anomaly indicators."""
from dataclasses import dataclass,asdict
import math
import torch
class AnomalyValidationError(ValueError): pass
@dataclass(frozen=True)
class EconomicGAEConfig:
    input_dim:int; hidden_dim:int; latent_dim:int; dropout:float=0.; graph_direction:str="stored"; self_loop_weight:float=1.; feature_loss_weight:float=1.; structure_loss_weight:float=1.; anomaly_structural_coefficient:float=.5; seed:int|None=None
    def __post_init__(self):
        for n in ("input_dim","hidden_dim","latent_dim"):
            v=getattr(self,n)
            if isinstance(v,bool) or not isinstance(v,int) or v<=0: raise AnomalyValidationError(f"{n} must be positive")
        if self.graph_direction not in {"stored","reverse"}: raise AnomalyValidationError("graph_direction must be stored or reverse")
        for n in ("dropout","self_loop_weight","feature_loss_weight","structure_loss_weight","anomaly_structural_coefficient"):
            v=getattr(self,n)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(float(v)) or v<0: raise AnomalyValidationError(f"{n} must be finite and non-negative")
        if self.dropout>=1: raise AnomalyValidationError("dropout must be less than one")
        if self.feature_loss_weight==self.structure_loss_weight==0: raise AnomalyValidationError("at least one training loss weight must be positive")
@dataclass(frozen=True)
class EconomicGAEOutput:
    latent:torch.Tensor; reconstructed_x:torch.Tensor; positive_edge_logits:torch.Tensor; negative_edge_logits:torch.Tensor
@dataclass(frozen=True)
class EconomicGAELoss:
    total:torch.Tensor; feature_reconstruction:torch.Tensor; structure_reconstruction:torch.Tensor; feature_weight:float; structure_weight:float
@dataclass(frozen=True)
class AnomalyScoreOutput:
    anomaly_scores:torch.Tensor; feature_reconstruction_error:torch.Tensor; embedding_consistency_error:torch.Tensor; latent:torch.Tensor; reconstructed_x:torch.Tensor; scoring_config:dict
__all__=["AnomalyValidationError","EconomicGAEConfig","EconomicGAEOutput","EconomicGAELoss","AnomalyScoreOutput"]
