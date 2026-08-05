from dataclasses import dataclass
import torch
from .contracts import ExperimentValidationError
@dataclass(frozen=True)
class MaskedLossResult: total:torch.Tensor; per_target:dict[str,torch.Tensor]; observed_counts:dict[str,int]; normalized_weights:dict[str,float]
def masked_multi_target_mse(predictions,targets,mask,target_names,weights=None):
    if predictions.shape!=targets.shape or mask.shape!=targets.shape: raise ExperimentValidationError("loss tensor shapes mismatch")
    if weights is not None and tuple(weights)!=tuple(target_names): raise ExperimentValidationError("target loss weight names must exactly match target names")
    raw={n:float((weights or {}).get(n,1.)) for n in target_names}; total_weight=sum(raw.values()); normalized={n:v/total_weight for n,v in raw.items()}
    losses={}; counts={}
    for i,n in enumerate(target_names):
        counts[n]=int(mask[:,i].sum())
        losses[n]=((predictions[:,i][mask[:,i]]-targets[:,i][mask[:,i]])**2).mean() if counts[n] else predictions[:,i].sum()*0
    observed=[n for n in target_names if counts[n]]
    if not observed: raise ExperimentValidationError("loss contains no observed labels")
    return MaskedLossResult(sum(losses[n]*normalized[n] for n in observed),losses,counts,normalized)
__all__=["MaskedLossResult","masked_multi_target_mse"]
