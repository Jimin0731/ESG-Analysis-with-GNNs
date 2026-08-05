import torch
from .contracts import BaselineResult,ExperimentValidationError
from .metrics import evaluate_predictions
class TrainMeanBaseline:
    def __init__(self): self.target_names=(); self.means={}
    def fit(self,snapshots):
        self.target_names=snapshots[0].target_names
        y=torch.cat([s.y for s in snapshots]); mask=torch.cat([s.target_observed_mask for s in snapshots])
        for i,n in enumerate(self.target_names):
            if not mask[:,i].any(): raise ExperimentValidationError(f"target {n} has no observed train label")
            self.means[n]=float(y[:,i][mask[:,i]].mean())
        return self
    def predict(self,snapshot): return torch.tensor([self.means[n] for n in self.target_names],dtype=snapshot.x.dtype).repeat(len(snapshot.node_ids),1)
    def evaluate(self,data): return BaselineResult(dict(self.means),{s:evaluate_predictions(getattr(data,s),[self.predict(x) for x in getattr(data,s)]) for s in ("train","validation","test")})
__all__=["TrainMeanBaseline"]
