import math,torch
from .contracts import ExperimentValidationError,MetricResult,SplitEvaluation
def regression_metrics(actual,predicted,mask,target_names):
    results=[]
    for i,n in enumerate(target_names):
        y=actual[:,i][mask[:,i]].double(); p=predicted[:,i][mask[:,i]].double(); count=y.numel()
        if not count: raise ExperimentValidationError(f"target {n} has no observed values")
        mae=float((y-p).abs().mean()); rmse=float(torch.sqrt(((y-p)**2).mean())); reason=None;r2=None
        if count<2: reason="fewer than two observed values"
        else:
            denominator=((y-y.mean())**2).sum()
            if denominator==0: reason="zero target variance"
            else: r2=float(1-((y-p)**2).sum()/denominator)
        results.append(MetricResult(n,mae,rmse,r2,count,reason))
    return tuple(results)
def evaluate_predictions(snapshots,predictions):
    target_names=snapshots[0].target_names; actual=torch.cat([s.y for s in snapshots]); pred=torch.cat(predictions); mask=torch.cat([s.target_observed_mask for s in snapshots]); metrics=regression_metrics(actual,pred,mask,target_names)
    records=[]; offset=0
    for s,p in zip(snapshots,predictions,strict=True):
        for row,node in enumerate(s.node_ids):
            for col,name in enumerate(target_names):
                observed=bool(s.target_observed_mask[row,col]); value=float(p[row,col])
                if not math.isfinite(value): raise ExperimentValidationError("predictions must be finite")
                records.append({"node_id":node,"period":s.period,"split":s.split,"target_name":name,"observed_target":float(s.y[row,col]) if observed else None,"prediction":value,"is_observed":observed})
    valid=[m.r2 for m in metrics if m.r2 is not None]
    return SplitEvaluation(snapshots[0].split,metrics,sum(m.mae for m in metrics)/len(metrics),sum(m.rmse for m in metrics)/len(metrics),sum(valid)/len(valid) if valid else None,tuple(records))
__all__=["regression_metrics","evaluate_predictions"]
