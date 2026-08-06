"""Forward-only local perturbation and frozen-model ablation checks."""
from __future__ import annotations
import math
import torch
from src.models import get_model_capabilities
from .attention_explanations import _infer
from .explanation_contracts import (InterpretabilityValidationError,NON_CAUSAL_WARNINGS,NodePerturbationRecord,NodePerturbationExplanation,FeatureAblationRecord,FeatureAblationExplanation,EdgeAblationRecord,EdgeAblationExplanation)

def _targets(model,snapshot,names):
    configured=tuple(model.config.target_names)
    if tuple(snapshot.target_names)!=configured: raise InterpretabilityValidationError("snapshot and model targets differ")
    if names is None:return configured
    names=tuple(names)
    if not names or len(names)!=len(set(names)) or any(n not in configured for n in names): raise InterpretabilityValidationError("invalid target names")
    return names

def run_node_feature_perturbation(model,snapshot,*,source_node_id,feature_changes,target_names=None,top_k=None):
    if source_node_id not in snapshot.node_ids: raise InterpretabilityValidationError("unknown source node")
    if not feature_changes: raise InterpretabilityValidationError("feature_changes must be non-empty")
    for name,delta in feature_changes.items():
        if name not in snapshot.feature_names: raise InterpretabilityValidationError("unknown feature name")
        if isinstance(delta,bool) or not isinstance(delta,(int,float)) or not math.isfinite(float(delta)): raise InterpretabilityValidationError("feature delta must be finite")
    selected=_targets(model,snapshot,target_names); target_index={n:i for i,n in enumerate(snapshot.target_names)}; source=snapshot.node_ids.index(source_node_id)
    changed=snapshot.x.clone(); specifications=[]
    for name,delta in feature_changes.items():
        col=snapshot.feature_names.index(name); original=float(changed[source,col]); changed[source,col]+=float(delta); specifications.append((name,float(delta),original,float(changed[source,col])))
    baseline=_infer(model,snapshot).predictions; perturbed=_infer(model,snapshot,x=changed).predictions
    records=[]
    for node,node_id in enumerate(snapshot.node_ids):
        for target in selected:
            col=target_index[target]; b=float(baseline[node,col]); p=float(perturbed[node,col]); d=p-b
            records.append(NodePerturbationRecord(node_id,node,target,b,p,d,abs(d),node==source,snapshot.period,snapshot.split))
    order={n:i for i,n in enumerate(selected)}; records.sort(key=lambda r:(-r.absolute_prediction_delta,r.node_id,order[r.target_name]))
    if top_k is not None:
        if isinstance(top_k,bool) or not isinstance(top_k,int) or top_k<=0: raise InterpretabilityValidationError("top_k must be positive")
        records=records[:top_k]
    return NodePerturbationExplanation(model.config.name,snapshot.period,snapshot.split,source_node_id,tuple(specifications),selected,tuple(records),"absolute prediction delta descending, node ID, target order",NON_CAUSAL_WARNINGS)

def run_feature_ablation(model,snapshot,*,feature_names=None,target_names=None,replacement_value=0.0):
    if not math.isfinite(float(replacement_value)): raise InterpretabilityValidationError("replacement value must be finite")
    features=tuple(feature_names or snapshot.feature_names)
    if not features or len(features)!=len(set(features)) or any(x not in snapshot.feature_names for x in features): raise InterpretabilityValidationError("invalid feature names")
    targets=_targets(model,snapshot,target_names); baseline=_infer(model,snapshot).predictions; records=[]
    for feature in features:
        col=snapshot.feature_names.index(feature); values=snapshot.x[:,col]; x=snapshot.x.clone(); x[:,col]=replacement_value; changed=_infer(model,snapshot,x=x).predictions
        for target in targets:
            t=snapshot.target_names.index(target); delta=changed[:,t]-baseline[:,t]; absolute=delta.abs(); maximum=int(torch.argmax(absolute))
            records.append(FeatureAblationRecord(feature,target,float(absolute.mean()),float(absolute[maximum]),snapshot.node_ids[maximum],float(delta[maximum]),float(replacement_value),float(values.mean()),float(values.min()),float(values.max())))
    order={n:i for i,n in enumerate(targets)}; records.sort(key=lambda r:(-r.mean_absolute_prediction_change,r.feature_name,order[r.target_name]))
    return FeatureAblationExplanation(model.config.name,snapshot.period,snapshot.split,targets,tuple(records),"one-at-a-time feature ablation sensitivity",NON_CAUSAL_WARNINGS)

def run_edge_ablation(model,snapshot,*,edge_positions,mode="remove",target_names=None):
    positions=tuple(edge_positions)
    if not positions or any(isinstance(x,bool) or not isinstance(x,int) for x in positions) or len(set(positions))!=len(positions) or any(x<0 or x>=snapshot.edge_index.shape[1] for x in positions): raise InterpretabilityValidationError("edge positions must be unique in-range integers")
    if mode not in {"remove","zero_weight"}: raise InterpretabilityValidationError("mode must be remove or zero_weight")
    capabilities=get_model_capabilities(model.config.name)
    if mode=="zero_weight" and not capabilities.uses_edge_weight: raise InterpretabilityValidationError("zero_weight requires a model that uses edge weights")
    targets=_targets(model,snapshot,target_names); baseline=_infer(model,snapshot).predictions
    if mode=="remove":
        mask=torch.ones(snapshot.edge_index.shape[1],dtype=torch.bool,device=snapshot.edge_index.device); mask[list(positions)]=False; edge=snapshot.edge_index[:,mask]; weight=snapshot.edge_weight[mask]
    else:
        edge=snapshot.edge_index; weight=snapshot.edge_weight.clone(); weight[list(positions)]=0
    changed=_infer(model,snapshot,edge_index=edge,edge_weight=weight).predictions
    pairs=tuple((snapshot.node_ids[int(snapshot.edge_index[0,p])],snapshot.node_ids[int(snapshot.edge_index[1,p])]) for p in positions); weights=tuple(float(snapshot.edge_weight[p]) for p in positions); records=[]
    for target in targets:
        t=snapshot.target_names.index(target); delta=changed[:,t]-baseline[:,t]; absolute=delta.abs(); maximum=int(torch.argmax(absolute))
        records.append(EdgeAblationRecord(positions,pairs,weights,mode,target,float(absolute.mean()),float(absolute[maximum]),snapshot.node_ids[maximum],float(baseline[maximum,t]),float(changed[maximum,t])))
    return EdgeAblationExplanation(model.config.name,snapshot.period,snapshot.split,targets,tuple(records),"frozen-model edge sensitivity",NON_CAUSAL_WARNINGS)
__all__=["run_node_feature_perturbation","run_feature_ablation","run_edge_ablation"]
