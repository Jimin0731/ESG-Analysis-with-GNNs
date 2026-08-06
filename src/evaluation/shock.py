from __future__ import annotations
from src.models import build_model,get_model_capabilities
from .attention_explanations import extract_attention_explanation,trace_attention_paths
from .propagation import extract_propagation_explanation
from .perturbation import run_node_feature_perturbation,run_feature_ablation,run_edge_ablation
from .explanation_contracts import ExplanationBundle,InterpretabilityValidationError,NON_CAUSAL_WARNINGS

def build_shock_explanation(model,snapshot,*,source_node_id,feature_changes,target_names=None,attention_layer=-1,attention_head_aggregation="mean",max_hops=3,top_k=10,feature_ablation=None,edge_ablation=None,model_state_source="constructed"):
    perturbation=run_node_feature_perturbation(model,snapshot,source_node_id=source_node_id,feature_changes=feature_changes,target_names=target_names,top_k=None)
    capabilities=get_model_capabilities(model.config.name); attention=paths=propagation=None
    if capabilities.provides_attention:
        attention=extract_attention_explanation(model,snapshot,layer_index=attention_layer,head_aggregation=attention_head_aggregation)
        paths=trace_attention_paths(attention,source_node_id=source_node_id,max_hops=max_hops,top_k=top_k)
    if model.config.name=="gpr_gnn": propagation=extract_propagation_explanation(model,snapshot)
    features=run_feature_ablation(model,snapshot,feature_names=feature_ablation,target_names=target_names) if feature_ablation is not None else None
    edges=run_edge_ablation(model,snapshot,target_names=target_names,**edge_ablation) if edge_ablation is not None else None
    names=tuple(target_names or model.config.target_names)
    flags=tuple(k for k,v in capabilities.to_report().items() if isinstance(v,bool) and v)
    return ExplanationBundle(model.config.name,snapshot.period,snapshot.split,source_node_id,names,flags,model_state_source,perturbation,attention,paths,propagation,features,edges,NON_CAUSAL_WARNINGS)

def explain_training_result(model_config,training_result,prepared_data,*,split,period,source_node_id,feature_changes,state_source="final",**kwargs):
    if state_source not in {"final","best"}: raise InterpretabilityValidationError("state_source must be final or best")
    if training_result.model_name!=model_config.name: raise InterpretabilityValidationError("model configuration does not match training result")
    if split not in {"train","validation","test"}: raise InterpretabilityValidationError("invalid explicit split")
    matches=[s for s in getattr(prepared_data,split) if s.period==period]
    if len(matches)!=1: raise InterpretabilityValidationError("requested split and period must identify exactly one snapshot")
    model=build_model(model_config); stored=training_result.final_state if state_source=="final" else training_result.best_state
    model.load_state_dict({k:v.detach().clone() for k,v in stored.items()})
    return build_shock_explanation(model,matches[0],source_node_id=source_node_id,feature_changes=feature_changes,model_state_source=state_source,**kwargs)
__all__=["build_shock_explanation","explain_training_result"]
