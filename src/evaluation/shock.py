from __future__ import annotations
import torch
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
    recorded=training_result.model_configuration
    if not recorded: raise InterpretabilityValidationError("training result lacks the exact recorded model configuration")
    if model_config.to_report()!=recorded or training_result.model_name!=model_config.name: raise InterpretabilityValidationError("model configuration does not exactly match training result")
    if model_config.input_dim!=len(prepared_data.feature_names): raise InterpretabilityValidationError("model input dimension does not match prepared feature names")
    if tuple(model_config.target_names)!=tuple(prepared_data.target_names): raise InterpretabilityValidationError("model targets do not match prepared target order")
    if tuple(training_result.normalized_target_weights)!=tuple(prepared_data.target_names): raise InterpretabilityValidationError("normalized target-weight order does not match prepared targets")
    if split not in {"train","validation","test"}: raise InterpretabilityValidationError("invalid explicit split")
    matches=[s for s in getattr(prepared_data,split) if s.period==period]
    if len(matches)!=1: raise InterpretabilityValidationError("requested split and period must identify exactly one snapshot")
    snapshot=matches[0]
    if tuple(snapshot.feature_names)!=tuple(prepared_data.feature_names) or tuple(snapshot.target_names)!=tuple(prepared_data.target_names): raise InterpretabilityValidationError("snapshot feature or target names do not match prepared data")
    model=build_model(model_config); stored=training_result.final_state if state_source=="final" else training_result.best_state
    expected=model.state_dict()
    if tuple(stored)!=tuple(expected) or any(not isinstance(stored[k],torch.Tensor) or stored[k].shape!=expected[k].shape for k in expected): raise InterpretabilityValidationError("stored state keys or tensor shapes do not exactly match model")
    model.load_state_dict({k:v.detach().clone() for k,v in stored.items()})
    return build_shock_explanation(model,snapshot,source_node_id=source_node_id,feature_changes=feature_changes,model_state_source=state_source,**kwargs)
__all__=["build_shock_explanation","explain_training_result"]
