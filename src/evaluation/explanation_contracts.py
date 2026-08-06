"""Immutable, JSON-safe contracts for post-training interpretation."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any

class InterpretabilityValidationError(ValueError): pass

@dataclass(frozen=True)
class ExplanationWarning: code:str; message:str
@dataclass(frozen=True)
class AttentionHeadRecord: edge_position:int; source_index:int; source_node_id:str; target_index:int; target_node_id:str; layer_index:int; head_index:int; attention_coefficient:float; economic_edge_weight:float|None
@dataclass(frozen=True)
class AttentionEdgeRecord: edge_position:int; source_index:int; source_node_id:str; target_index:int; target_node_id:str; layer_index:int; aggregated_attention_coefficient:float; economic_edge_weight:float|None
@dataclass(frozen=True)
class AttentionExplanation: model_name:str; period:int; split:str; graph_direction:str; layer_index:int; head_aggregation:str; normalization:str; normalization_valid:bool; uses_economic_edge_weight:bool; heads:tuple[AttentionHeadRecord,...]; edges:tuple[AttentionEdgeRecord,...]; ranking_rule:str; warnings:tuple[ExplanationWarning,...]
@dataclass(frozen=True)
class PropagationCoefficientRecord: step:int; coefficient:float; alpha:float; graph_direction:str; state_kind:str
@dataclass(frozen=True)
class PropagationExplanation: model_name:str; period:int; split:str; records:tuple[PropagationCoefficientRecord,...]; statement:str; warnings:tuple[ExplanationWarning,...]
@dataclass(frozen=True)
class NodePerturbationRecord: node_id:str; node_index:int; target_name:str; baseline_prediction:float; perturbed_prediction:float; prediction_delta:float; absolute_prediction_delta:float; is_source:bool; period:int; split:str
@dataclass(frozen=True)
class NodePerturbationExplanation: model_name:str; period:int; split:str; source_node_id:str; feature_changes:tuple[tuple[str,float,float,float],...]; target_names:tuple[str,...]; records:tuple[NodePerturbationRecord,...]; ranking_rule:str; warnings:tuple[ExplanationWarning,...]
@dataclass(frozen=True)
class FeatureAblationRecord: feature_name:str; target_name:str; mean_absolute_prediction_change:float; maximum_absolute_prediction_change:float; maximum_node_id:str; signed_change_at_maximum:float; replacement_value:float; original_mean:float; original_minimum:float; original_maximum:float
@dataclass(frozen=True)
class FeatureAblationExplanation: model_name:str; period:int; split:str; target_names:tuple[str,...]; records:tuple[FeatureAblationRecord,...]; label:str; warnings:tuple[ExplanationWarning,...]
@dataclass(frozen=True)
class EdgeAblationRecord: edge_positions:tuple[int,...]; source_target_node_ids:tuple[tuple[str,str],...]; economic_edge_weights:tuple[float,...]; mode:str; target_name:str; mean_absolute_prediction_change:float; maximum_absolute_prediction_change:float; maximum_node_id:str; baseline_prediction_at_maximum:float; ablated_prediction_at_maximum:float
@dataclass(frozen=True)
class EdgeAblationExplanation: model_name:str; period:int; split:str; target_names:tuple[str,...]; records:tuple[EdgeAblationRecord,...]; label:str; warnings:tuple[ExplanationWarning,...]
@dataclass(frozen=True)
class AttentionPathRecord: destination_node_id:str; hop_count:int; path_node_ids:tuple[str,...]; path_edge_positions:tuple[int,...]; attention_path_score:float
@dataclass(frozen=True)
class AttentionPathExplanation: source_node_id:str; max_hops:int; records:tuple[AttentionPathRecord,...]; statement:str; warnings:tuple[ExplanationWarning,...]
@dataclass(frozen=True)
class ExplanationBundle: model_name:str; period:int; split:str; source_node_id:str; target_names:tuple[str,...]; model_capabilities:tuple[str,...]; model_state_source:str; perturbation:NodePerturbationExplanation; attention:AttentionExplanation|None=None; attention_paths:AttentionPathExplanation|None=None; propagation:PropagationExplanation|None=None; feature_ablation:FeatureAblationExplanation|None=None; edge_ablation:EdgeAblationExplanation|None=None; warnings:tuple[ExplanationWarning,...]=()
@dataclass(frozen=True)
class ExplanationExportManifest: generated_files:tuple[str,...]; warnings:tuple[ExplanationWarning,...]

def to_json_dict(value:Any)->dict[str,Any]: return asdict(value)

NON_CAUSAL_WARNINGS=(ExplanationWarning("attention_association","Attention weights are model-internal association scores, not causal effects."),ExplanationWarning("local_sensitivity","Feature and edge perturbations measure local model sensitivity to an artificial input change. They do not establish real-world causal impact."),ExplanationWarning("distinct_quantities","Attention path scores and prediction changes are distinct quantities and must not be presented as interchangeable."),ExplanationWarning("model_dependence","Results depend on the trained model, feature scaling, graph construction, target definition and selected period."))

__all__=["InterpretabilityValidationError","ExplanationWarning","AttentionHeadRecord","AttentionEdgeRecord","AttentionExplanation","PropagationCoefficientRecord","PropagationExplanation","NodePerturbationRecord","NodePerturbationExplanation","FeatureAblationRecord","FeatureAblationExplanation","EdgeAblationRecord","EdgeAblationExplanation","AttentionPathRecord","AttentionPathExplanation","ExplanationBundle","ExplanationExportManifest","NON_CAUSAL_WARNINGS","to_json_dict"]
