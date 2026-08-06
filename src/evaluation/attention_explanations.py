"""Direction-preserving extraction and descriptive path tracing."""
from __future__ import annotations
import math
import torch
from .explanation_contracts import (AttentionEdgeRecord,AttentionExplanation,AttentionHeadRecord,AttentionPathExplanation,AttentionPathRecord,InterpretabilityValidationError,NON_CAUSAL_WARNINGS)

def _infer(model,snapshot,*,aux=False,x=None,edge_index=None,edge_weight=None):
    prior=model.training; model.eval()
    try:
        with torch.no_grad(): return model(snapshot.x if x is None else x,snapshot.edge_index if edge_index is None else edge_index,snapshot.edge_weight if edge_weight is None else edge_weight,return_aux=aux)
    finally: model.train(prior)

def validate_destination_attention_normalization(edge_index, coefficients, *, weighted=False, edge_weight=None, atol=1e-5):
    if coefficients.ndim!=2 or edge_index.shape!=(2,coefficients.shape[0]): raise InterpretabilityValidationError("attention and edge shapes disagree")
    if not torch.isfinite(coefficients).all(): raise InterpretabilityValidationError("attention must be finite")
    if weighted:
        if not isinstance(edge_weight,torch.Tensor) or edge_weight.shape!=(coefficients.shape[0],): raise InterpretabilityValidationError("weighted attention requires aligned edge weights shaped [E]")
        if not torch.isfinite(edge_weight).all() or (edge_weight<0).any(): raise InterpretabilityValidationError("edge weights must be finite and non-negative")
    for target in torch.unique(edge_index[1]):
        sums=coefficients[edge_index[1]==target].sum(0)
        expected=0.0 if weighted and float(edge_weight[edge_index[1]==target].sum())==0.0 else 1.0
        if not torch.allclose(sums,torch.full_like(sums,expected),atol=atol): raise InterpretabilityValidationError("incoming attention does not match destination economic mass")
    return True

def extract_attention_explanation(model,snapshot,*,layer_index=-1,head_aggregation="mean",top_k=None):
    name=getattr(getattr(model,"config",None),"name",None)
    if name not in {"gat","weighted_gat"}: raise InterpretabilityValidationError("model does not provide edge attention")
    out=_infer(model,snapshot,aux=True); aux=out.auxiliary; layers=aux.get("attention_by_layer")
    if not isinstance(layers,tuple) or not layers: raise InterpretabilityValidationError("missing per-layer attention")
    index=layer_index if layer_index>=0 else len(layers)+layer_index
    if isinstance(layer_index,bool) or not 0<=index<len(layers): raise InterpretabilityValidationError("invalid attention layer")
    edge=aux.get("edge_index"); alpha=layers[index]
    if not torch.equal(edge,snapshot.edge_index): raise InterpretabilityValidationError("returned edge index does not match snapshot order")
    validate_destination_attention_normalization(edge,alpha,weighted=name=="weighted_gat",edge_weight=snapshot.edge_weight if name=="weighted_gat" else None)
    heads=alpha.shape[1]
    if isinstance(head_aggregation,int) and not isinstance(head_aggregation,bool):
        if not 0<=head_aggregation<heads: raise InterpretabilityValidationError("invalid attention head")
        aggregate=alpha[:,head_aggregation]; label=str(head_aggregation)
    elif head_aggregation in {"mean","max"}:
        aggregate=alpha.mean(1) if head_aggregation=="mean" else alpha.max(1).values; label=head_aggregation
    else: raise InterpretabilityValidationError("head aggregation must be mean, max, or a head index")
    weights=snapshot.edge_weight if snapshot.edge_weight is not None else None
    head_records=[]; edge_records=[]
    for pos in range(edge.shape[1]):
        s,t=int(edge[0,pos]),int(edge[1,pos]); weight=float(weights[pos]) if weights is not None else None
        for h in range(heads): head_records.append(AttentionHeadRecord(pos,s,snapshot.node_ids[s],t,snapshot.node_ids[t],index,h,float(alpha[pos,h]),weight))
        edge_records.append(AttentionEdgeRecord(pos,s,snapshot.node_ids[s],t,snapshot.node_ids[t],index,float(aggregate[pos]),weight))
    ranked=sorted(edge_records,key=lambda r:(-r.aggregated_attention_coefficient,r.source_node_id,r.target_node_id,r.edge_position))
    if top_k is not None:
        if isinstance(top_k,bool) or not isinstance(top_k,int) or top_k<=0: raise InterpretabilityValidationError("top_k must be a positive integer")
        ranked=ranked[:top_k]
    return AttentionExplanation(name,snapshot.period,snapshot.split,"source_to_target",index,label,"incoming_target_per_head",True,name=="weighted_gat",tuple(head_records),tuple(ranked),"aggregated attention descending, source ID, target ID, edge position",NON_CAUSAL_WARNINGS)

def trace_attention_paths(explanation,*,source_node_id,max_hops=3,top_k=10):
    for key,value in (("max_hops",max_hops),("top_k",top_k)):
        if isinstance(value,bool) or not isinstance(value,int) or value<=0: raise InterpretabilityValidationError(f"{key} must be a positive integer")
    nodes={e.source_node_id for e in explanation.edges}|{e.target_node_id for e in explanation.edges}
    if source_node_id not in nodes: raise InterpretabilityValidationError("source node is absent from attention graph")
    adjacency={n:[] for n in nodes}
    for e in explanation.edges: adjacency[e.source_node_id].append(e)
    candidates=[]
    def walk(node,path_nodes,path_edges,score):
        if path_edges: candidates.append(AttentionPathRecord(node,len(path_edges),tuple(path_nodes),tuple(path_edges),score))
        if len(path_edges)==max_hops:return
        for e in sorted(adjacency.get(node,()),key=lambda x:(x.edge_position,x.target_node_id)):
            if e.target_node_id not in path_nodes: walk(e.target_node_id,path_nodes+[e.target_node_id],path_edges+[e.edge_position],score*e.aggregated_attention_coefficient)
    walk(source_node_id,[source_node_id],[],1.0)
    best={}
    for r in candidates:
        key=r.destination_node_id; order=(-r.attention_path_score,r.hop_count,r.path_edge_positions)
        if key not in best or order<best[key][0]: best[key]=(order,r)
    records=sorted((x[1] for x in best.values()),key=lambda r:(-r.attention_path_score,r.hop_count,r.destination_node_id,r.path_edge_positions))[:top_k]
    return AttentionPathExplanation(source_node_id,max_hops,tuple(records),"Attention path scores trace high-attention routes inside this fitted model. They do not estimate causal economic propagation.",NON_CAUSAL_WARNINGS)

__all__=["extract_attention_explanation","trace_attention_paths","validate_destination_attention_normalization"]
