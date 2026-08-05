"""OECD/ICIO transaction graph backend."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd
from .contracts import GraphNode, GraphSnapshot, GraphConstructionConfig, GraphConstructionReport
from .common import *

@dataclass(frozen=True)
class ICIOGraphConfig(GraphConstructionConfig):
    backend: str="icio"
    source_column: str|None=None
    missing_tokens: tuple[str,...] = ("", "..", "--", "NA", "N/A", "SUPPRESSED")

def _clean(x): return str(x).strip()
def _numeric(x, missing):
    if pd.isna(x) or _clean(x) in missing: return None
    try: return float(str(x).replace(",",""))
    except ValueError: raise GraphInputError(f"invalid numeric transaction cell: {x}")

def _wide(df, cfg):
    cols=[_clean(c) for c in df.columns]; df=df.copy(); df.columns=cols
    if cfg.source_column is None:
        candidates=[c for c in cols if c.lower() in {"source","source_industry","source industry","industry"}]
        if len(candidates)!=1: raise GraphInputError("ambiguous source column")
        source=candidates[0]
    elif cfg.source_column not in cols: raise GraphInputError("configured source column not found")
    else: source=cfg.source_column
    src=ordered_labels(df[source]); dst=[c for c in cols if c!=source]
    reject_duplicate_labels(src,"source industries"); reject_duplicate_labels(dst,"destination industries")
    labels=[x for x in src if x in set(dst)] or list(dict.fromkeys(src+dst))
    mat=pd.DataFrame(index=labels, columns=labels, dtype=object)
    for _, row in df.iterrows():
        s=_clean(row[source])
        for d in labels:
            if d in df.columns: mat.loc[s,d]=row[d]
    return mat

def build_icio_graph(transactions: pd.DataFrame, *, config: ICIOGraphConfig|None=None, period=None):
    cfg=config or ICIOGraphConfig(); df=transactions.copy()
    clean_index=ordered_labels(df.index); clean_cols=ordered_labels(df.columns)
    if cfg.source_column is not None or df.shape[0] != df.shape[1] or set(clean_index) != set(clean_cols):
        mat=_wide(df,cfg)
    else:
        mat=df.copy(); mat.index=ordered_labels(mat.index); mat.columns=ordered_labels(mat.columns); reject_duplicate_labels(mat.index,"source industries"); reject_duplicate_labels(mat.columns,"destination industries")
        if set(mat.index)!=set(mat.columns): raise GraphInputError("ambiguous square ICIO schema")
        mat=mat.loc[list(mat.index), list(mat.index)]
    labels=list(mat.index)
    if not labels: raise GraphInputError("no usable industries")
    edges=[]; missing=zero=dropped=0
    for i,s in enumerate(labels):
        for j,d in enumerate(labels):
            val=_numeric(mat.loc[s,d], cfg.missing_tokens)
            if val is None: missing+=1; continue
            if not np.isfinite(val): raise GraphInputError("non-finite transaction value")
            if val < 0: raise GraphInputError("negative transaction value")
            if val == 0: zero+=1; continue
            if i==j and not cfg.include_self_loops: dropped+=1; continue
            edges.append((i,j,val,val))
    if not edges: raise GraphInputError("no usable positive flows")
    mask=threshold_mask([e[3] for e in edges], cfg.threshold_policy, cfg.threshold_value); kept=[e for e,m in zip(edges,mask) if m]
    if not kept: raise GraphInputError("no flows retained after threshold")
    kept=aggregate_duplicate_edges(kept,cfg.duplicate_edge_policy); ei=to_edge_index(kept); flows=np.asarray([e[3] for e in kept],float); weights=model_weights(flows,ei,len(labels),cfg.weight_transform)
    rep=GraphConstructionReport("icio",len(labels),len(kept),density(len(labels),len(kept),cfg.include_self_loops),cfg.threshold_policy,cfg.threshold_value,{"missing_count":missing,"zero_count":zero,"dropped_self_loops":dropped,"retained_flow_count":len(kept)})
    return GraphSnapshot(tuple(GraphNode(x,x) for x in labels),ei,weights,flows,"icio",period,{},rep)
