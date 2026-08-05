"""USE/MAKE economic graph backend."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd
from .contracts import GraphNode, GraphSnapshot, GraphConstructionConfig, GraphConstructionReport, LeontiefComputationReport
from .common import *

@dataclass(frozen=True)
class UseMakeGraphConfig(GraphConstructionConfig):
    backend: str = "use_make"
    use_orientation: str = "commodity_by_sector"
    make_orientation: str = "sector_by_commodity"
    min_alignment_coverage: float = 0.5
    negative_policy: str = "reject"
    leontief_fallback: str = "error"

def _orient(df, orient, name):
    df=df.copy(); df.index=ordered_labels(df.index); df.columns=ordered_labels(df.columns)
    reject_duplicate_labels(df.index, f"{name} rows"); reject_duplicate_labels(df.columns, f"{name} columns")
    vals=df.apply(pd.to_numeric, errors="raise")
    if orient not in {"commodity_by_sector","sector_by_commodity"}: raise GraphInputError("explicit USE/MAKE orientation is required")
    return vals if orient=="commodity_by_sector" else vals

def compute_leontief_inverse(A, fallback_policy="error"):
    if fallback_policy not in {"error","pinv","regularized"}: raise GraphInputError("invalid Leontief fallback policy")
    A=np.asarray(A,float)
    if A.ndim!=2 or A.shape[0]!=A.shape[1] or not np.isfinite(A).all(): raise GraphInputError("A must be square and finite")
    M=np.eye(A.shape[0])-A; cond=float(np.linalg.cond(M)); method="inverse"; used=False
    try:
        if not np.isfinite(cond) or cond>1e12: raise np.linalg.LinAlgError("ill-conditioned")
        L=np.linalg.inv(M)
    except np.linalg.LinAlgError:
        if fallback_policy=="error": raise
        used=True
        if fallback_policy=="pinv": method="pinv"; L=np.linalg.pinv(M)
        else: method="regularized"; L=np.linalg.inv(M + np.eye(M.shape[0])*1e-8)
    return L, LeontiefComputationReport(A.shape, method, fallback_policy, cond, used)

def build_use_make_graph(use_table: pd.DataFrame, make_table: pd.DataFrame, *, config: UseMakeGraphConfig|None=None, period=None):
    cfg=config or UseMakeGraphConfig(); use=_orient(use_table,cfg.use_orientation,"USE"); make=_orient(make_table,cfg.make_orientation,"MAKE")
    if cfg.negative_policy!="reject": raise GraphInputError("only reject negative policy is supported")
    if not np.isfinite(use.to_numpy(float)).all() or not np.isfinite(make.to_numpy(float)).all(): raise GraphInputError("economic values must be finite")
    if (use.to_numpy(float)<-1e-12).any() or (make.to_numpy(float)<-1e-12).any(): raise GraphInputError("negative economic values are not allowed")
    common_comm=[x for x in use.index if x in set(make.columns)]; common_sec=[x for x in use.columns if x in set(make.index)]
    coverage=min(len(common_comm)/max(len(use.index),1), len(common_sec)/max(len(use.columns),1), len(common_sec)/max(len(make.index),1), len(common_comm)/max(len(make.columns),1))
    if coverage < cfg.min_alignment_coverage: raise GraphInputError("alignment coverage below configured minimum")
    use=use.loc[common_comm, common_sec]; make=make.loc[common_sec, common_comm]
    if use.shape[0]!=make.shape[1] or use.shape[1]!=make.shape[0]: raise GraphInputError("incompatible USE/MAKE dimensions")
    commodity_output=use.sum(axis=1).to_numpy(float)+make.sum(axis=0).to_numpy(float); sector_output=make.sum(axis=1).to_numpy(float)
    B=use.to_numpy(float)/(commodity_output[:,None]+1e-12); D=make.to_numpy(float)/(sector_output[:,None]+1e-12); A=B.T @ D.T
    if not np.isfinite(A).all(): raise GraphInputError("technical coefficients must be finite")
    raw=[]
    for s in range(A.shape[0]):
        for t in range(A.shape[1]): raw.append((s,t,float(A[s,t]),float(A[s,t])))
    raw=apply_self_loop_policy(raw, cfg.include_self_loops); vals=[e[3] for e in raw]; mask=threshold_mask(vals,cfg.threshold_policy,cfg.threshold_value); kept=[e for e,m in zip(raw,mask) if m]
    kept=aggregate_duplicate_edges(kept,cfg.duplicate_edge_policy); ei=to_edge_index(kept); flows=np.asarray([e[3] for e in kept],float); weights=model_weights(flows,ei,len(common_sec),cfg.weight_transform)
    rep=GraphConstructionReport("use_make",len(common_sec),len(kept),density(len(common_sec),len(kept),cfg.include_self_loops),cfg.threshold_policy,cfg.threshold_value,{"alignment_coverage":coverage,"removed_use_commodities":[x for x in use_table.index.astype(str) if x not in common_comm],"removed_sectors":[x for x in ordered_labels(use_table.columns) if x not in common_sec],"B_shape":B.shape,"D_shape":D.shape,"A":A})
    nodes=tuple(GraphNode(x,x) for x in common_sec)
    return GraphSnapshot(nodes,ei,weights,flows,"use_make",period,{"unthresholded_A":A},rep)
