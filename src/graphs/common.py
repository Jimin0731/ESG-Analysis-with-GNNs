"""Shared economic graph utilities."""
from __future__ import annotations
from collections import Counter
import numpy as np
import pandas as pd

class GraphInputError(ValueError): pass

def ordered_labels(labels): return [str(x).strip() for x in labels]
def duplicate_labels(labels): return sorted([k for k,v in Counter(ordered_labels(labels)).items() if v>1])
def reject_duplicate_labels(labels, axis="labels"):
    dup=duplicate_labels(labels)
    if dup: raise GraphInputError(f"duplicate {axis}: {dup}")

def validate_square_labelled_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.shape[0] != frame.shape[1]:
        raise GraphInputError("matrix must be a non-empty square DataFrame")
    frame = frame.copy(); frame.index=ordered_labels(frame.index); frame.columns=ordered_labels(frame.columns)
    reject_duplicate_labels(frame.index,"row labels"); reject_duplicate_labels(frame.columns,"column labels")
    if list(frame.index)!=list(frame.columns): raise GraphInputError("square matrix row and column labels must match in order")
    return frame.apply(pd.to_numeric, errors="raise")

def sort_edges(edges): return sorted(edges, key=lambda e:(int(e[0]), int(e[1]), float(e[2])))
def apply_self_loop_policy(edges, include): return edges if include else [e for e in edges if e[0]!=e[1]]
def aggregate_duplicate_edges(edges, policy="reject"):
    buckets={}
    for s,t,w,r in edges:
        k=(int(s),int(t)); buckets.setdefault(k,[]).append((float(w),float(r)))
    if policy=="reject" and any(len(v)>1 for v in buckets.values()): raise GraphInputError("duplicate edges found")
    if policy not in {"reject","sum","mean"}: raise GraphInputError("unsupported duplicate-edge policy")
    out=[]
    for (s,t), vals in buckets.items():
        arr=np.asarray(vals,float); val=arr.sum(0) if policy=="sum" else arr.mean(0)
        out.append((s,t,float(val[0]),float(val[1])))
    return sort_edges(out)

def threshold_mask(values, policy="absolute", value=0.0):
    vals=np.asarray(values,float)
    if policy=="absolute": return np.abs(vals) > float(value)
    if policy=="percentile":
        if not 0 <= float(value) <= 100: raise GraphInputError("percentile threshold must be 0..100")
        positives=np.abs(vals[vals>0]); thr=np.percentile(positives, value) if positives.size else np.inf
        return np.abs(vals) > thr
    raise GraphInputError("unsupported threshold policy")

def density(node_count, edge_count, include_self_loops=False):
    denom=node_count*node_count if include_self_loops else node_count*(node_count-1)
    return 0.0 if denom<=0 else float(edge_count)/float(denom)
def degree_summary(edge_index, node_count):
    ei=np.asarray(edge_index,int); out=np.bincount(ei[0], minlength=node_count) if ei.size else np.zeros(node_count,int); inn=np.bincount(ei[1], minlength=node_count) if ei.size else np.zeros(node_count,int); return {"in_degree":inn.tolist(),"out_degree":out.tolist()}
def to_edge_index(edges): return np.asarray([[e[0] for e in edges],[e[1] for e in edges]], dtype=np.int64) if edges else np.zeros((2,0), dtype=np.int64)
def model_weights(raw, edge_index=None, node_count=None, transform="raw"):
    raw=np.asarray(raw,float)
    if transform=="raw": out=raw.copy()
    elif transform=="log1p": out=np.log1p(raw)
    elif transform=="row_normalized":
        if edge_index is None or node_count is None: raise GraphInputError("row_normalized requires edge_index and node_count")
        out=np.zeros_like(raw); sums=np.zeros(node_count,float)
        if raw.size: np.add.at(sums, np.asarray(edge_index)[0], raw)
        for i,(s,_) in enumerate(np.asarray(edge_index).T): out[i]=0.0 if sums[s]==0 else raw[i]/sums[s]
    elif transform=="standardized":
        if raw.size <= 1: out=np.zeros_like(raw)
        else:
            std=float(raw.std()); out=np.zeros_like(raw) if std==0 else (raw-raw.mean())/std
    else: raise GraphInputError("unsupported weight transform")
    if not np.isfinite(out).all(): raise GraphInputError("model weights must be finite")
    return out.astype(float)
