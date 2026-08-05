from __future__ import annotations
import pandas as pd
from .contracts import FeatureBlock, FeatureAssemblyReport, FeatureValidationError, sort_feature_frame

def assemble_feature_blocks(blocks:list[FeatureBlock], *, canonical_node_order:list[str], periods:list[int], strict=True):
    if not blocks: raise FeatureValidationError("at least one feature block required")
    allnames=[]
    for b in blocks: allnames+=list(b.feature_names)
    if len(allnames)!=len(set(allnames)): raise FeatureValidationError("feature name collision")
    keys=pd.DataFrame([(n,p) for p in sorted(periods) for n in canonical_node_order], columns=["node_id","period"])
    final=keys.copy(); missing={}; extra={}; by={}
    keyset=set(map(tuple,keys.values.tolist()))
    for b in blocks:
        if b.frame.duplicated(["node_id","period"]).any(): raise FeatureValidationError("duplicate keys in block")
        bkeys=set(map(tuple,b.frame[["node_id","period"]].values.tolist()))
        missing[b.name]=[{"node_id":n,"period":int(p)} for n,p in sorted(keyset-bkeys, key=lambda x:(x[1],canonical_node_order.index(x[0]) if x[0] in canonical_node_order else 999))]
        extra[b.name]=[{"node_id":n,"period":int(p)} for n,p in sorted(bkeys-keyset)]
        if strict and (missing[b.name] or extra[b.name]): raise FeatureValidationError(f"strict key mismatch for block {b.name}")
        final=final.merge(b.frame[["node_id","period",*b.feature_names]], on=["node_id","period"], how="left", validate="one_to_one")
        by[b.name]=list(b.feature_names)
    final=sort_feature_frame(final, canonical_node_order)
    report=FeatureAssemblyReport(len(final),len(allnames),by,{c:int(final[c].isna().sum()) for c in allnames},missing,extra,canonical_node_order,sorted(periods),canonical_node_order,sorted(periods))
    return final, report, tuple(p for b in blocks for p in b.provenance)
