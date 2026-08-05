from __future__ import annotations
import numpy as np, pandas as pd
from .contracts import FeatureBlock, FeatureProvenance, FeatureValidationError, sort_feature_frame

def build_environmental_feature_block(df, *, levels=(), log1p=(), intensities=None, canonical_node_order=None, invalid_denominator_policy="error"):
    intensities=intensities or {}; out=df[["node_id","period"]].copy(); names=[]
    for c in levels:
        nm=f"environmental__{c}_level"; out[nm]=pd.to_numeric(df[c]); names.append(nm)
    for c in log1p:
        v=pd.to_numeric(df[c]);
        if (v.dropna()<0).any(): raise FeatureValidationError("log1p requires non-negative environmental values")
        nm=f"environmental__{c}_log1p"; out[nm]=np.log1p(v); names.append(nm)
    for nm,cfg in intensities.items():
        num,den=cfg["numerator"],cfg.get("denominator")
        if not den: raise FeatureValidationError("intensity requires named denominator")
        d=pd.to_numeric(df[den]); bad=d<=0
        if bad.any() and invalid_denominator_policy=="error": raise FeatureValidationError("intensity denominator must be positive")
        val=pd.to_numeric(df[num])/d.mask(bad)
        fn=f"environmental__{nm}"; out[fn]=val; names.append(fn)
    out=sort_feature_frame(out, canonical_node_order or sorted(out.node_id.unique()))
    prov=tuple(FeatureProvenance(n,"environmental","normalized_environmental",n.split('__',1)[1],"configured", "annual observation",False,(),"preserve",None,"intensity ratios use explicit numerator and denominator") for n in names)
    return FeatureBlock("environmental",out,tuple(names),prov)
