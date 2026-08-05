from __future__ import annotations
import numpy as np, pandas as pd
from .contracts import FeatureBlock, FeatureProvenance, FeatureValidationError, sort_feature_frame

def build_temporal_feature_block(df, *, source_features, lags=(1,), rolling_windows=(), min_history=1, include_current=False, changes=True, slopes=False, canonical_node_order=None):
    if any(int(w)<=0 for w in rolling_windows): raise FeatureValidationError("rolling windows must be positive")
    if any(int(l)<=0 for l in lags): raise FeatureValidationError("lags must be positive")
    d=df.sort_values(["node_id","period"]).copy(); out=d[["node_id","period"]].copy(); names=[]
    for f in source_features:
        g=d.groupby("node_id", sort=False)[f]
        for lag in lags:
            nm=f"temporal__{f}_lag_{lag}"; out[nm]=g.shift(lag); names.append(nm)
        if changes:
            nm=f"temporal__{f}_change_1"; out[nm]=g.diff(); names.append(nm)
        hist=g if include_current else g.shift(1).groupby(d["node_id"], sort=False)
        for w in rolling_windows:
            r=hist.rolling(w, min_periods=min_history)
            for stat,series in {"rolling_mean":r.mean().reset_index(level=0,drop=True),"rolling_std":r.std(ddof=0).reset_index(level=0,drop=True)}.items():
                nm=f"temporal__{f}_{stat}_{w}"; out[nm]=series; names.append(nm)
            if slopes:
                nm=f"temporal__{f}_slope_{w}"
                def slope(x):
                    if len(x)<min_history: return np.nan
                    return float(np.polyfit(np.arange(len(x)), x, 1)[0])
                out[nm]=hist.rolling(w,min_periods=min_history).apply(slope, raw=True).reset_index(level=0,drop=True); names.append(nm)
    out=sort_feature_frame(out, canonical_node_order or sorted(out.node_id.unique()))
    prov=tuple(FeatureProvenance(n,"temporal","assembled_features",n,"lag/rolling/change","uses only prior periods by default",False,(),"preserve") for n in names)
    return FeatureBlock("temporal",out,tuple(names),prov)
