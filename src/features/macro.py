from __future__ import annotations
import numpy as np, pandas as pd
from .contracts import FeatureBlock, FeatureProvenance, FeatureValidationError, sort_feature_frame

def build_macro_feature_block(df:pd.DataFrame, *, metrics:dict[str,list[str]], canonical_node_order=None, zero_previous_policy="missing")->FeatureBlock:
    base=df.copy(); names=[]; out=base[["node_id","period"]].copy(); base=base.sort_values(["node_id","period"])
    for col,transforms in metrics.items():
        if col not in base: raise FeatureValidationError(f"missing macro column {col}")
        if not pd.api.types.is_numeric_dtype(base[col]): raise FeatureValidationError(f"non-numeric macro column {col}")
        aligned=base.set_index(["node_id","period"])[col]
        for tr in transforms:
            nm=f"macro__{col}_{tr}"; vals=aligned.copy()
            if tr=="level": pass
            elif tr=="log1p":
                if (vals.dropna()<0).any(): raise FeatureValidationError("log1p requires non-negative values")
                vals=np.log1p(vals)
            elif tr in {"change","pct_change"}:
                prev=base.groupby("node_id")[col].shift(1); cur=base[col]
                vals=cur-prev if tr=="change" else (cur-prev)/prev.replace(0,np.nan)
                vals.index=base.set_index(["node_id","period"]).index
            else: raise FeatureValidationError(f"unsupported macro transform {tr}")
            out[nm]=out.set_index(["node_id","period"]).index.map(vals.to_dict()).astype(float); names.append(nm)
    out=sort_feature_frame(out, canonical_node_order or sorted(out.node_id.unique()))
    prov=tuple(FeatureProvenance(n,"macro","normalized_macro",n.split('__',1)[1],n.rsplit('_',1)[-1],"annual observation",False,(),"preserve") for n in names)
    return FeatureBlock("macro",out,tuple(names),prov)
