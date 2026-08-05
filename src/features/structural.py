from __future__ import annotations
import numpy as np, pandas as pd
from src.graphs.contracts import GraphSnapshot
from .contracts import FeatureBlock, FeatureProvenance, FeatureValidationError, sort_feature_frame
FEATURES=("structural__retained_in_degree","structural__retained_out_degree","structural__raw_in_strength","structural__raw_out_strength","structural__model_weight_in_strength","structural__model_weight_out_strength","structural__raw_in_share","structural__raw_out_share")
def build_structural_feature_block(snapshot: GraphSnapshot, *, period:int|None=None, canonical_node_order=None, include_technical_coefficients=True)->FeatureBlock:
    n=len(snapshot.nodes); p=int(period if period is not None else snapshot.period); src=snapshot.edge_index[0]; dst=snapshot.edge_index[1]
    raw=np.asarray(snapshot.raw_flow,float); w=np.asarray(snapshot.edge_weight,float)
    data={"node_id":snapshot.node_ids,"period":[p]*n}
    for name,vals in {
        FEATURES[0]:np.bincount(dst,minlength=n), FEATURES[1]:np.bincount(src,minlength=n), FEATURES[2]:np.bincount(dst,weights=raw,minlength=n), FEATURES[3]:np.bincount(src,weights=raw,minlength=n), FEATURES[4]:np.bincount(dst,weights=w,minlength=n), FEATURES[5]:np.bincount(src,weights=w,minlength=n)}.items(): data[name]=vals.astype(float)
    tin=sum(data[FEATURES[2]]) or np.nan; tout=sum(data[FEATURES[3]]) or np.nan
    data[FEATURES[6]]=data[FEATURES[2]]/tin; data[FEATURES[7]]=data[FEATURES[3]]/tout
    names=list(FEATURES)
    A=snapshot.source_metadata.get("unthresholded_A")
    if include_technical_coefficients and A is not None:
        A=np.asarray(A,float)
        if A.shape!=(n,n): raise FeatureValidationError("unthresholded_A shape must match node count")
        for nm,vals in {"structural__technical_coefficient_row_sum":A.sum(1),"structural__technical_coefficient_column_sum":A.sum(0),"structural__technical_coefficient_diagonal":np.diag(A)}.items(): data[nm]=vals; names.append(nm)
    df=sort_feature_frame(pd.DataFrame(data), canonical_node_order or snapshot.node_ids)
    prov=tuple(FeatureProvenance(n,"structural",snapshot.backend,n.replace('structural__',''),"graph_statistic","snapshot annual",False,(),"preserve",None,"raw flows are distinct from model edge weights") for n in names)
    return FeatureBlock("structural",df,tuple(names),prov)
