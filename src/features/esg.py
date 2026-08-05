from __future__ import annotations
import pandas as pd
from .contracts import FeatureBlock, FeatureProvenance, FeatureValidationError, sort_feature_frame

def build_esg_feature_block(df, *, aggregation="mean", canonical_node_order=None):
    if aggregation not in {"mean","median"}: raise FeatureValidationError("unsupported ESG aggregation policy")
    pillars=[c for c in ["overall_esg_score","environmental_score","social_score","governance_score","observation_count"] if c in df]
    if not pillars: raise FeatureValidationError("no observed ESG feature columns")
    rows=int(len(df)); unmatched=sorted(df.loc[df["node_id"].isna()| (df["node_id"].astype(str).str.strip()==""),"entity_id"].astype(str).tolist()) if "entity_id" in df else []
    d=df[df["node_id"].notna() & (df["node_id"].astype(str).str.strip()!="")].copy()
    agg=d.groupby(["node_id","period"])[pillars].agg(aggregation).reset_index()
    ren={c:f"esg__{c}" for c in pillars}; agg=agg.rename(columns=ren); names=tuple(ren.values())
    counts=d.groupby(["node_id","period"]).size().reset_index(name="entities"); report={"source_entity_rows":rows,"resulting_node_period_rows":int(len(agg)),"entities_per_node_period":counts.to_dict("records"),"missing_pillar_counts":{c:int(d[c].isna().sum()) for c in pillars},"unmatched_node_ids":unmatched}
    agg.attrs["aggregation_report"]=report; agg=sort_feature_frame(agg, canonical_node_order or sorted(agg.node_id.unique()))
    prov=tuple(FeatureProvenance(n,"esg","normalized_esg_scores",n.replace('esg__',''),aggregation,"annual observation",False,(),"preserve") for n in names)
    return FeatureBlock("esg",agg,names,prov)
