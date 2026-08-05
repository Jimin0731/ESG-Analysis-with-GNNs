from __future__ import annotations
import numpy as np, pandas as pd
from src.features.contracts import ChronologicalSplit, FeatureValidationError

def chronological_split(df, *, train_end=None, validation_end=None, ratios=None, period_col="period"):
    periods=sorted({int(p) for p in df[period_col].tolist()})
    if len(periods)<3: raise FeatureValidationError("at least three periods required")
    if ratios is not None:
        if not np.isclose(sum(ratios),1): raise FeatureValidationError("split ratios must sum to 1")
        n=len(periods); ntr=max(1,int(n*ratios[0])); nv=max(1,int(n*ratios[1]));
        if ntr+nv>=n: raise FeatureValidationError("split ratios leave empty test split")
        train=periods[:ntr]; val=periods[ntr:ntr+nv]; test=periods[ntr+nv:]
    else:
        if train_end is None or validation_end is None or train_end>=validation_end: raise FeatureValidationError("explicit boundaries require train_end < validation_end")
        train=[p for p in periods if p<=train_end]; val=[p for p in periods if train_end<p<=validation_end]; test=[p for p in periods if p>validation_end]
    if not train or not val or not test: raise FeatureValidationError("each split must contain at least one period")
    s=df[period_col].astype(int); tm=s.isin(train).to_numpy(); vm=s.isin(val).to_numpy(); xm=s.isin(test).to_numpy()
    if not ((tm.astype(int)+vm.astype(int)+xm.astype(int))==1).all(): raise FeatureValidationError("split masks must cover rows exactly once")
    return ChronologicalSplit(tuple(train),tuple(val),tuple(test),tm,vm,xm,{"train":int(tm.sum()),"validation":int(vm.sum()),"test":int(xm.sum())},{"periods":periods,"mode":"ratios" if ratios else "boundaries"})
