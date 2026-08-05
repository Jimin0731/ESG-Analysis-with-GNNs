from __future__ import annotations
import numpy as np, pandas as pd
from .contracts import PreprocessingConfig, FittedPreprocessingState, FeatureValidationError
class TrainOnlyPreprocessor:
    def __init__(self, config:PreprocessingConfig): self.config=config; self.state_=None
    def fit(self, frame, *, feature_names, train_mask, periods):
        names=tuple(feature_names)
        if len(names)!=len(set(names)): raise FeatureValidationError("duplicate feature names")
        X=frame.loc[:,names].to_numpy(float).copy(); m=np.asarray(train_mask,bool)
        if m.shape[0]!=X.shape[0]: raise FeatureValidationError("train mask length mismatch")
        train=X[m]; imp={}; filled=X.copy(); miss={n:int(np.isnan(X[:,i]).sum()) for i,n in enumerate(names)}
        for i,n in enumerate(names):
            col=train[:,i]; ok=col[~np.isnan(col)]
            if self.config.missing_policy=="error" and np.isnan(X[:,i]).any(): raise FeatureValidationError("missing values present")
            if self.config.missing_policy=="preserve": imp[n]=None
            elif self.config.missing_policy=="train_mean":
                if ok.size==0: raise FeatureValidationError(f"all training values missing for {n}")
                imp[n]=float(ok.mean()); filled[np.isnan(filled[:,i]),i]=imp[n]
            elif self.config.missing_policy=="train_median":
                if ok.size==0: raise FeatureValidationError(f"all training values missing for {n}")
                imp[n]=float(np.median(ok)); filled[np.isnan(filled[:,i]),i]=imp[n]
            elif self.config.missing_policy=="constant":
                if self.config.constant_value is None or not np.isfinite(self.config.constant_value): raise FeatureValidationError("finite constant required")
                imp[n]=float(self.config.constant_value); filled[np.isnan(filled[:,i]),i]=imp[n]
            else: raise FeatureValidationError("unsupported missing policy")
        means={}; scales={}; zeros=[]
        for i,n in enumerate(names):
            col=filled[m,i]
            if self.config.scaling_policy=="none": means[n]=0.0; scales[n]=1.0
            elif self.config.scaling_policy=="standard":
                means[n]=float(np.nanmean(col)); sd=float(np.nanstd(col))
                if not np.isfinite(sd) or sd==0.0: sd=1.0; zeros.append(n)
                scales[n]=sd
            else: raise FeatureValidationError("unsupported scaling policy")
        self.state_=FittedPreprocessingState(names,tuple(sorted({int(p) for p in pd.Series(periods)[m]})),self.config.missing_policy,self.config.scaling_policy,imp,means,scales,tuple(zeros),miss)
        return self
    def transform(self, frame):
        if self.state_ is None: raise FeatureValidationError("transform before fit")
        names=self.state_.feature_names
        if list(frame.loc[:, names].columns) != list(names): raise FeatureValidationError("feature schema mismatch")
        X=frame.loc[:,names].to_numpy(float).copy(); missing=np.isnan(X.copy())
        for i,n in enumerate(names):
            v=self.state_.imputation_values[n]
            if v is not None: X[np.isnan(X[:,i]),i]=v
            X[:,i]=(X[:,i]-self.state_.means[n])/self.state_.scales[n]
        if self.state_.missing_policy!="preserve" and not np.isfinite(X).all(): raise FeatureValidationError("processed values must be finite")
        return X, missing
    def fit_transform(self, frame, *, feature_names, train_mask, periods):
        return self.fit(frame,feature_names=feature_names,train_mask=train_mask,periods=periods).transform(frame)
