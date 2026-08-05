from __future__ import annotations
import numpy as np, pandas as pd
from .contracts import TargetBlock, TargetProvenance, TargetValidationError, validate_keys

def _require_enabled(x):
    if not x: raise TargetValidationError('derived targets are disabled unless allow_derived_targets=True')
def _check(df, features):
    validate_keys(df)
    for f in features:
        if f not in df.columns: raise TargetValidationError(f'missing source feature {f}')
        if not pd.api.types.is_numeric_dtype(df[f]): raise TargetValidationError(f'source feature {f} must be numeric')
        if np.isinf(df[f].to_numpy(dtype=float)).any(): raise TargetValidationError(f'source feature {f} must not contain infinity')
def _clip(series,bounds):
    if bounds is None: return series,'none'
    lo,hi=bounds
    if not np.isfinite([lo,hi]).all() or lo>hi: raise TargetValidationError('invalid clipping bounds')
    return series.clip(lo,hi),f'clipped to [{lo}, {hi}]'
def _prov(name, sources, formula, clipdesc):
    return TargetProvenance(target_name=name, target_kind='derived', source_dataset_or_formula=formula, source_column_names=(), source_feature_names=tuple(sources), transformation=formula + ('; '+clipdesc if clipdesc!='none' else ''), suitable_as_independent_ground_truth=False, limitations='Mechanically constructed from model inputs for legacy notebook reproduction; construct validity has not been independently established.')

def build_derived_esg_risk_block(df, *, environmental_feature, social_feature, governance_feature, weights, clip_bounds=(0.0,1.0), allow_derived_targets=False):
    _require_enabled(allow_derived_targets); src=(environmental_feature,social_feature,governance_feature); _check(df,src)
    w=np.array([weights.get(k) for k in ('environmental','social','governance')],dtype=float)
    if not np.isfinite(w).all() or abs(w.sum()-1.0)>1e-9: raise TargetValidationError('ESG risk weights must be finite and sum to one within 1e-9')
    y=w[0]*df[src[0]]+w[1]*df[src[1]]+w[2]*df[src[2]]; y,clip=_clip(y,clip_bounds); name='target__derived_esg_risk'
    return TargetBlock(pd.DataFrame({'node_id':df.node_id,'period':df.period,name:y}), (name,), (_prov(name,src,f'{w[0]}*{src[0]} + {w[1]}*{src[1]} + {w[2]}*{src[2]}',clip),))

def build_derived_economic_impact_block(df, *, environmental_exposure_feature, economic_multiplier_feature, clip_bounds=None, allow_derived_targets=False):
    _require_enabled(allow_derived_targets); src=(environmental_exposure_feature,economic_multiplier_feature); _check(df,src)
    maxv=df[src[1]].max(skipna=True)
    if pd.isna(maxv) or maxv==0: raise TargetValidationError('economic multiplier maximum must be non-zero and non-missing')
    y=df[src[0]]*(df[src[1]]/maxv); y,clip=_clip(y,clip_bounds); name='target__derived_economic_impact'
    return TargetBlock(pd.DataFrame({'node_id':df.node_id,'period':df.period,name:y}), (name,), (_prov(name,src,f'{src[0]} * ({src[1]} / max({src[1]}))',clip),))

def build_derived_volatility_block(df, *, volatility_feature, clip_bounds=None, allow_derived_targets=False):
    _require_enabled(allow_derived_targets); src=(volatility_feature,); _check(df,src); y,clip=_clip(df[src[0]].copy(),clip_bounds); name='target__derived_volatility'
    return TargetBlock(pd.DataFrame({'node_id':df.node_id,'period':df.period,name:y}), (name,), (_prov(name,src,f'{src[0]}',clip),))

def build_derived_transition_cost_block(df, *, carbon_intensity_feature, transition_readiness_feature, clip_bounds=None, allow_derived_targets=False):
    _require_enabled(allow_derived_targets); src=(carbon_intensity_feature,transition_readiness_feature); _check(df,src)
    y=df[src[0]]*(1-df[src[1]]); y,clip=_clip(y,clip_bounds); name='target__derived_transition_cost'
    return TargetBlock(pd.DataFrame({'node_id':df.node_id,'period':df.period,name:y}), (name,), (_prov(name,src,f'{src[0]} * (1 - {src[1]})',clip),))

def build_derived_compliance_probability_block(df, *, governance_score_feature, regulatory_pressure_feature, clip_bounds=(0.0,1.0), allow_derived_targets=False):
    _require_enabled(allow_derived_targets); src=(governance_score_feature,regulatory_pressure_feature); _check(df,src)
    y=df[src[0]]*(1-df[src[1]]); y,clip=_clip(y,clip_bounds); name='target__derived_compliance_probability'
    return TargetBlock(pd.DataFrame({'node_id':df.node_id,'period':df.period,name:y}), (name,), (_prov(name,src,f'{src[0]} * (1 - {src[1]})',clip),))
