from __future__ import annotations
import pandas as pd
from .contracts import TargetBlock, TargetProvenance, TargetValidationError, validate_keys
POLICIES={'error','preserve','drop'}

def build_observed_target_block(df: pd.DataFrame, *, target_columns, forecast_horizon:int=0, missing_policy='error', source_dataset='observed_long_form') -> TargetBlock:
    if missing_policy not in POLICIES: raise TargetValidationError('unsupported missing target policy')
    if not isinstance(forecast_horizon,int) or forecast_horizon<0: raise TargetValidationError('forecast horizon must be a non-negative integer')
    target_columns=tuple(target_columns)
    if not target_columns: raise TargetValidationError('target columns must be configured explicitly')
    validate_keys(df)
    for c in target_columns:
        if c not in df.columns: raise TargetValidationError(f'missing observed target column {c}')
    obs=df[['node_id','period',*target_columns]].copy()
    if obs.duplicated(['node_id','period']).any(): raise TargetValidationError('duplicate observation keys are not allowed')
    obs['observation_period']=obs['period']
    obs['period']=obs['period']-forecast_horizon
    names=tuple(c if c.startswith('target__') else f'target__observed_{c}' for c in target_columns)
    out=obs.rename(columns=dict(zip(target_columns,names)))
    missing=out[out[list(names)].isna().any(axis=1)][['node_id','period']]
    removed=[]
    if missing_policy=='error' and not missing.empty: raise TargetValidationError('missing observed target values')
    if missing_policy=='drop':
        removed=missing.to_dict(orient='records')
        out=out.dropna(subset=list(names))
    prov=tuple(TargetProvenance(n,'observed',source_dataset,(c,),(),f'observed column {c}; aligned from observation period t+{forecast_horizon}',forecast_horizon=forecast_horizon,missing_value_policy=missing_policy,suitable_as_independent_ground_truth=True,limitations='Observed data quality and construct validity depend on upstream source.',metadata={'observation_period_column':'observation_period'}) for n,c in zip(names,target_columns))
    return TargetBlock(out[['node_id','period',*names]],names,prov,{'removed_keys':removed,'forecast_horizon':forecast_horizon,'observation_period_column':'observation_period'})
