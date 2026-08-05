import pandas as pd, pytest
from src.targets import build_observed_target_block, TargetValidationError

def df(): return pd.DataFrame({'node_id':['A','A','B','B'],'period':[2020,2021,2020,2021],'esg_score':[1.,2.,3.,4.],'growth':[10.,11.,12.,13.]})

def test_observed_explicit_columns_and_horizon_zero():
    b=build_observed_target_block(df(), target_columns=['esg_score'], forecast_horizon=0, missing_policy='preserve')
    assert b.target_names==('target__observed_esg_score',); assert b.frame.loc[0,'target__observed_esg_score']==1

def test_observed_horizon_one_aligns_next_period_without_feature_mutation():
    features=df()[['node_id','period']].copy(); b=build_observed_target_block(df(), target_columns=['growth'], forecast_horizon=1, missing_policy='preserve')
    assert b.frame.query("node_id=='A' and period==2020")['target__observed_growth'].iloc[0]==11
    assert 'target__observed_growth' not in features.columns

def test_missing_policies_drop_and_error():
    x=df(); x.loc[1,'esg_score']=None
    with pytest.raises(TargetValidationError): build_observed_target_block(x,target_columns=['esg_score'],missing_policy='error')
    b=build_observed_target_block(x,target_columns=['esg_score'],missing_policy='drop')
    assert {'node_id':'A','period':2021} in b.metadata['removed_keys']
    assert len(b.frame)==3

def test_duplicates_malformed_periods_and_unsupported_policy_fail():
    with pytest.raises(TargetValidationError): build_observed_target_block(df(),target_columns=['esg_score'],missing_policy='impute')
    dup=pd.concat([df(),df().iloc[[0]]])
    with pytest.raises(TargetValidationError): build_observed_target_block(dup,target_columns=['esg_score'])
    bad=df().astype({'period': object}); bad.loc[0,'period']=2020.5
    with pytest.raises(TargetValidationError): build_observed_target_block(bad,target_columns=['esg_score'])
