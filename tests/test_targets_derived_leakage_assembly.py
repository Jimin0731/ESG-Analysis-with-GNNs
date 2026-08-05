import numpy as np, pandas as pd, pytest
from src.targets import *

def features():
    return pd.DataFrame({'node_id':['A','B'],'period':[2020,2020],'env':[.2,.4],'soc':[.3,.2],'gov':[.5,.4],'mult':[2.,4.],'vol':[.1,.2],'carbon':[.6,.8],'ready':[.5,.25],'reg':[.2,.1],'safe':[9.,8.]})

def all_blocks(df=None):
    f=features() if df is None else df
    return [build_derived_esg_risk_block(f,environmental_feature='env',social_feature='soc',governance_feature='gov',weights={'environmental':.2,'social':.3,'governance':.5},allow_derived_targets=True), build_derived_economic_impact_block(f,environmental_exposure_feature='env',economic_multiplier_feature='mult',allow_derived_targets=True), build_derived_volatility_block(f,volatility_feature='vol',allow_derived_targets=True), build_derived_transition_cost_block(f,carbon_intensity_feature='carbon',transition_readiness_feature='ready',allow_derived_targets=True), build_derived_compliance_probability_block(f,governance_score_feature='gov',regulatory_pressure_feature='reg',allow_derived_targets=True)]

def test_derived_formulas_named_reorder_deterministic_and_provenance():
    b=all_blocks()[0]; r=features()[list(reversed(features().columns))]
    b2=build_derived_esg_risk_block(r,environmental_feature='env',social_feature='soc',governance_feature='gov',weights={'environmental':.2,'social':.3,'governance':.5},allow_derived_targets=True)
    assert np.allclose(b.frame[list(b.target_names)], b2.frame[list(b2.target_names)])
    assert b.provenance[0].source_feature_names==('env','soc','gov')
    assert b.provenance[0].target_kind=='derived'
    assert all_blocks()[1].frame['target__derived_economic_impact'].iloc[1]==.4

def test_all_legacy_proxy_names_present():
    names=[b.target_names[0] for b in all_blocks()]
    assert names==['target__derived_esg_risk','target__derived_economic_impact','target__derived_volatility','target__derived_transition_cost','target__derived_compliance_probability']

def test_derived_disabled_missing_infinity_and_bad_weights_fail():
    with pytest.raises(TargetValidationError): build_derived_volatility_block(features(),volatility_feature='vol')
    with pytest.raises(TargetValidationError): build_derived_volatility_block(features(),volatility_feature='missing',allow_derived_targets=True)
    inf=features(); inf.loc[0,'vol']=np.inf
    with pytest.raises(TargetValidationError): build_derived_volatility_block(inf,volatility_feature='vol',allow_derived_targets=True)
    with pytest.raises(TargetValidationError): build_derived_esg_risk_block(features(),environmental_feature='env',social_feature='soc',governance_feature='gov',weights={'environmental':1,'social':1,'governance':1},allow_derived_targets=True)

def panel(): return assemble_target_blocks(features()[['node_id','period']], all_blocks()[:2], strict=True)

def test_leakage_error_and_drop_union_keep_unrelated():
    p=panel()
    with pytest.raises(TargetValidationError): audit_and_filter_features(['env','mult','safe','target__derived_esg_risk'],p,policy='error')
    safe,removed,report,reasons=audit_and_filter_features(['env','mult','safe','target__derived_esg_risk'],p,policy='drop_declared_sources')
    assert safe==('safe',); assert set(removed)=={'env','mult','target__derived_esg_risk'}; assert 'safe' not in reasons

def test_observed_targets_do_not_remove_unrelated():
    b=build_observed_target_block(pd.DataFrame({'node_id':['A','B'],'period':[2020,2020],'y':[1.,2.]}),target_columns=['y'],missing_policy='preserve')
    p=assemble_target_blocks(features()[['node_id','period']], [b])
    assert audit_and_filter_features(['safe'],p,policy='drop_declared_sources')[0]==('safe',)

def test_key_misalignment_and_horizon_violation_detected():
    p=panel(); fk=pd.DataFrame({'node_id':['A'],'period':[2020]})
    assert any(f.finding_type=='target_key_misalignment' for f in audit_direct_leakage(['safe'],p,feature_keys=fk).findings)
    b=build_observed_target_block(pd.DataFrame({'node_id':['A'],'period':[2021],'y':[1.]}),target_columns=['y'],forecast_horizon=1,missing_policy='preserve')
    bad=TargetPanel(b.frame.assign(observation_period=2025),b.target_names,b.provenance)
    assert any(f.finding_type=='forecast_horizon_violation' for f in audit_direct_leakage(['safe'],bad).findings)

def test_assembly_strict_nonstrict_collision_and_order_no_cartesian():
    fk=pd.DataFrame({'node_id':['B','A'],'period':[2020,2020]}); b=all_blocks()[0]
    with pytest.raises(TargetValidationError): assemble_target_blocks(pd.DataFrame({'node_id':['A'],'period':[2020]}), [b], strict=True)
    p=assemble_target_blocks(fk,[b],strict=False)
    assert list(p.frame.node_id)==['B','A']; assert len(p.frame)==2
    assert p.assembly_report.extra_keys_by_block['target__derived_esg_risk']==[]
    with pytest.raises(TargetValidationError): assemble_target_blocks(features()[['node_id','period']], [b,b], strict=True)
