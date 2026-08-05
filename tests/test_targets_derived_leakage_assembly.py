import numpy as np, pandas as pd, pytest
from src.targets import *


def features():
    return pd.DataFrame({
        'node_id':['A','B','A','B'],
        'period':[2020,2020,2021,2021],
        'env':[.2,.4,.3,.8],
        'soc':[.3,.2,.2,.4],
        'gov':[.5,.4,.7,.9],
        'mult':[2.,4.,10.,20.],
        'vol':[-.1,.2,-3.0,.0],
        'carbon':[.6,2.0,.8,.4],
        'ready':[.5,.25,.0,.5],
        'reg':[.2,.1,.5,2.0],
        'safe':[9.,8.,7.,6.],
    })


def all_blocks(df=None):
    f=features() if df is None else df
    return [
        build_derived_esg_risk_block(f,environmental_feature='env',social_feature='soc',governance_feature='gov',weights={'environmental':.2,'social':.3,'governance':.5},allow_derived_targets=True),
        build_derived_economic_impact_block(f,environmental_exposure_feature='env',economic_multiplier_feature='mult',allow_derived_targets=True),
        build_derived_volatility_block(f,volatility_feature='vol',allow_derived_targets=True),
        build_derived_transition_cost_block(f,carbon_intensity_feature='carbon',transition_readiness_feature='ready',allow_derived_targets=True),
        build_derived_compliance_probability_block(f,governance_score_feature='gov',regulatory_pressure_feature='reg',allow_derived_targets=True),
    ]


def test_derived_formulas_named_reorder_deterministic_and_provenance():
    b=all_blocks()[0]; r=features()[list(reversed(features().columns))]
    b2=build_derived_esg_risk_block(r,environmental_feature='env',social_feature='soc',governance_feature='gov',weights={'environmental':.2,'social':.3,'governance':.5},allow_derived_targets=True)
    expected=1.0-(0.2*features()['env']+0.3*features()['soc']+0.5*features()['gov'])
    assert np.allclose(b.frame['target__derived_esg_risk'], expected)
    assert np.allclose(b.frame[list(b.target_names)], b2.frame[list(b2.target_names)])
    assert b.provenance[0].source_feature_names==('env','soc','gov')
    assert b.provenance[0].target_kind=='derived'
    assert '1.0 - (0.2*env + 0.3*soc + 0.5*gov)' in b.provenance[0].source_dataset_or_formula


def test_economic_impact_uses_offset_scale_and_period_specific_normalization():
    f=features()
    b=build_derived_economic_impact_block(f,environmental_exposure_feature='env',economic_multiplier_feature='mult',allow_derived_targets=True)
    expected=(f['env']-0.5)*(f['mult']/f.groupby('period')['mult'].transform('max'))*0.2
    expected=expected.clip(-0.1,0.333)
    assert np.allclose(b.frame['target__derived_economic_impact'], expected)
    assert 'period-wise max(mult)' in b.provenance[0].source_dataset_or_formula
    changed=f.copy(); changed.loc[changed['period']==2021,'mult']*=1000
    changed_block=build_derived_economic_impact_block(changed,environmental_exposure_feature='env',economic_multiplier_feature='mult',allow_derived_targets=True)
    assert np.allclose(b.frame.loc[f['period']==2020,'target__derived_economic_impact'], changed_block.frame.loc[changed['period']==2020,'target__derived_economic_impact'])


def test_volatility_transition_and_compliance_legacy_formulas():
    f=features()
    vol=all_blocks(f)[2]
    assert np.allclose(vol.frame['target__derived_volatility'], (abs(f['vol'])+0.01).clip(0.01,2.0))
    assert 'abs(vol) + 0.01' in vol.provenance[0].source_dataset_or_formula
    transition=all_blocks(f)[3]
    assert np.allclose(transition.frame['target__derived_transition_cost'], (f['carbon']*(1.0-f['ready'])).clip(0.0,1.0))
    assert transition.frame['target__derived_transition_cost'].max()<=1.0
    compliance=all_blocks(f)[4]
    assert np.allclose(compliance.frame['target__derived_compliance_probability'], (f['gov']*(1.0-f['reg']*0.3)).clip(0.0,1.0))
    assert 'gov * (1.0 - reg * 0.3)' in compliance.provenance[0].source_dataset_or_formula


def test_all_legacy_proxy_names_present():
    names=[b.target_names[0] for b in all_blocks()]
    assert names==['target__derived_esg_risk','target__derived_economic_impact','target__derived_volatility','target__derived_transition_cost','target__derived_compliance_probability']


def test_derived_disabled_missing_infinity_and_bad_weights_fail():
    with pytest.raises(TargetValidationError): build_derived_volatility_block(features(),volatility_feature='vol')
    with pytest.raises(TargetValidationError): build_derived_volatility_block(features(),volatility_feature='missing',allow_derived_targets=True)
    inf=features(); inf.loc[0,'vol']=np.inf
    with pytest.raises(TargetValidationError): build_derived_volatility_block(inf,volatility_feature='vol',allow_derived_targets=True)
    with pytest.raises(TargetValidationError): build_derived_esg_risk_block(features(),environmental_feature='env',social_feature='soc',governance_feature='gov',weights={'environmental':1,'social':1,'governance':1},allow_derived_targets=True)
    bad=features(); bad.loc[bad['period']==2021,'mult']=0.0
    with pytest.raises(TargetValidationError): build_derived_economic_impact_block(bad,environmental_exposure_feature='env',economic_multiplier_feature='mult',allow_derived_targets=True)
    with pytest.raises(TargetValidationError): build_derived_compliance_probability_block(features(),governance_score_feature='gov',regulatory_pressure_feature='reg',regulatory_pressure_coefficient=-0.1,allow_derived_targets=True)


def panel(): return assemble_target_blocks(features()[['node_id','period']], all_blocks()[:2], strict=True)


def test_leakage_error_and_drop_union_keep_unrelated():
    p=panel()
    with pytest.raises(TargetValidationError): audit_and_filter_features(['env','mult','safe','target__derived_esg_risk'],p,policy='error')
    safe,removed,report,reasons=audit_and_filter_features(['env','mult','safe','target__derived_esg_risk'],p,policy='drop_declared_sources')
    assert safe==('safe',); assert set(removed)=={'env','mult','target__derived_esg_risk'}; assert 'safe' not in reasons


def test_observed_targets_do_not_remove_unrelated():
    b=build_observed_target_block(pd.DataFrame({'node_id':['A','B','A','B'],'period':[2020,2020,2021,2021],'y':[1.,2.,3.,4.]}),target_columns=['y'],missing_policy='preserve')
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
    assert p.assembly_report.extra_keys_by_block['target__derived_esg_risk']==[{'node_id':'A','period':2021},{'node_id':'B','period':2021}]
    with pytest.raises(TargetValidationError): assemble_target_blocks(features()[['node_id','period']], [b,b], strict=True)
