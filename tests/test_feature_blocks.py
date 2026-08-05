import json
import numpy as np
import pandas as pd
import pytest
from src.graphs.contracts import GraphNode, GraphSnapshot
from src.features.contracts import FeatureValidationError
from src.features.environmental import build_environmental_feature_block
from src.features.esg import build_esg_feature_block
from src.features.macro import build_macro_feature_block
from src.features.structural import build_structural_feature_block

def snap(period=2020, reverse=False, metadata=None):
    edge=np.array([[0],[1]]) if not reverse else np.array([[1],[0]])
    return GraphSnapshot((GraphNode('A','A'),GraphNode('B','B')),edge,np.array([0.5]),np.array([10.0]),'backend',period,metadata or {'unthresholded_A':np.eye(2)})

def test_structural_direction_and_raw_weight_separation():
    f=build_structural_feature_block(snap()).frame
    assert f.loc[f.node_id=='A','structural__raw_out_strength'].iloc[0] == 10
    assert f.loc[f.node_id=='A','structural__model_weight_out_strength'].iloc[0] == 0.5
    r=build_structural_feature_block(snap(reverse=True)).frame
    assert r.loc[r.node_id=='A','structural__raw_in_strength'].iloc[0] == 10

def test_structural_period_and_matrix_validation():
    for bad in [None, True, 2020.5, '2020']:
        with pytest.raises(FeatureValidationError): build_structural_feature_block(snap(period=bad))
    with pytest.raises(FeatureValidationError): build_structural_feature_block(snap(metadata={'unthresholded_A':np.ones((2,3))}))
    with pytest.raises(FeatureValidationError): build_structural_feature_block(snap(metadata={'unthresholded_A':np.array([[1,np.nan],[0,1]])}))
    with pytest.raises(FeatureValidationError): build_structural_feature_block(snap(metadata={'unthresholded_A':[['x']]}))

def macro_df(): return pd.DataFrame({'node_id':['A','A','A','B','B'],'period':[2020,2021,2022,2020,2021],'real_output':[0.0,10.0,np.nan,np.nan,5.0]})

def test_macro_zero_previous_policies():
    missing=build_macro_feature_block(macro_df(),metrics={'real_output':['pct_change']},zero_previous_policy='missing').frame
    assert pd.isna(missing.loc[(missing.node_id=='A')&(missing.period==2021),'macro__real_output_pct_change'].iloc[0])
    zero=build_macro_feature_block(macro_df(),metrics={'real_output':['pct_change']},zero_previous_policy='zero').frame
    assert zero.loc[(zero.node_id=='A')&(zero.period==2021),'macro__real_output_pct_change'].iloc[0] == 0
    assert pd.isna(zero.loc[(zero.node_id=='B')&(zero.period==2021),'macro__real_output_pct_change'].iloc[0])
    with pytest.raises(FeatureValidationError): build_macro_feature_block(macro_df(),metrics={'real_output':['pct_change']},zero_previous_policy='error')
    with pytest.raises(FeatureValidationError): build_macro_feature_block(macro_df(),metrics={'real_output':['pct_change']},zero_previous_policy='bad')

def test_macro_log_and_period_validation():
    with pytest.raises(FeatureValidationError): build_macro_feature_block(pd.DataFrame({'node_id':['A'],'period':[2020],'x':[-1.0]}),metrics={'x':['log1p']})
    with pytest.raises(FeatureValidationError): build_macro_feature_block(pd.DataFrame({'node_id':['A'],'period':[2020.5],'x':[1.0]}),metrics={'x':['level']})

def env_df(): return pd.DataFrame({'node_id':['A','B','C'],'period':[2020,2020,2020],'emissions':[10.0,20.0,np.inf],'output':[2.0,0.0,3.0],'energy':[5.0,-1.0,2.0]})

def test_environmental_denominator_policies_and_validation():
    good=env_df().iloc[:2].copy()
    with pytest.raises(FeatureValidationError): build_environmental_feature_block(good,intensities={'emissions_intensity':{'numerator':'emissions','denominator':'output'}},invalid_denominator_policy='ignore')
    with pytest.raises(FeatureValidationError): build_environmental_feature_block(good,intensities={'emissions_intensity':{'numerator':'emissions','denominator':'output'}})
    miss=build_environmental_feature_block(good,intensities={'emissions_intensity':{'numerator':'emissions','denominator':'output'}},invalid_denominator_policy='missing').frame
    assert pd.isna(miss.loc[miss.node_id=='B','environmental__emissions_intensity'].iloc[0])
    with pytest.raises(FeatureValidationError): build_environmental_feature_block(good,intensities={'bad':{'numerator':'emissions','denominator':'emissions'}})
    with pytest.raises(FeatureValidationError): build_environmental_feature_block(env_df(),levels=['emissions'])
    with pytest.raises(FeatureValidationError): build_environmental_feature_block(good,log1p=['energy'])

def esg_df(): return pd.DataFrame({'entity_id':['E1','E2','E3','E3','E4'],'node_id':['A','A','B','B',None],'period':[2020,2020,2020,2020,2020],'overall_esg_score':[60,80,50,70,99],'environmental_score':[55,75,np.nan,65,99],'governance_score':[50,70,50,70,99]})

def test_esg_mean_median_typed_report_survives_copy():
    mean=build_esg_feature_block(esg_df(),aggregation='mean')
    med=build_esg_feature_block(esg_df(),aggregation='median')
    assert mean.frame.loc[mean.frame.node_id=='A','esg__overall_esg_score'].iloc[0] == 70
    assert med.frame.loc[med.frame.node_id=='B','esg__overall_esg_score'].iloc[0] == 60
    assert mean.report['source_entity_rows'] == 5
    assert mean.report['aggregation_policy'] == 'mean'
    assert mean.report['missing_counts_by_pillar']['environmental_score'] == 1
    assert mean.report['unmatched_or_missing_node_ids'] == ['E4']
    copied=mean.frame.copy(); assert len(copied)==mean.report['resulting_node_period_rows']
    json.dumps(mean.report)
    assert 'target' not in ' '.join(mean.feature_names).lower()

def test_esg_unsupported_and_bad_period():
    with pytest.raises(FeatureValidationError): build_esg_feature_block(esg_df(),aggregation='mode')
    bad=esg_df(); bad['period']=bad['period'].astype(object); bad.loc[0,'period']='2020'
    with pytest.raises(FeatureValidationError): build_esg_feature_block(bad)
