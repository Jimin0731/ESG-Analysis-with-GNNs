import numpy as np, pandas as pd, pytest
from src.graphs.contracts import GraphSnapshot, GraphNode
from src.features.structural import build_structural_feature_block
from src.features.macro import build_macro_feature_block
from src.features.environmental import build_environmental_feature_block
from src.features.esg import build_esg_feature_block
from src.features.contracts import FeatureValidationError
def snap(rev=False):
    e=np.array([[0],[1]]) if not rev else np.array([[1],[0]])
    return GraphSnapshot((GraphNode('a','A'),GraphNode('b','B')),e,np.array([.5]),np.array([10.]),'test',2020,{'unthresholded_A':np.eye(2)})
def test_structural_direction_and_a_shape():
    f=build_structural_feature_block(snap()).frame
    assert f.loc[f.node_id=='a','structural__raw_out_strength'].iloc[0]==10
    r=build_structural_feature_block(snap(True)).frame
    assert r.loc[r.node_id=='a','structural__raw_in_strength'].iloc[0]==10
    bad=snap(); bad.source_metadata['unthresholded_A']=np.ones((3,3))
    with pytest.raises(FeatureValidationError): build_structural_feature_block(bad)
def test_macro_environmental_esg():
    df=pd.DataFrame({'node_id':['a','a','b','b'],'period':[2020,2021,2020,2021],'real_output':[0,10,5,15]})
    m=build_macro_feature_block(df,metrics={'real_output':['level','change','pct_change']}).frame
    assert pd.isna(m.loc[(m.node_id=='a')&(m.period==2020),'macro__real_output_change'].iloc[0])
    assert pd.isna(m.loc[(m.node_id=='a')&(m.period==2021),'macro__real_output_pct_change'].iloc[0])
    with pytest.raises(FeatureValidationError): build_macro_feature_block(pd.DataFrame({'node_id':['a'],'period':[2020],'x':[-1]}),metrics={'x':['log1p']})
    env=pd.DataFrame({'node_id':['a','b'],'period':[2020,2020],'emissions':[10,20],'output':[2,0]})
    with pytest.raises(FeatureValidationError): build_environmental_feature_block(env,intensities={'emissions_intensity':{'numerator':'emissions','denominator':'output'}})
    e=build_environmental_feature_block(env,intensities={'emissions_intensity':{'numerator':'emissions','denominator':'output'}},invalid_denominator_policy='missing').frame
    assert pd.isna(e.loc[e.node_id=='b','environmental__emissions_intensity'].iloc[0])
    esg=pd.DataFrame({'entity_id':['x','y','z'],'node_id':['a','a',None],'period':[2020,2020,2020],'overall_esg_score':[1,3,9],'governance_score':[2,4,None]})
    b=build_esg_feature_block(esg,aggregation='median')
    assert b.frame['esg__overall_esg_score'].iloc[0]==2
    assert 'target' not in ' '.join(b.feature_names).lower()
    assert b.frame.attrs['aggregation_report']['unmatched_node_ids']==['z']
