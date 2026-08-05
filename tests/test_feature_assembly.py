import pandas as pd
import pytest
from src.features.assembly import assemble_feature_blocks
from src.features.contracts import FeatureBlock, FeatureProvenance, FeatureValidationError

def prov(feature, block): return FeatureProvenance(feature,block,'s','c','t','annual')
def block(name, node_ids=('A','B'), periods=(2020,2020), feature='b__x'):
    return FeatureBlock(name,pd.DataFrame({'node_id':list(node_ids),'period':list(periods),feature:[1.0,2.0]}),(feature,),(prov(feature,name),))

def test_strict_key_matching_and_non_strict_reports():
    b1=block('b1',feature='b1__x')
    b2=FeatureBlock('b2',pd.DataFrame({'node_id':['A'],'period':[2020],'b2__x':[3.0]}),('b2__x',),(prov('b2__x','b2'),))
    with pytest.raises(FeatureValidationError): assemble_feature_blocks([b1,b2],canonical_node_order=['A','B'],periods=[2020])
    frame,report,_=assemble_feature_blocks([b1,b2],canonical_node_order=['A','B'],periods=[2020],strict=False)
    assert len(frame)==2
    assert pd.isna(frame.loc[frame.node_id=='B','b2__x']).iloc[0]
    assert report.missing_keys_by_block['b2']==[{'node_id':'B','period':2020}]
    assert report.node_count==2 and report.period_count==1

def test_extra_keys_reported_non_strict_and_rejected_strict():
    extra=FeatureBlock('e',pd.DataFrame({'node_id':['A','B','C'],'period':[2020,2020,2020],'e__x':[1,2,3]}),('e__x',),(prov('e__x','e'),))
    with pytest.raises(FeatureValidationError): assemble_feature_blocks([extra],canonical_node_order=['A','B'],periods=[2020])
    _,report,_=assemble_feature_blocks([extra],canonical_node_order=['A','B'],periods=[2020],strict=False)
    assert report.extra_keys_by_block['e']==[{'node_id':'C','period':2020}]

def test_canonical_validation_and_collision():
    b1=block('b1',feature='same__x'); b2=block('b2',feature='same__x')
    with pytest.raises(FeatureValidationError): assemble_feature_blocks([b1],canonical_node_order=['A','A'],periods=[2020])
    with pytest.raises(FeatureValidationError): assemble_feature_blocks([b1],canonical_node_order=[' A'],periods=[2020])
    with pytest.raises(FeatureValidationError): assemble_feature_blocks([b1],canonical_node_order=['A'],periods=[2020,2020])
    with pytest.raises(FeatureValidationError): assemble_feature_blocks([b1,b2],canonical_node_order=['A','B'],periods=[2020],strict=False)

def test_cartesian_expansion_prevented_by_block_contract():
    with pytest.raises(FeatureValidationError): FeatureBlock('dup',pd.DataFrame({'node_id':['A','A'],'period':[2020,2020],'dup__x':[1,2]}),('dup__x',),(prov('dup__x','dup'),))
