import pandas as pd, pytest
from src.features.contracts import FeatureBlock, FeatureProvenance, FeatureValidationError
from src.features.assembly import assemble_feature_blocks
def block(name, rows, feat):
    return FeatureBlock(name,pd.DataFrame(rows),tuple(feat),tuple(FeatureProvenance(f,name,'s','c','t','annual') for f in feat))
def test_strict_non_strict_and_collision():
    b1=block('b1',{'node_id':['a','b'],'period':[2020,2020],'b1__x':[1,2]},['b1__x'])
    b2=block('b2',{'node_id':['a'],'period':[2020],'b2__x':[3]},['b2__x'])
    with pytest.raises(FeatureValidationError): assemble_feature_blocks([b1,b2],canonical_node_order=['a','b'],periods=[2020])
    f,r,_=assemble_feature_blocks([b1,b2],canonical_node_order=['a','b'],periods=[2020],strict=False)
    assert pd.isna(f.loc[f.node_id=='b','b2__x']).iloc[0]
    assert r.missing_keys_by_block['b2']==[{'node_id':'b','period':2020}]
    b3=block('b3',{'node_id':['a','b'],'period':[2020,2020],'b1__x':[1,2]},['b1__x'])
    with pytest.raises(FeatureValidationError): assemble_feature_blocks([b1,b3],canonical_node_order=['a','b'],periods=[2020],strict=False)
