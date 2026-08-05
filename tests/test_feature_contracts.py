import json, numpy as np, pandas as pd, pytest
from src.features.contracts import *
def test_contract_rejects_bad_and_sorts():
    with pytest.raises(FeatureValidationError): FeatureBlock('x',pd.DataFrame({'node_id':['a','a'],'period':[2020,2020],'x__a':[1,2]}),('x__a',),())
    with pytest.raises(FeatureValidationError): FeatureBlock('x',pd.DataFrame({'node_id':[' '],'period':[2020],'x__a':[1]}),('x__a',),())
    with pytest.raises(FeatureValidationError): FeatureBlock('x',pd.DataFrame({'node_id':['a'],'period':[2020.5],'x__a':[1]}),('x__a',),())
    with pytest.raises(FeatureValidationError): FeatureBlock('x',pd.DataFrame({'node_id':['a'],'period':[2020],'x__a':[np.inf]}),('x__a',),())
    p=FeatureProvenance('x__a','x','s','c','t','annual'); json.dumps(p.to_json_dict())
