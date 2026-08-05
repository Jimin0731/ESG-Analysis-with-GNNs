import json
import numpy as np
import pandas as pd
import pytest
from src.features.contracts import *

def prov(name='x__a'):
    return FeatureProvenance(name,'x','source','column','level','annual')

def test_feature_key_period_validation():
    FeatureKey('A', 2020)
    for bad in ['', ' A', 'A ', '   ']:
        with pytest.raises(FeatureValidationError): FeatureKey(bad, 2020)
    for bad in [True, 2020.5, '2020', None, np.inf]:
        with pytest.raises(FeatureValidationError): FeatureKey('A', bad)

def test_feature_block_rejects_duplicate_keys_and_bad_periods():
    frame = pd.DataFrame({'node_id':['A','A'], 'period':[2020,2020], 'x__a':[1.0,2.0]})
    with pytest.raises(FeatureValidationError): FeatureBlock('x', frame, ('x__a',), (prov(),))
    for bad in [2020.5, '2020', True, None]:
        frame = pd.DataFrame({'node_id':['A'], 'period':[bad], 'x__a':[1.0]})
        with pytest.raises(FeatureValidationError): FeatureBlock('x', frame, ('x__a',), (prov(),))

def test_feature_block_provenance_and_feature_names():
    frame = pd.DataFrame({'node_id':['A'], 'period':[2020], 'x__a':[1.0]})
    with pytest.raises(FeatureValidationError): FeatureBlock('', frame, ('x__a',), (prov(),))
    with pytest.raises(FeatureValidationError): FeatureBlock('x', frame, ('a',), (FeatureProvenance('x__a','x','s','c','t','annual'),))
    with pytest.raises(FeatureValidationError): FeatureBlock('x', frame, ('x__a',), ())
    with pytest.raises(FeatureValidationError): FeatureBlock('x', frame, ('x__a',), (prov('x__b'),))
    assert json.dumps(FeatureBlock('x', frame, ('x__a',), (prov(),), {'rows':1}).report)

def test_chronological_split_contract_masks():
    mask = np.array([True, False, False])
    ChronologicalSplit((2020,), (2021,), (2022,), mask, ~mask & np.array([False, True, False]), np.array([False,False,True]), {'train':1,'validation':1,'test':1}, {'mode':'x'})
    with pytest.raises(FeatureValidationError): ChronologicalSplit((2021,), (2020,), (2022,), mask, mask, mask, {}, {})
    with pytest.raises(FeatureValidationError): ChronologicalSplit((2020,), (2021,), (2022,), np.array([[True]]), mask, mask, {}, {})

def test_fitted_state_json_and_finite_stats():
    state = FittedPreprocessingState(('x__a',), (2020,), 'train_mean', 'standard', {'x__a':1.0}, {'x__a':1.0}, {'x__a':1.0}, ())
    json.dumps(state.to_json_dict())
    with pytest.raises(FeatureValidationError): FittedPreprocessingState(('x__a',), (2020,), 'm', 's', {'x__a':np.inf}, {'x__a':1}, {'x__a':1}, ())

def test_feature_panel_validation():
    report=FeatureAssemblyReport(2,1,{'x':['x__a']},{'x__a':0},{},{},['A'],[2020],1,1,['A'],[2020])
    state=FittedPreprocessingState(('x__a',),(2020,),'error','none',{'x__a':None},{'x__a':0.0},{'x__a':1.0},())
    FeaturePanel(('A',),(2020,),np.ones((2,1)),np.ones((2,1)),('x__a',),np.zeros((2,1),bool),np.array([1,0],bool),np.array([0,1],bool),np.array([0,0],bool),(prov(),),report,state)
    with pytest.raises(FeatureValidationError): FeaturePanel(('A',),(2020,),np.ones((2,1)),np.ones((2,1)),('x__a',),np.zeros((2,1),bool),np.array([1,0],bool),np.array([1,0],bool),np.array([0,0],bool),(prov(),),report,state)
