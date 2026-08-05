import warnings
import numpy as np
import pandas as pd
import pytest
from src.features.contracts import FeatureValidationError, PreprocessingConfig
from src.features.preprocessing import TrainOnlyPreprocessor

def frame(out=1000.0):
    return pd.DataFrame({'prep__f1':[1.0,3.0,out,out], 'prep__f2':[5.0,5.0,5.0,5.0], 'prep__f3':[np.nan,2.0,9.0,9.0]})
def periods(): return pd.Series([2020,2021,2022,2023])
def mask(): return np.array([True,True,False,False])

def fit_state(out=1000.0, policy='train_mean'):
    p=TrainOnlyPreprocessor(PreprocessingConfig(policy,'standard')).fit(frame(out),feature_names=tuple(frame().columns),train_mask=mask(),periods=periods())
    return p

def test_holdout_changes_do_not_change_fitted_state():
    state1=fit_state(1000).state_.to_json_dict()
    state2=fit_state(999999).state_.to_json_dict()
    assert state1['imputation_values']==state2['imputation_values']
    assert state1['means']==state2['means']
    assert state1['scales']==state2['scales']
    assert state1['zero_variance_features']==state2['zero_variance_features']
    assert state1['fit_periods']==state2['fit_periods']

def test_zero_variance_and_finite_transform():
    p=fit_state()
    X,missing=p.transform(frame()[list(frame().columns)])
    assert np.isfinite(X).all()
    assert 'prep__f2' in p.state_.zero_variance_features
    assert missing[0,2]

def test_train_median_uses_no_holdout():
    p=TrainOnlyPreprocessor(PreprocessingConfig('train_median','none')).fit(frame(),feature_names=tuple(frame().columns),train_mask=mask(),periods=periods())
    assert p.state_.imputation_values['prep__f3']==2.0

def test_all_training_missing_and_constant_policy():
    df=pd.DataFrame({'prep__a':[np.nan,np.nan,1.0]}); m=np.array([True,True,False]); per=pd.Series([2020,2021,2022])
    with pytest.raises(FeatureValidationError): TrainOnlyPreprocessor(PreprocessingConfig('train_mean','none')).fit(df,feature_names=('prep__a',),train_mask=m,periods=per)
    with pytest.raises(FeatureValidationError): PreprocessingConfig('constant','none')
    p=TrainOnlyPreprocessor(PreprocessingConfig('constant','none',0.0)).fit(df,feature_names=('prep__a',),train_mask=m,periods=per)
    assert p.transform(df)[0][0,0] == 0.0

def test_schema_and_public_exception_normalization():
    p=fit_state()
    with pytest.raises(FeatureValidationError): TrainOnlyPreprocessor(PreprocessingConfig()).transform(frame())
    with pytest.raises(FeatureValidationError): p.transform(frame().rename(columns={'prep__f1':'prep__renamed'}))
    with pytest.raises(FeatureValidationError): p.transform(frame()[['prep__f2','prep__f1','prep__f3']])
    with pytest.raises(FeatureValidationError): p.transform(frame().assign(prep__extra=1.0))

def test_fit_input_validation():
    with pytest.raises(FeatureValidationError): fit_state().fit(frame(),feature_names=(),train_mask=mask(),periods=periods())
    with pytest.raises(FeatureValidationError): fit_state().fit(frame(),feature_names=('prep__f1','prep__f1'),train_mask=mask(),periods=periods())
    with pytest.raises(FeatureValidationError): fit_state().fit(frame(),feature_names=('prep__missing',),train_mask=mask(),periods=periods())
    bad=frame(); bad.loc[0,'prep__f1']=np.inf
    with pytest.raises(FeatureValidationError): fit_state().fit(bad,feature_names=tuple(frame().columns),train_mask=mask(),periods=periods())
    with pytest.raises(FeatureValidationError): fit_state().fit(frame(),feature_names=tuple(frame().columns),train_mask=np.array([[True]]),periods=periods())
    with pytest.raises(FeatureValidationError): fit_state().fit(frame(),feature_names=tuple(frame().columns),train_mask=np.array([False]*4),periods=periods())
    with pytest.raises(FeatureValidationError): fit_state().fit(frame(),feature_names=tuple(frame().columns),train_mask=mask(),periods=pd.Series([2020,2021]))
    with pytest.raises(FeatureValidationError): fit_state().fit(frame(),feature_names=tuple(frame().columns),train_mask=mask(),periods=pd.Series([2020,2021,2022.5,2023]))

def test_preserve_standard_all_missing_rejected_without_warnings():
    df=pd.DataFrame({'prep__a':[np.nan,np.nan,1.0]}); m=np.array([True,True,False]); per=pd.Series([2020,2021,2022])
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with pytest.raises(FeatureValidationError): TrainOnlyPreprocessor(PreprocessingConfig('preserve','standard')).fit(df,feature_names=('prep__a',),train_mask=m,periods=per)
