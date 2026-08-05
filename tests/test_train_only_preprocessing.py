import numpy as np, pandas as pd, pytest
from src.features.preprocessing import TrainOnlyPreprocessor
from src.features.contracts import PreprocessingConfig, FeatureValidationError
def frame(out=1000): return pd.DataFrame({'period':[2020,2021,2022,2023],'f1':[1.,3.,out,out],'f2':[5.,5.,5.,5.],'f3':[np.nan,2.,9.,9.]})
def test_train_only_outlier_invariance_and_zero_variance():
    mask=np.array([1,1,0,0],bool)
    p=TrainOnlyPreprocessor(PreprocessingConfig('train_mean','standard')).fit(frame(),feature_names=('f1','f2','f3'),train_mask=mask,periods=frame().period)
    state=p.state_.to_json_dict(); X,miss=p.transform(frame())
    p2=TrainOnlyPreprocessor(PreprocessingConfig('train_mean','standard')).fit(frame(999999),feature_names=('f1','f2','f3'),train_mask=mask,periods=frame().period)
    assert p2.state_.to_json_dict()==state
    assert np.isfinite(X).all() and 'f2' in p.state_.zero_variance_features and miss[0,2]
def test_missing_and_schema_errors():
    mask=np.array([1,1,0],bool); df=pd.DataFrame({'period':[2020,2021,2022],'a':[np.nan,np.nan,1.]})
    with pytest.raises(FeatureValidationError): TrainOnlyPreprocessor(PreprocessingConfig('train_median','none')).fit(df,feature_names=('a',),train_mask=mask,periods=df.period)
    with pytest.raises(FeatureValidationError): TrainOnlyPreprocessor(PreprocessingConfig('constant','none')).fit(df,feature_names=('a',),train_mask=mask,periods=df.period)
    p=TrainOnlyPreprocessor(PreprocessingConfig('constant','none',0)).fit(df,feature_names=('a',),train_mask=mask,periods=df.period)
    with pytest.raises(FeatureValidationError): TrainOnlyPreprocessor(PreprocessingConfig()).transform(df)
    with pytest.raises(KeyError): p.transform(pd.DataFrame({'b':[1]}))
