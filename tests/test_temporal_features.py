import pandas as pd
import pytest
from src.features.temporal import build_temporal_feature_block
from src.features.contracts import FeatureValidationError

def panel():
    return pd.DataFrame({'node_id':['B','A','A','B','A','B','A','B'], 'period':[2021,2020,2021,2020,2022,2022,2023,2023], 'x':[20.,1.,3.,10.,6.,40.,10.,80.]})

def get(df,node,period,col): return df[(df.node_id==node)&(df.period==period)][col].iloc[0]

def test_prior_only_lag_change_rolling_unsorted_interleaved():
    f=build_temporal_feature_block(panel(),source_features=['x'],lags=(1,2),rolling_windows=(2,),min_history=1).frame
    assert get(f,'A',2022,'temporal__x_lag_1') == 3
    assert get(f,'A',2022,'temporal__x_lag_2') == 1
    assert get(f,'A',2022,'temporal__x_change_1') == 2
    assert get(f,'B',2022,'temporal__x_lag_1') == 20
    assert get(f,'B',2022,'temporal__x_rolling_mean_2') == 15

def test_current_period_leakage_default():
    base=panel(); f=build_temporal_feature_block(base,source_features=['x'],lags=(1,),rolling_windows=(2,),min_history=1).frame
    changed=base.copy(); changed.loc[(changed.node_id=='A')&(changed.period==2022),'x']=9999
    g=build_temporal_feature_block(changed,source_features=['x'],lags=(1,),rolling_windows=(2,),min_history=1).frame
    cols=[c for c in f.columns if c.startswith('temporal__')]
    pd.testing.assert_series_equal(f[(f.node_id=='A')&(f.period==2022)][cols].iloc[0], g[(g.node_id=='A')&(g.period==2022)][cols].iloc[0], check_names=False)

def test_future_invariance():
    f=build_temporal_feature_block(panel(),source_features=['x'],lags=(1,),rolling_windows=(2,),min_history=1,slopes=True).frame
    changed=panel(); changed.loc[changed.period>2021,'x']=99999
    g=build_temporal_feature_block(changed,source_features=['x'],lags=(1,),rolling_windows=(2,),min_history=1,slopes=True).frame
    cols=[c for c in f.columns if c.startswith('temporal__')]
    pd.testing.assert_frame_equal(f[f.period<=2021][cols].reset_index(drop=True),g[g.period<=2021][cols].reset_index(drop=True))

def test_insufficient_history_and_include_current_difference():
    f=build_temporal_feature_block(panel(),source_features=['x'],lags=(1,),rolling_windows=(3,),min_history=3).frame
    assert pd.isna(get(f,'A',2022,'temporal__x_rolling_mean_3'))
    prior=build_temporal_feature_block(panel(),source_features=['x'],lags=(1,),rolling_windows=(),include_current=False).frame
    cur=build_temporal_feature_block(panel(),source_features=['x'],lags=(1,),rolling_windows=(),include_current=True).frame
    assert get(prior,'A',2021,'temporal__x_lag_1') == 1
    assert get(cur,'A',2021,'temporal__x_lag_1') == 3
    assert pd.isna(get(prior,'A',2021,'temporal__x_change_1'))
    assert get(cur,'A',2021,'temporal__x_change_1') == 2

def test_temporal_invalid_configuration():
    with pytest.raises(FeatureValidationError): build_temporal_feature_block(panel(),source_features=['x'],lags=(0,))
    with pytest.raises(FeatureValidationError): build_temporal_feature_block(panel(),source_features=['x'],lags=(-1,))
    with pytest.raises(FeatureValidationError): build_temporal_feature_block(panel(),source_features=['x'],rolling_windows=(0,))
    with pytest.raises(FeatureValidationError): build_temporal_feature_block(panel(),source_features=['x'],rolling_windows=(-2,))
    bad=panel(); bad['period']=bad['period'].astype(object); bad.loc[0,'period']=2020.5
    with pytest.raises(FeatureValidationError): build_temporal_feature_block(bad,source_features=['x'])
