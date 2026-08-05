import pandas as pd, pytest
from src.features.temporal import build_temporal_feature_block
from src.features.contracts import FeatureValidationError
def panel(): return pd.DataFrame({'node_id':['b','a','a','b','a','b'],'period':[2021,2020,2021,2020,2022,2022],'x':[100,1,2,10,3,300]})
def test_temporal_lag_rolling_is_per_node_and_prior_only_future_invariant():
    f=build_temporal_feature_block(panel(),source_features=['x'],lags=(1,),rolling_windows=(2,),min_history=1).frame
    assert f[(f.node_id=='a')&(f.period==2021)]['temporal__x_lag_1'].iloc[0]==1
    assert f[(f.node_id=='b')&(f.period==2021)]['temporal__x_lag_1'].iloc[0]==10
    changed=panel(); changed.loc[changed.period>2021,'x']=99999
    g=build_temporal_feature_block(changed,source_features=['x'],lags=(1,),rolling_windows=(2,),min_history=1).frame
    cols=[c for c in f.columns if c.startswith('temporal__')]
    pd.testing.assert_frame_equal(f[f.period<=2021][cols].reset_index(drop=True),g[g.period<=2021][cols].reset_index(drop=True))
    with pytest.raises(FeatureValidationError): build_temporal_feature_block(panel(),source_features=['x'],rolling_windows=(0,))
