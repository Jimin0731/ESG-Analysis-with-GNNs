import pandas as pd, pytest
from src.splits import chronological_split
from src.features.contracts import FeatureValidationError
def df(): return pd.DataFrame({'node_id':['a','b']*6,'period':sum(([y,y] for y in [2022,2020,2021,2023,2024,2025]),[])})
def test_boundaries_ratios_coverage_deterministic():
    s=chronological_split(df(),train_end=2021,validation_end=2023)
    assert s.train_periods==(2020,2021) and s.validation_periods==(2022,2023) and s.test_periods==(2024,2025)
    assert ((s.train_mask.astype(int)+s.validation_mask.astype(int)+s.test_mask.astype(int))==1).all()
    r=chronological_split(df(),ratios=(.5,.25,.25)); assert r.train_periods==(2020,2021,2022)
    with pytest.raises(FeatureValidationError): chronological_split(pd.DataFrame({'period':[2020,2021]}),ratios=(.6,.2,.2))
