import numpy as np
import pandas as pd
import pytest
from src.features.contracts import FeatureValidationError
from src.splits import chronological_split

def uneven(): return pd.DataFrame({'period':[2020,2020,2021,2022,2022,2022,2023,2024,2025], 'x':range(9)})

def test_boundary_split_complete_exclusive():
    split=chronological_split(uneven(),train_end=2021,validation_end=2023)
    assert split.train_periods==(2020,2021)
    assert split.validation_periods==(2022,2023)
    assert split.test_periods==(2024,2025)
    assert np.all(split.train_mask.astype(int)+split.validation_mask.astype(int)+split.test_mask.astype(int)==1)

def test_ratio_split_deterministic_for_shuffled_uneven_rows():
    a=chronological_split(uneven(),ratios=(0.5,0.25,0.25))
    b=chronological_split(uneven().sample(frac=1,random_state=1),ratios=(0.5,0.25,0.25))
    assert a.train_periods==b.train_periods and a.validation_periods==b.validation_periods and a.test_periods==b.test_periods

def test_ratio_validation_errors():
    for ratios in [(-.1,.6,.5), (np.nan,.5,.5), (np.inf,0,0), (.5,.5), (.2,.2,.2), (0,0,1)]:
        with pytest.raises(FeatureValidationError): chronological_split(uneven(),ratios=ratios)

def test_boundary_validation_errors():
    with pytest.raises(FeatureValidationError): chronological_split(uneven(),train_end=2022,validation_end=2021)
    with pytest.raises(FeatureValidationError): chronological_split(uneven(),train_end=2019,validation_end=2020)
    with pytest.raises(FeatureValidationError): chronological_split(uneven(),train_end=2021,validation_end=2023,ratios=(.5,.25,.25))
    with pytest.raises(FeatureValidationError): chronological_split(pd.DataFrame({'period':[2020,2021]}),ratios=(.5,.25,.25))
    with pytest.raises(FeatureValidationError): chronological_split(pd.DataFrame({'period':[2020,2020.5,2021]}),ratios=(.5,.25,.25))
