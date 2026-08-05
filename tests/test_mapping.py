import pandas as pd, pytest
from src.data.mapping import map_industries
from src.data.contracts import DataValidationError

def df(): return pd.DataFrame({'industry':['A','B','C','C'], 'v':[1,2,3,4]})
def test_complete_mapping(): assert map_industries(df(),'industry',{'A':'a','B':'b','C':'c'})[1].row_coverage_ratio==1
def test_partial_mapping(): assert map_industries(df(),'industry',{'A':'a'})[1].matched_rows==1
def test_unmatched_value_reporting(): assert map_industries(df(),'industry',{'A':'a'})[1].unmatched_values==('B','C')
def test_strict_minimum_coverage_failure():
    with pytest.raises(DataValidationError): map_industries(df(),'industry',{'A':'a'}, min_coverage=.9, strict=True)
def test_deterministic_coverage_report(): assert map_industries(df(),'industry',{'A':'a'})[1].unmatched_values==('B','C')
def test_duplicate_preservation_behavior(): assert len(map_industries(df(),'industry',{'C':'c'})[0])==4
