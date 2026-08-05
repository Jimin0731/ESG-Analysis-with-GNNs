import pytest
from src.data.esg import load_esg_scores
from src.data.contracts import DataValidationError, MissingValuePolicy, DuplicatePolicy

def test_valid_esg_file(): assert load_esg_scores('tests/fixtures/esg/valid_esg.csv').row_count==2
def test_column_aliases(): assert 'sector' in load_esg_scores('tests/fixtures/esg/valid_esg.csv').column_mapping.values()
def test_missing_required_columns(tmp_path):
    p=tmp_path/'x.csv'; p.write_text('company_id,sector\nA,X\n')
    with pytest.raises(DataValidationError): load_esg_scores(p)
def test_empty_input(tmp_path):
    p=tmp_path/'x.csv'; p.write_text('')
    with pytest.raises(Exception): load_esg_scores(p)
def test_duplicate_records():
    with pytest.raises(DataValidationError): load_esg_scores('tests/fixtures/esg/duplicate.csv')
def test_missing_value_policy_error():
    with pytest.raises(DataValidationError): load_esg_scores('tests/fixtures/esg/missing_score.csv')
def test_missing_value_policy_drop(): assert load_esg_scores('tests/fixtures/esg/missing_score.csv', missing_score_policy=MissingValuePolicy.DROP).dropped_row_count==1
def test_missing_value_policy_preserve(): assert load_esg_scores('tests/fixtures/esg/missing_score.csv', missing_score_policy=MissingValuePolicy.PRESERVE).warnings
def test_non_numeric_scores():
    with pytest.raises(DataValidationError): load_esg_scores('tests/fixtures/esg/bad_score.csv')
def test_infinite_scores(tmp_path):
    p=tmp_path/'x.csv'; p.write_text('company_id,sector,esg_score\nA,X,inf\n')
    with pytest.raises(DataValidationError): load_esg_scores(p)
def test_configurable_score_ranges():
    with pytest.raises(DataValidationError): load_esg_scores('tests/fixtures/esg/valid_esg.csv', score_range=(0,75))
def test_missing_industries(tmp_path):
    p=tmp_path/'x.csv'; p.write_text('company_id,sector,esg_score\nA,,80\n')
    with pytest.raises(DataValidationError): load_esg_scores(p)
def test_duplicate_keep_first(): assert load_esg_scores('tests/fixtures/esg/duplicate.csv', duplicate_policy=DuplicatePolicy.KEEP_FIRST).row_count==1
