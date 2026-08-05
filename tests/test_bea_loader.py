from __future__ import annotations

import pytest
from openpyxl import Workbook

from src.data.bea import load_bea_table
from src.data.contracts import DataValidationError


def create_synthetic_bea_workbook(path, *, metadata_rows: bool = True, ambiguous: bool = False, missing_time: bool = False) -> None:
    """Create a synthetic BEA-like workbook under pytest's tmp_path."""
    workbook = Workbook()
    worksheet = workbook.active
    if metadata_rows:
        worksheet.append(["Synthetic metadata row"])
        worksheet.append(["Generated only for pytest"])
    if ambiguous:
        worksheet.append(["Industry", "Industry Name", "2020"])
    elif missing_time:
        worksheet.append(["Industry Code", "Industry Name", "Value"])
    else:
        worksheet.append(["Industry Code", "Industry Name", "2020", "2021"])
    worksheet.append(["0011", "Agriculture [1]", "1,000", "1,100"])
    worksheet.append(["0021", "Mining", "(D)", "900"])
    workbook.save(path)


def test_explicit_header_row():
    assert load_bea_table("tests/fixtures/bea/header_first.csv", header_row=0).schema.header_row == 0


def test_automatic_header_detection():
    assert load_bea_table("tests/fixtures/bea/metadata_header.csv").schema.header_row == 2


def test_csv_loading():
    assert load_bea_table("tests/fixtures/bea/header_first.csv").row_count == 2


def test_xlsx_loading_header_first_row(tmp_path):
    path = tmp_path / "synthetic_header_first.xlsx"
    create_synthetic_bea_workbook(path, metadata_rows=False)
    result = load_bea_table(path)
    assert result.schema.header_row == 0
    assert result.row_count == 2


def test_xlsx_loading_with_metadata_header_detection(tmp_path):
    path = tmp_path / "synthetic_metadata.xlsx"
    create_synthetic_bea_workbook(path, metadata_rows=True)
    result = load_bea_table(path)
    assert result.schema.header_row == 2


def test_xlsx_industry_column_alias_detection(tmp_path):
    path = tmp_path / "synthetic_aliases.xlsx"
    create_synthetic_bea_workbook(path, metadata_rows=True)
    result = load_bea_table(path)
    assert result.schema.industry_name_column == "Industry Name"
    assert result.schema.industry_code_column == "Industry Code"


def test_xlsx_year_column_detection(tmp_path):
    path = tmp_path / "synthetic_years.xlsx"
    create_synthetic_bea_workbook(path, metadata_rows=False)
    assert load_bea_table(path).schema.time_value_columns == ("2020", "2021")


def test_xlsx_comma_formatted_numbers(tmp_path):
    path = tmp_path / "synthetic_commas.xlsx"
    create_synthetic_bea_workbook(path, metadata_rows=False)
    assert load_bea_table(path).data.loc[0, "2020"] == 1000


def test_xlsx_preserves_leading_zero_industry_codes(tmp_path):
    path = tmp_path / "synthetic_leading_zero_codes.xlsx"
    create_synthetic_bea_workbook(path, metadata_rows=False)
    assert load_bea_table(path).data["industry_code"].tolist() == ["0011", "0021"]


def test_xlsx_suppressed_value_handling(tmp_path):
    path = tmp_path / "synthetic_suppressed.xlsx"
    create_synthetic_bea_workbook(path, metadata_rows=False)
    result = load_bea_table(path)
    assert result.data["2020"].isna().iloc[1]


def test_xlsx_ambiguous_workbook_errors(tmp_path):
    path = tmp_path / "synthetic_ambiguous.xlsx"
    create_synthetic_bea_workbook(path, ambiguous=True)
    with pytest.raises(DataValidationError):
        load_bea_table(path)


def test_xlsx_malformed_workbook_errors(tmp_path):
    path = tmp_path / "synthetic_malformed.xlsx"
    create_synthetic_bea_workbook(path, missing_time=True)
    with pytest.raises(DataValidationError):
        load_bea_table(path)


def test_industry_column_alias_detection():
    assert load_bea_table("tests/fixtures/bea/metadata_header.csv").schema.industry_name_column == "Description"


def test_year_time_column_detection():
    assert "2019" in load_bea_table("tests/fixtures/bea/header_first.csv").schema.time_value_columns


def test_numeric_normalization():
    assert load_bea_table("tests/fixtures/bea/header_first.csv").data.loc[0, "2019"] == 1200


def test_suppressed_value_handling():
    assert load_bea_table("tests/fixtures/bea/header_first.csv").data["2020"].isna().any()


def test_empty_input(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(DataValidationError):
        load_bea_table(path)


def test_preserves_meaningful_parenthesized_industry_names(tmp_path):
    path = tmp_path / "parentheses.csv"
    path.write_text("Industry Code,Industry Name,2020\n0031,Finance and insurance (except funds),42\n")
    result = load_bea_table(path)
    assert result.data.loc[0, "industry_name"] == "Finance and insurance (except funds)"
    assert result.data.loc[0, "industry_code"] == "0031"


def test_invalid_explicit_header_row_errors(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("Industry Code,Industry Name,2020\n0011,Agriculture,1\n")
    with pytest.raises(DataValidationError):
        load_bea_table(path, header_row=10)


def test_ambiguous_schema():
    with pytest.raises(DataValidationError):
        load_bea_table("tests/fixtures/bea/ambiguous.csv")


def test_missing_industry_columns():
    with pytest.raises(DataValidationError):
        load_bea_table("tests/fixtures/bea/missing_industry.csv")


def test_missing_time_value_columns():
    with pytest.raises(DataValidationError):
        load_bea_table("tests/fixtures/bea/no_time.csv")
