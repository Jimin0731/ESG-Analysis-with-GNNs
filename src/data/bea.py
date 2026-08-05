"""Robust BEA-style tabular loader for small synthetic/local files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from .contracts import DataValidationError


@dataclass(frozen=True)
class BEASchema:
    industry_code_column: str | None
    industry_name_column: str
    time_value_columns: tuple[str, ...]
    header_row: int


@dataclass(frozen=True)
class BEALoadResult:
    data: pd.DataFrame
    schema: BEASchema
    source_path: Path
    row_count: int


def normalize_column_name(value: object) -> str:
    """Normalize provider column names for deterministic matching/output."""
    normalized = re.sub(r"\s+", "_", str(value).strip().lower())
    return re.sub(r"[^0-9a-z_]+", "", normalized).strip("_")


def _clean_text(value: object) -> str:
    """Trim whitespace and verified footnotes while preserving meaningful text."""
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value).strip())
    text = re.sub(r"\s*\[\d+\]\s*$", "", text)
    text = re.sub(r"\s*\(\d+\)\s*$", "", text)
    return text.strip()


def _num(value: object):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().replace(",", "").replace("$", "")
    if text in {"", "--", "---", "(NA)", "NA", "N/A", "n.a.", "...", "suppressed", "Suppressed", "(D)", "D"}:
        return pd.NA
    if re.fullmatch(r"\([^()]+\)", text):
        text = "-" + text[1:-1]
    return pd.to_numeric(text, errors="coerce")


def _validate_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".xls", ".xlsx"}:
        raise DataValidationError(f"unsupported BEA file format: {path.suffix}")
    return suffix


def _read_preview(path: Path, search_rows: int) -> pd.DataFrame:
    suffix = _validate_suffix(path)
    if search_rows <= 0:
        raise DataValidationError("search_rows must be a positive integer")
    try:
        if suffix == ".csv":
            return pd.read_csv(path, header=None, nrows=search_rows, dtype=object, keep_default_na=False, engine="python")
        return pd.read_excel(path, header=None, nrows=search_rows, dtype=object)
    except EmptyDataError as exc:
        raise DataValidationError("BEA input is empty") from exc
    except (ParserError, ValueError, OSError) as exc:
        raise DataValidationError(f"unable to read BEA input: {path}") from exc


def _read_table(path: Path, header_row: int) -> pd.DataFrame:
    suffix = _validate_suffix(path)
    try:
        if suffix == ".csv":
            return pd.read_csv(path, header=header_row, dtype=object, keep_default_na=False, engine="python")
        return pd.read_excel(path, header=header_row, dtype=object)
    except EmptyDataError as exc:
        raise DataValidationError("BEA input is empty") from exc
    except (ParserError, ValueError, StopIteration, OSError) as exc:
        raise DataValidationError(f"unable to read BEA input with header_row={header_row}") from exc


def _find_cols(columns: Iterable[object], aliases: Iterable[str]) -> list[object]:
    normalized_aliases = {normalize_column_name(alias) for alias in aliases}
    found = [column for column in columns if normalize_column_name(column) in normalized_aliases]
    return list(dict.fromkeys(found))


def _time_cols(columns: Iterable[object]) -> list[object]:
    out = []
    for column in columns:
        normalized = normalize_column_name(column)
        if (
            re.fullmatch(r"(19|20)\d{2}", normalized)
            or re.fullmatch(r"(19|20)\d{2}q[1-4]", normalized)
            or re.fullmatch(r"value_(19|20)\d{2}", normalized)
        ):
            out.append(column)
    return out


def _non_empty_row_values(row: pd.Series) -> list[object]:
    values = []
    for value in row.tolist():
        if value is None:
            continue
        text = str(value).strip()
        if text and not text.lower().startswith("unnamed:"):
            values.append(value)
    return values


def _detect_header_row(
    preview: pd.DataFrame,
    *,
    industry_code_aliases: Iterable[str],
    industry_name_aliases: Iterable[str],
) -> int:
    if preview.empty:
        raise DataValidationError("BEA input is empty")
    candidates: list[int] = []
    for row_index in range(len(preview.index)):
        values = _non_empty_row_values(preview.iloc[row_index])
        if not values:
            continue
        industry_names = _find_cols(values, industry_name_aliases)
        industry_codes = _find_cols(values, industry_code_aliases)
        times = _time_cols(values)
        if len(industry_names) == 1 and len(industry_codes) <= 1 and times:
            candidates.append(row_index)
    if not candidates:
        raise DataValidationError("could not detect a BEA header row")
    if len(candidates) > 1:
        raise DataValidationError(f"multiple plausible BEA header rows detected: {candidates}")
    return candidates[0]


def _validate_explicit_header_row(path: Path, header_row: int, search_rows: int) -> None:
    if not isinstance(header_row, int):
        raise DataValidationError("header_row must be an integer")
    if header_row < 0:
        raise DataValidationError("header_row must be greater than or equal to zero")
    preview = _read_preview(path, max(header_row + 1, search_rows))
    if header_row >= len(preview.index):
        raise DataValidationError(f"header_row {header_row} is outside the available row range")


def load_bea_table(
    path: str | Path,
    *,
    header_row: int | None = None,
    search_rows: int = 12,
    industry_code_aliases=("Industry Code", "Code", "LineCode"),
    industry_name_aliases=("Industry", "Industry Name", "Description"),
) -> BEALoadResult:
    """Load a BEA-style table with explicit validation and deterministic schema metadata."""
    path = Path(path)
    _validate_suffix(path)
    if header_row is None:
        preview = _read_preview(path, search_rows)
        header_row = _detect_header_row(
            preview,
            industry_code_aliases=industry_code_aliases,
            industry_name_aliases=industry_name_aliases,
        )
    else:
        _validate_explicit_header_row(path, header_row, search_rows)

    df = _read_table(path, header_row)
    if df.empty:
        raise DataValidationError("BEA input is empty")

    industry_name_columns = _find_cols(df.columns, industry_name_aliases)
    industry_code_columns = _find_cols(df.columns, industry_code_aliases)
    time_columns = _time_cols(df.columns)
    if len(industry_name_columns) != 1:
        raise DataValidationError("BEA schema must contain exactly one industry-name column")
    if len(industry_code_columns) > 1:
        raise DataValidationError("BEA schema has ambiguous industry-code columns")
    if not time_columns:
        raise DataValidationError("BEA schema has no usable time/value columns")

    name_column = industry_name_columns[0]
    code_column = industry_code_columns[0] if industry_code_columns else None
    out = pd.DataFrame({"industry_name": df[name_column].map(_clean_text)})
    if code_column:
        out["industry_code"] = df[code_column].map(lambda value: _clean_text(value) if not pd.isna(value) else pd.NA).astype("string")
    for column in time_columns:
        out[normalize_column_name(column)] = df[column].map(_num).astype("Float64")

    out = out[out["industry_name"].astype(str).str.len() > 0].reset_index(drop=True)
    if out.empty:
        raise DataValidationError("BEA file contains no usable industry rows")
    if not any(out[normalize_column_name(column)].notna().any() for column in time_columns):
        raise DataValidationError("BEA file contains no usable time/value cells")

    return BEALoadResult(
        data=out,
        schema=BEASchema(
            industry_code_column=str(code_column) if code_column is not None else None,
            industry_name_column=str(name_column),
            time_value_columns=tuple(normalize_column_name(column) for column in time_columns),
            header_row=header_row,
        ),
        source_path=path,
        row_count=len(out),
    )
