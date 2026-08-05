"""Robust BEA-style tabular loader for small synthetic/local files."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd
from .contracts import DataValidationError

@dataclass(frozen=True)
class BEASchema:
    industry_code_column: str|None; industry_name_column: str; time_value_columns: tuple[str,...]; header_row: int
@dataclass(frozen=True)
class BEALoadResult:
    data: pd.DataFrame; schema: BEASchema; source_path: Path; row_count: int

def normalize_column_name(v: object) -> str:
    s=re.sub(r"\s+","_",str(v).strip().lower()); return re.sub(r"[^0-9a-z_]+","",s).strip("_")
def _clean_text(v: object) -> str:
    return re.sub(r"\s+"," ", re.sub(r"\[[^]]*\]|\([^)]*\)$", "", str(v).strip())).strip()
def _num(v: object):
    if pd.isna(v): return pd.NA
    s=str(v).strip().replace(",","").replace("$","")
    if s in {"","--","---","(NA)","NA","N/A","n.a.","...","suppressed","Suppressed","(D)","D"}: return pd.NA
    if re.fullmatch(r"\([^()]+\)", s): s="-"+s[1:-1]
    return pd.to_numeric(s, errors="coerce")
def _read(path: Path, header):
    if path.suffix.lower()=='.csv': return pd.read_csv(path, header=header)
    if path.suffix.lower() in {'.xls','.xlsx'}: return pd.read_excel(path, header=header)
    raise DataValidationError(f"unsupported BEA file format: {path.suffix}")
def _find_cols(cols, aliases):
    norm={normalize_column_name(c):c for c in cols}; found=[]
    for a in aliases:
        n=normalize_column_name(a)
        found += [orig for key,orig in norm.items() if key==n]
    return list(dict.fromkeys(found))
def _time_cols(cols):
    out=[]
    for c in cols:
        n=normalize_column_name(c)
        if re.fullmatch(r"(19|20)\d{2}", n) or re.fullmatch(r"(19|20)\d{2}q[1-4]", n) or re.fullmatch(r"value_(19|20)\d{2}", n): out.append(c)
    return out

def load_bea_table(path: str|Path, *, header_row: int|None=None, search_rows: int=12, industry_code_aliases=("Industry Code","Code","LineCode"), industry_name_aliases=("Industry","Industry Name","Description")) -> BEALoadResult:
    path=Path(path)
    if header_row is None:
        candidates=[]
        for h in range(search_rows):
            df=_read(path,h)
            inds=_find_cols(df.columns, industry_name_aliases); times=_time_cols(df.columns)
            if inds and times: candidates.append((h,df,inds,times))
        if len(candidates)!=1: raise DataValidationError("could not detect a unique BEA header row")
        header_row,df,inds,times=candidates[0]
    else:
        df=_read(path,header_row); inds=_find_cols(df.columns, industry_name_aliases); times=_time_cols(df.columns)
    if df.empty: raise DataValidationError("BEA input is empty")
    code_cols=_find_cols(df.columns, industry_code_aliases)
    if len(inds)!=1: raise DataValidationError("BEA schema must contain exactly one industry-name column")
    if len(code_cols)>1: raise DataValidationError("BEA schema has ambiguous industry-code columns")
    if not times: raise DataValidationError("BEA schema has no usable time/value columns")
    name_col=inds[0]; code_col=code_cols[0] if code_cols else None
    out=pd.DataFrame({"industry_name": df[name_col].map(_clean_text)})
    if code_col: out["industry_code"]=df[code_col].map(lambda x: _clean_text(x) if not pd.isna(x) else pd.NA).astype("string")
    for c in times: out[normalize_column_name(c)] = df[c].map(_num).astype("Float64")
    out=out[out["industry_name"].astype(str).str.len()>0].reset_index(drop=True)
    if out.empty: raise DataValidationError("BEA file contains no usable industry rows")
    if not any(out[normalize_column_name(c)].notna().any() for c in times): raise DataValidationError("BEA file contains no usable time/value cells")
    return BEALoadResult(out, BEASchema(code_col, name_col, tuple(normalize_column_name(c) for c in times), header_row), path, len(out))
