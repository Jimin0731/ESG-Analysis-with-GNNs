"""Validated ESG score loader."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd
from .bea import normalize_column_name
from .contracts import DataValidationError, DuplicatePolicy, MissingValuePolicy
@dataclass(frozen=True)
class ESGLoadResult:
    data: pd.DataFrame; column_mapping: dict[str,str]; row_count:int; dropped_row_count:int; duplicate_count:int; warnings: tuple[str,...]
def _read(path: Path):
    if path.suffix.lower()=='.csv': return pd.read_csv(path)
    if path.suffix.lower() in {'.xls','.xlsx'}: return pd.read_excel(path)
    raise DataValidationError(f"unsupported ESG file format: {path.suffix}")
def _match(cols, aliases):
    norm={normalize_column_name(c):c for c in cols}; hits=[norm[normalize_column_name(a)] for a in aliases if normalize_column_name(a) in norm]
    hits=list(dict.fromkeys(hits))
    if len(hits)>1: raise DataValidationError(f"ambiguous ESG aliases {hits}")
    return hits[0] if hits else None
def load_esg_scores(path: str|Path, *, aliases: dict[str,tuple[str,...]]|None=None, missing_score_policy: MissingValuePolicy=MissingValuePolicy.ERROR, duplicate_policy: DuplicatePolicy=DuplicatePolicy.ERROR, score_range=(0.0,100.0)) -> ESGLoadResult:
    defaults={"entity_id":("entity_id","company_id","ticker"),"entity_name":("entity_name","company","company_name"),"industry":("industry","sector"),"overall_score":("esg_score","overall_esg_score","esg"),"environmental_score":("environmental_score","environment_score","e_score"),"social_score":("social_score","s_score"),"governance_score":("governance_score","g_score"),"period":("year","period")}
    if aliases: defaults.update(aliases)
    df=_read(Path(path))
    if df.empty: raise DataValidationError("ESG input is empty")
    mapping={k:_match(df.columns,v) for k,v in defaults.items()}
    req=["entity_id","industry","overall_score"]
    miss=[k for k in req if not mapping.get(k)]
    if miss: raise DataValidationError(f"missing required ESG columns: {', '.join(miss)}")
    out=pd.DataFrame({k:df[v] for k,v in mapping.items() if v})
    scores=[c for c in ("overall_score","environmental_score","social_score","governance_score") if c in out]
    for c in scores:
        out[c]=pd.to_numeric(out[c], errors='coerce')
        if np.isinf(out[c].astype(float, errors='ignore')).any(): raise DataValidationError(f"ESG score column {c} contains infinite values")
    bad_nonnum=df[[mapping[c] for c in scores]].notna().to_numpy().sum() - out[scores].notna().to_numpy().sum()
    if bad_nonnum: raise DataValidationError("ESG score columns contain non-numeric values")
    low,high=score_range
    if ((out[scores].lt(low)|out[scores].gt(high)) & out[scores].notna()).any().any(): raise DataValidationError("ESG scores outside allowed range")
    if out["industry"].isna().any() or out["industry"].astype(str).str.strip().eq("").any(): raise DataValidationError("ESG records contain missing industry identifiers")
    dropped=0; warnings=[]
    if out[scores].isna().any().any():
        if missing_score_policy==MissingValuePolicy.ERROR: raise DataValidationError("ESG scores contain missing values")
        if missing_score_policy==MissingValuePolicy.DROP:
            before=len(out); out=out.dropna(subset=scores); dropped=before-len(out)
        else: warnings.append("missing scores preserved")
    keys=["entity_id"] + (["period"] if "period" in out else [])
    dup=int(out.duplicated(keys, keep=False).sum())
    if dup:
        if duplicate_policy==DuplicatePolicy.ERROR: raise DataValidationError("duplicate ESG entity-period records")
        if duplicate_policy==DuplicatePolicy.KEEP_FIRST: out=out.drop_duplicates(keys, keep='first')
        elif duplicate_policy==DuplicatePolicy.KEEP_LAST: out=out.drop_duplicates(keys, keep='last')
        elif duplicate_policy!=DuplicatePolicy.PRESERVE: raise DataValidationError(f"unsupported duplicate policy: {duplicate_policy}")
    return ESGLoadResult(out.reset_index(drop=True), {k:v for k,v in mapping.items() if v}, len(out), dropped, dup, tuple(warnings))
