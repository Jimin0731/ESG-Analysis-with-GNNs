"""Industry mapping and coverage reporting."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from .contracts import DataValidationError
@dataclass(frozen=True)
class MappingCoverageReport:
    total_source_rows:int; unique_source_industries:int; matched_rows:int; unmatched_rows:int; matched_unique_industries:int; unmatched_unique_industries:int; row_coverage_ratio:float; unique_industry_coverage_ratio:float; unmatched_values:tuple[str,...]
def map_industries(df: pd.DataFrame, source_column: str, mapping: dict[str,str], *, output_column='canonical_industry_id', min_coverage=0.0, strict=False, aggregate=False) -> tuple[pd.DataFrame, MappingCoverageReport]:
    if source_column not in df: raise DataValidationError(f"missing source industry column: {source_column}")
    if not aggregate and df.index.has_duplicates: raise DataValidationError("duplicate input index could hide row duplication; reset index or enable aggregation")
    out=df.copy(); vals=out[source_column].astype('string').str.strip(); out[output_column]=vals.map(mapping)
    matched=out[output_column].notna(); unique=set(vals.dropna()); unmatched=tuple(sorted(str(v) for v in unique if v not in mapping))
    report=MappingCoverageReport(len(out), len(unique), int(matched.sum()), int((~matched).sum()), len(unique)-len(unmatched), len(unmatched), (float(matched.mean()) if len(out) else 0.0), ((len(unique)-len(unmatched))/len(unique) if unique else 0.0), unmatched)
    if strict and report.row_coverage_ratio < min_coverage: raise DataValidationError(f"mapping coverage {report.row_coverage_ratio:.3f} below minimum {min_coverage:.3f}")
    return out, report
