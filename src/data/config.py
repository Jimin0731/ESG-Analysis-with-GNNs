"""Load and validate typed dataset configuration."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from .contracts import DataValidationError, DatasetConfig, DatasetIdentity, DuplicatePolicy, FileFormat, MissingValuePolicy, SourcePath

def _tuple(v: Any) -> tuple[str,...]:
    if v is None: return ()
    if isinstance(v, str): return (v,)
    return tuple(str(x) for x in v)

def dataset_from_mapping(item: dict[str, Any]) -> DatasetConfig:
    for field in ("id","provider","expected_path","format","required"):
        if field not in item: raise DataValidationError(f"dataset config missing required field {field!r}")
    return DatasetConfig(
        identity=DatasetIdentity(str(item["id"]), str(item.get("description", ""))),
        provider=str(item["provider"]), source_path=SourcePath(str(item["expected_path"])),
        file_format=FileFormat.parse(str(item["format"])), required=bool(item["required"]),
        series_or_table_id=item.get("series_or_table_id") or item.get("table_id"),
        industry_id_column=item.get("industry_id_column"), industry_name_column=item.get("industry_name_column"),
        time_columns=_tuple(item.get("time_columns")), detect_time_columns=bool(item.get("detect_time_columns", True)),
        score_or_value_columns=_tuple(item.get("score_or_value_columns")),
        missing_value_policy=MissingValuePolicy(str(item.get("missing_value_policy", "error"))),
        duplicate_policy=DuplicatePolicy(str(item.get("duplicate_policy", "error"))),
        minimum_mapping_coverage=float(item.get("minimum_mapping_coverage", 0.0)), metadata={k:v for k,v in item.items() if k not in {"id","description","provider","expected_path","format","required"}}
    )

def load_dataset_configs(path: str | Path) -> list[DatasetConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    datasets = raw.get("datasets")
    if not isinstance(datasets, list): raise DataValidationError("datasets.yaml must contain a datasets list")
    out=[]; seen=set()
    for item in datasets:
        cfg=dataset_from_mapping(item)
        if cfg.identity.id in seen: raise DataValidationError(f"duplicate dataset id: {cfg.identity.id}")
        seen.add(cfg.identity.id); out.append(cfg)
    return out
