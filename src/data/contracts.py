"""Typed data contracts shared by dataset loaders."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

class DataValidationError(ValueError):
    """Raised when configured data or schema is invalid."""

class FileFormat(str, Enum):
    CSV="csv"; XLS="xls"; XLSX="xlsx"; XLSX_OR_CSV="xlsx_or_csv"; JSON="json"; SQLITE="sqlite"
    @classmethod
    def parse(cls, value: str) -> "FileFormat":
        try: return cls(str(value).lower())
        except ValueError as e: raise DataValidationError(f"Unsupported file format: {value!r}") from e

class MissingValuePolicy(str, Enum):
    ERROR="error"; DROP="drop"; PRESERVE="preserve"
class DuplicatePolicy(str, Enum):
    ERROR="error"; KEEP_FIRST="keep_first"; KEEP_LAST="keep_last"; PRESERVE="preserve"; AGGREGATE="aggregate"

@dataclass(frozen=True)
class DatasetIdentity:
    id: str
    description: str = ""
    def __post_init__(self):
        if not self.id or not self.id.strip(): raise DataValidationError("dataset id is required")

@dataclass(frozen=True)
class SourcePath:
    path: str
    def __post_init__(self):
        if not self.path or str(self.path).startswith(("/Users/", "/home/")):
            raise DataValidationError(f"source path must be repository-relative or env-substituted: {self.path!r}")

@dataclass(frozen=True)
class DatasetConfig:
    identity: DatasetIdentity
    provider: str
    source_path: SourcePath
    file_format: FileFormat
    required: bool
    series_or_table_id: str | None = None
    industry_id_column: str | None = None
    industry_name_column: str | None = None
    time_columns: tuple[str, ...] = ()
    detect_time_columns: bool = True
    score_or_value_columns: tuple[str, ...] = ()
    missing_value_policy: MissingValuePolicy = MissingValuePolicy.ERROR
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.ERROR
    minimum_mapping_coverage: float = 0.0
    optional: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        if not self.provider: raise DataValidationError(f"provider is required for {self.identity.id}")
        if not 0 <= self.minimum_mapping_coverage <= 1: raise DataValidationError("minimum mapping coverage must be between 0 and 1")
        object.__setattr__(self, "optional", not self.required)
