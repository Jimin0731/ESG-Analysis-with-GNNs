"""Framework-neutral contracts for economic graph snapshots."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np

class GraphValidationError(ValueError): pass

@dataclass(frozen=True)
class GraphNode:
    node_id: str
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class DirectedWeightedEdge:
    source: int
    target: int
    weight: float
    raw_flow: float
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class GraphConstructionConfig:
    backend: str
    threshold_policy: str = "absolute"
    threshold_value: float = 0.0
    include_self_loops: bool = False
    duplicate_edge_policy: str = "reject"
    weight_transform: str = "raw"

@dataclass(frozen=True)
class GraphConstructionReport:
    backend: str
    node_count: int
    edge_count: int
    density: float
    threshold_policy: str
    threshold_value: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class LeontiefComputationReport:
    shape: tuple[int, int]
    method: str
    fallback_policy: str
    condition_number: float
    used_fallback: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class GraphSnapshot:
    nodes: tuple[GraphNode, ...]
    edge_index: np.ndarray
    edge_weight: np.ndarray
    raw_flow: np.ndarray
    backend: str
    period: str | int | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    report: GraphConstructionReport | None = None

    @property
    def node_ids(self) -> list[str]: return [n.node_id for n in self.nodes]
    @property
    def labels(self) -> list[str]: return [n.label for n in self.nodes]

    def __post_init__(self) -> None:
        ids = [n.node_id for n in self.nodes]
        if any(not str(i) for i in ids) or len(ids) != len(set(ids)):
            raise GraphValidationError("node IDs must be unique and non-empty")
        ei = np.asarray(self.edge_index)
        ew = np.asarray(self.edge_weight, dtype=float)
        rf = np.asarray(self.raw_flow, dtype=float)
        if ei.shape[0] != 2:
            raise GraphValidationError("edge_index must have shape (2, E)")
        if ei.shape[1] != ew.shape[0] or ew.shape[0] != rf.shape[0]:
            raise GraphValidationError("edge arrays must have matching lengths")
        if ei.size and (ei.min() < 0 or ei.max() >= len(ids)):
            raise GraphValidationError("edge endpoints must be in range")
        if not np.isfinite(ew).all() or not np.isfinite(rf).all():
            raise GraphValidationError("edge weights and raw flows must be finite")
        if (rf < 0).any():
            raise GraphValidationError("raw economic flows must be non-negative")
        pairs = list(map(tuple, ei.T.tolist()))
        if len(pairs) != len(set(pairs)):
            raise GraphValidationError("duplicate edges require prior explicit aggregation")
