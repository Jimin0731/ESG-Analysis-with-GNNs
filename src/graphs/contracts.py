"""Framework-neutral contracts for economic graph snapshots."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .common import validate_config_common


class GraphValidationError(ValueError):
    """Raised when an in-memory graph snapshot violates array contracts."""


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

    def __post_init__(self) -> None:
        validate_config_common(
            backend=self.backend,
            threshold_policy=self.threshold_policy,
            threshold_value=self.threshold_value,
            duplicate_edge_policy=self.duplicate_edge_policy,
            weight_transform=self.weight_transform,
        )


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
    condition_number: float
    fallback_policy: str
    method: str
    used_fallback: bool
    regularization_value: float | None = None


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
    leontief_report: LeontiefComputationReport | None = None

    @property
    def node_ids(self) -> list[str]:
        return [node.node_id for node in self.nodes]

    @property
    def labels(self) -> list[str]:
        return [node.label for node in self.nodes]

    def __post_init__(self) -> None:
        if not self.nodes:
            raise GraphValidationError("at least one node is required")
        trimmed_ids: list[str] = []
        for node in self.nodes:
            if not isinstance(node.node_id, str) or not isinstance(node.label, str):
                raise GraphValidationError("node IDs and labels must be strings")
            node_id = node.node_id.strip()
            label = node.label.strip()
            if not node_id or not label:
                raise GraphValidationError("node IDs and labels must be non-empty after trimming")
            trimmed_ids.append(node_id)
        if len(trimmed_ids) != len(set(trimmed_ids)):
            raise GraphValidationError("node IDs must remain unique after trimming")

        edge_index = np.asarray(self.edge_index)
        if not np.issubdtype(edge_index.dtype, np.integer):
            raise GraphValidationError("edge_index must be an integer array")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise GraphValidationError("edge_index must have shape (2, E)")
        edge_weight = np.asarray(self.edge_weight, dtype=float)
        raw_flow = np.asarray(self.raw_flow, dtype=float)
        if edge_weight.ndim != 1 or raw_flow.ndim != 1:
            raise GraphValidationError("edge_weight and raw_flow must be one-dimensional")
        if edge_index.shape[1] != edge_weight.shape[0] or edge_weight.shape[0] != raw_flow.shape[0]:
            raise GraphValidationError("edge arrays must have matching lengths")
        if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= len(trimmed_ids)):
            raise GraphValidationError("edge endpoints must be valid node positions")
        if not np.isfinite(edge_weight).all() or not np.isfinite(raw_flow).all():
            raise GraphValidationError("edge weights and raw flows must be finite")
        if (raw_flow < 0).any():
            raise GraphValidationError("raw economic flows must be non-negative")
        pairs = list(map(tuple, edge_index.T.tolist()))
        if len(pairs) != len(set(pairs)):
            raise GraphValidationError("duplicate edges require prior explicit aggregation")
