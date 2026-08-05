"""Public economic graph backend interface."""
from .common import GraphInputError
from .contracts import (
    DirectedWeightedEdge,
    GraphConstructionConfig,
    GraphConstructionReport,
    GraphNode,
    GraphSnapshot,
    GraphValidationError,
    LeontiefComputationReport,
)
from .icio import ICIOGraphConfig, build_icio_graph
from .use_make import UseMakeGraphConfig, build_use_make_graph, compute_leontief_inverse

__all__ = [
    "DirectedWeightedEdge",
    "GraphConstructionConfig",
    "GraphConstructionReport",
    "GraphInputError",
    "GraphNode",
    "GraphSnapshot",
    "GraphValidationError",
    "ICIOGraphConfig",
    "LeontiefComputationReport",
    "UseMakeGraphConfig",
    "build_economic_graph",
    "build_icio_graph",
    "build_use_make_graph",
    "compute_leontief_inverse",
]


def build_economic_graph(backend: str, *args, **kwargs):
    if backend == "use_make":
        return build_use_make_graph(*args, **kwargs)
    if backend == "icio":
        return build_icio_graph(*args, **kwargs)
    raise ValueError(f"unsupported graph backend: {backend}")
