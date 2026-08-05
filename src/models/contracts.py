"""Common contracts for Migration PR 6 model registry."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping
import math
import torch

CANONICAL_MODEL_NAMES=("mlp","gcn","gat","weighted_gat","bidirectional_gnn","gpr_gnn")

class ModelValidationError(ValueError):
    """Raised when model configuration, inputs, or outputs violate contracts."""

def _pos_int(name: str, value: int) -> None:
    if isinstance(value,bool) or not isinstance(value,int) or value <= 0:
        raise ModelValidationError(f"{name} must be a positive integer")

def _finite_num(name: str, value: Any) -> None:
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(float(value)):
        raise ModelValidationError(f"{name} must be finite numeric")

@dataclass(frozen=True)
class ModelConfig:
    name: str
    input_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    target_names: tuple[str,...]
    attention_heads: int | None = None
    activation: str = "relu"
    residual: bool = True
    seed: int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    alpha: float = 0.1
    propagation_steps: int = 10
    graph_direction: str = "stored"
    self_loop_weight: float = 1.0
    def __post_init__(self):
        if self.name not in CANONICAL_MODEL_NAMES: raise ModelValidationError(f"unsupported model name: {self.name}")
        _pos_int("input_dim", self.input_dim); _pos_int("hidden_dim", self.hidden_dim); _pos_int("num_layers", self.num_layers)
        _finite_num("dropout", self.dropout)
        if not (0.0 <= float(self.dropout) < 1.0): raise ModelValidationError("dropout must be in [0, 1)")
        if not self.target_names: raise ModelValidationError("target_names must be non-empty")
        if len(set(self.target_names)) != len(self.target_names): raise ModelValidationError("target_names must be unique")
        for t in self.target_names:
            if not isinstance(t,str) or not t.startswith("target__"): raise ModelValidationError("target names must begin with target__")
        if self.attention_heads is not None: _pos_int("attention_heads", self.attention_heads)
        if self.name in {"gat","weighted_gat"} and self.attention_heads is None: raise ModelValidationError("attention_heads is required for attention models")
        if self.name in {"gat","weighted_gat"} and self.hidden_dim % int(self.attention_heads) != 0: raise ModelValidationError("hidden_dim must be divisible by attention_heads")
        if self.seed is not None: _pos_int("seed", self.seed)
        if self.activation not in {"relu","gelu","tanh","identity"}: raise ModelValidationError("unsupported activation")
        _finite_num("alpha", self.alpha)
        if not 0.0 <= float(self.alpha) <= 1.0: raise ModelValidationError("alpha must be in [0, 1]")
        if isinstance(self.propagation_steps,bool) or not isinstance(self.propagation_steps,int) or self.propagation_steps < 0: raise ModelValidationError("propagation_steps must be a non-negative integer")
        if self.graph_direction not in {"stored","reverse"}: raise ModelValidationError("graph_direction must be stored or reverse")
        _finite_num("self_loop_weight", self.self_loop_weight)
        if float(self.self_loop_weight) < 0: raise ModelValidationError("self_loop_weight must be non-negative")
        allowed={"combination","weighted"}
        bad=set(self.options)-allowed
        if bad: raise ModelValidationError(f"unsupported configuration keys: {sorted(bad)}")
        for k,v in self.options.items():
            if isinstance(v,(int,float)): _finite_num(k,v)
    def to_report(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class ModelCapabilities:
    name: str; uses_graph: bool; preserves_direction: bool; uses_edge_weight: bool; provides_attention: bool; uses_reverse_graph: bool; supports_multiple_targets: bool; combination: str | None = None
    def to_report(self) -> dict[str, Any]: return asdict(self)

@dataclass(frozen=True)
class ModelOutput:
    predictions: torch.Tensor
    target_names: tuple[str,...]
    node_embeddings: torch.Tensor
    auxiliary: Mapping[str, Any] = field(default_factory=dict)
    def validate(self) -> "ModelOutput":
        if self.predictions.ndim != 2: raise ModelValidationError("predictions must be two-dimensional")
        if self.predictions.shape[1] != len(self.target_names): raise ModelValidationError("prediction width must match target_names")
        if self.node_embeddings.ndim != 2: raise ModelValidationError("node_embeddings must be two-dimensional")
        if self.predictions.shape[0] != self.node_embeddings.shape[0]: raise ModelValidationError("node counts must match")
        if not torch.isfinite(self.predictions).all() or not torch.isfinite(self.node_embeddings).all(): raise ModelValidationError("outputs must be finite")
        return self
__all__=["CANONICAL_MODEL_NAMES","ModelValidationError","ModelConfig","ModelCapabilities","ModelOutput"]
