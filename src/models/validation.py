"""Tensor and graph validation for directed source-to-target models."""
from __future__ import annotations
import torch
from .contracts import ModelValidationError

def validate_features(x: torch.Tensor, input_dim: int) -> None:
    if not isinstance(x, torch.Tensor): raise ModelValidationError("x must be a tensor")
    if x.ndim != 2: raise ModelValidationError("x must have shape [N, F]")
    if not x.is_floating_point(): raise ModelValidationError("x must be floating point")
    if x.shape[1] != input_dim: raise ModelValidationError("feature width does not match input_dim")
    if not torch.isfinite(x).all(): raise ModelValidationError("x must be finite")

def validate_graph(x: torch.Tensor, edge_index: torch.Tensor | None, edge_weight: torch.Tensor | None = None, *, require_weights: bool=False, non_negative_weights: bool=False):
    if edge_index is None:
        edge_index=torch.empty((2,0), dtype=torch.long, device=x.device)
    if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2: raise ModelValidationError("edge_index must have shape [2, E]")
    if edge_index.dtype not in (torch.int64, torch.int32): raise ModelValidationError("edge_index must be an integer tensor")
    if edge_index.device != x.device: raise ModelValidationError("edge_index device must match x")
    if edge_index.numel():
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= x.shape[0]: raise ModelValidationError("edge endpoints out of range")
    if edge_weight is None:
        if require_weights: raise ModelValidationError("edge_weight is required")
    else:
        if not isinstance(edge_weight, torch.Tensor) or edge_weight.ndim != 1: raise ModelValidationError("edge_weight must have shape [E]")
        if edge_weight.shape[0] != edge_index.shape[1]: raise ModelValidationError("edge_weight length mismatch")
        if edge_weight.device != x.device: raise ModelValidationError("edge_weight device must match x")
        if not edge_weight.is_floating_point(): raise ModelValidationError("edge_weight must be floating point")
        if not torch.isfinite(edge_weight).all(): raise ModelValidationError("edge weights must be finite")
        if non_negative_weights and bool((edge_weight < 0).any()): raise ModelValidationError("edge weights must be non-negative")
    return edge_index, edge_weight
__all__=["validate_features","validate_graph"]
