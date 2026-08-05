"""GNN models extracted from the canonical ESG notebook.

The canonical notebook combines GAT/GCN-style message passing, ESG channel
heads, and economic-prior fusion.  This module exposes the reusable subset in a
script/test friendly form and gracefully falls back to dense PyTorch message
passing when ``torch_geometric`` is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict

import importlib.util

import torch
from torch import nn
import torch.nn.functional as F

if importlib.util.find_spec("torch_geometric") is not None:
    from torch_geometric.nn import GATConv, GCNConv
else:  # optional dependency for smoke-test environments
    GATConv = GCNConv = None


@dataclass
class MeanScoreBaseline:
    mean_score: float | None = None
    def fit(self, targets: list[float]) -> "MeanScoreBaseline":
        if len(targets) == 0:
            raise ValueError("targets must contain at least one score")
        self.mean_score = mean(targets)
        return self
    def predict(self, n_samples: int) -> list[float]:
        if self.mean_score is None:
            raise ValueError("Model must be fitted before calling predict().")
        return [self.mean_score] * n_samples


class DenseGraphConv(nn.Module):
    """Small weighted message-passing fallback compatible with edge_index."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.self_linear = nn.Linear(in_channels, out_channels)
        self.neighbor_linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        row, col = edge_index
        if edge_weight is None:
            edge_weight = torch.ones(row.shape[0], device=x.device, dtype=x.dtype)
        messages = x[row] * edge_weight.to(x.dtype).unsqueeze(-1)
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, col, messages)
        degree = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        degree.index_add_(0, col, edge_weight.to(x.dtype).abs())
        aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(-1)
        return self.self_linear(x) + self.neighbor_linear(aggregated)


class ESGChannelGating(nn.Module):
    """Notebook-inspired gating for environmental/social/governance channels."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3), nn.Sigmoid())
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.gate(hidden)
        scale = scores.mean(dim=1, keepdim=True)
        return self.proj(hidden) * scale + hidden, scores


class EconomicESGGNN(nn.Module):
    """End-to-end ESG prediction model based on the canonical notebook design."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        if GCNConv is not None:
            self.upstream = GCNConv(hidden_dim, hidden_dim)
            self.downstream = GCNConv(hidden_dim, hidden_dim)
            self.uses_pyg = True
        else:
            self.upstream = DenseGraphConv(hidden_dim, hidden_dim)
            self.downstream = DenseGraphConv(hidden_dim, hidden_dim)
            self.uses_pyg = False
        self.norm = nn.LayerNorm(hidden_dim)
        self.gating = ESGChannelGating(hidden_dim)
        self.heads = nn.ModuleDict({
            "esg_risk": nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1)),
            "economic_impact": nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1), nn.Tanh()),
            "volatility": nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1), nn.Softplus()),
        })

    def _conv(self, layer: nn.Module, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        if self.uses_pyg:
            return layer(x, edge_index, edge_weight=edge_weight)
        return layer(x, edge_index, edge_weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = F.relu(self.input_projection(self.input_norm(x)))
        up = self._conv(self.upstream, h, edge_index, edge_weight)
        down = self._conv(self.downstream, h, edge_index.flip(0), edge_weight)
        h = self.norm(F.relu(up + down + h))
        h, channel_scores = self.gating(h)
        out = {name: head(h).squeeze(-1) for name, head in self.heads.items()}
        out["node_embedding"] = h
        out["esg_channel_scores"] = channel_scores
        return out
