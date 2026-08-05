from __future__ import annotations
import torch
from torch import nn
class NodeRegressionHead(nn.Module):
    """Ordered node regression head; identity transforms by default."""
    def __init__(self, embedding_dim: int, target_names: tuple[str,...]):
        super().__init__(); self.target_names=tuple(target_names); self.projection=nn.Linear(embedding_dim, len(self.target_names))
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor: return self.projection(embeddings)
__all__=["NodeRegressionHead"]
