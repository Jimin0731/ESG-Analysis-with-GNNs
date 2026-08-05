from __future__ import annotations
from torch import nn
from .contracts import ModelConfig, ModelOutput
from .validation import validate_features
from .heads import NodeRegressionHead
from .layers import activation
class MLPBaseline(nn.Module):
    """Required graph-independent baseline for fair graph-model comparison."""
    def __init__(self, config: ModelConfig):
        super().__init__(); self.config=config; layers=[]; d=config.input_dim
        for _ in range(config.num_layers): layers += [nn.Linear(d,config.hidden_dim), activation(config.activation), nn.Dropout(config.dropout)]; d=config.hidden_dim
        self.encoder=nn.Sequential(*layers); self.head=NodeRegressionHead(config.hidden_dim, config.target_names)
    def forward(self,x,edge_index=None,edge_weight=None,*,return_aux=False):
        validate_features(x,self.config.input_dim); h=self.encoder(x); return ModelOutput(self.head(h), self.config.target_names, h, {}).validate()
__all__=["MLPBaseline"]
