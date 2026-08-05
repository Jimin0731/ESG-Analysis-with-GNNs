from __future__ import annotations
from torch import nn
from .contracts import ModelConfig, ModelOutput
from .validation import validate_features, validate_graph
from .heads import NodeRegressionHead
from .layers import DirectedGCNLayer, activation
class DirectedGCN(nn.Module):
    """Direction-preserving GCN: h'_i = W_self h_i + W_msg mean_{j -> i}(h_j)."""
    def __init__(self, config: ModelConfig):
        super().__init__(); self.config=config; self.input=nn.Linear(config.input_dim,config.hidden_dim); self.layers=nn.ModuleList([DirectedGCNLayer(config.hidden_dim) for _ in range(config.num_layers)]); self.act=activation(config.activation); self.drop=nn.Dropout(config.dropout); self.head=NodeRegressionHead(config.hidden_dim, config.target_names)
    def forward(self,x,edge_index=None,edge_weight=None,*,return_aux=False):
        validate_features(x,self.config.input_dim); edge_index,_=validate_graph(x,edge_index,edge_weight); h=self.act(self.input(x))
        for layer in self.layers: h=self.drop(self.act(layer(h,edge_index)))
        return ModelOutput(self.head(h), self.config.target_names, h, {}).validate()
__all__=["DirectedGCN"]
