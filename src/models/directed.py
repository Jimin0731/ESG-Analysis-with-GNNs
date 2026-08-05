from __future__ import annotations
import torch
from torch import nn
from .contracts import ModelConfig, ModelOutput
from .validation import validate_features, validate_graph
from .heads import NodeRegressionHead
from .layers import DirectedGCNLayer, activation
class BidirectionalGNN(nn.Module):
    """Separately parameterized stored-direction and edge_index.flip(0) branches combined by concat projection."""
    def __init__(self, config: ModelConfig):
        super().__init__(); self.config=config; self.combination=str(config.options.get("combination","concat_projection")); self.f_in=nn.Linear(config.input_dim,config.hidden_dim); self.r_in=nn.Linear(config.input_dim,config.hidden_dim); self.forward_layers=nn.ModuleList([DirectedGCNLayer(config.hidden_dim) for _ in range(config.num_layers)]); self.reverse_layers=nn.ModuleList([DirectedGCNLayer(config.hidden_dim) for _ in range(config.num_layers)]); self.project=nn.Linear(config.hidden_dim*2,config.hidden_dim); self.act=activation(config.activation); self.drop=nn.Dropout(config.dropout); self.head=NodeRegressionHead(config.hidden_dim,config.target_names)
    def _branch(self,h,edge_index,layers):
        for layer in layers: h=self.drop(self.act(layer(h,edge_index)))
        return h
    def forward(self,x,edge_index=None,edge_weight=None,*,return_aux=False):
        validate_features(x,self.config.input_dim); edge_index,_=validate_graph(x,edge_index,edge_weight)
        f=self._branch(self.act(self.f_in(x)),edge_index,self.forward_layers); r=self._branch(self.act(self.r_in(x)),edge_index.flip(0),self.reverse_layers); c=self.act(self.project(torch.cat([f,r],dim=-1)))
        aux={}
        if return_aux: aux={"forward_embedding":f,"reverse_embedding":r,"combined_embedding":c,"combination":"concat_projection"}
        return ModelOutput(self.head(c), self.config.target_names, c, aux).validate()
__all__=["BidirectionalGNN"]
