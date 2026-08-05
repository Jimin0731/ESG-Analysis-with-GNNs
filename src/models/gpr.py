"""Pure-PyTorch generalized PageRank model for directed economic graphs."""
from __future__ import annotations
import torch
from torch import nn
from .contracts import ModelConfig, ModelOutput
from .heads import NodeRegressionHead
from .layers import activation
from .validation import validate_features, validate_graph

def weighted_incoming(h, edge_index, edge_weight, self_loop_weight):
    n=h.shape[0]; source,target=edge_index
    weights=torch.ones(source.numel(),device=h.device,dtype=h.dtype) if edge_weight is None else edge_weight.to(h.dtype)
    present=torch.zeros(n,dtype=torch.bool,device=h.device); loops=source==target
    if loops.any(): present[target[loops]]=True
    missing=torch.arange(n,device=h.device)[~present]
    source=torch.cat((source,missing)); target=torch.cat((target,missing))
    weights=torch.cat((weights,torch.full((missing.numel(),),float(self_loop_weight),device=h.device,dtype=h.dtype)))
    total=h.new_zeros(n); total.index_add_(0,target,weights)
    result=h.new_zeros(h.shape); result.index_add_(0,target,h[source]*weights[:,None])
    return result/total.clamp_min(torch.finfo(h.dtype).tiny)[:,None]

class GPRGNN(nn.Module):
    def __init__(self,config):
        super().__init__(); self.config=config
        self.encoder=nn.Sequential(nn.Linear(config.input_dim,config.hidden_dim),activation(config.activation),nn.Dropout(config.dropout))
        self.gamma=nn.Parameter(torch.zeros(config.propagation_steps+1)); self.head=NodeRegressionHead(config.hidden_dim,config.target_names)
    def forward(self,x,edge_index=None,edge_weight=None,*,return_aux=False):
        validate_features(x,self.config.input_dim); edge_index,edge_weight=validate_graph(x,edge_index,edge_weight,non_negative_weights=True)
        if self.config.graph_direction=="reverse": edge_index=edge_index.flip(0)
        h0=self.encoder(x); h=h0; states=[h0]
        for _ in range(self.config.propagation_steps):
            h=(1-self.config.alpha)*weighted_incoming(h,edge_index,edge_weight,self.config.self_loop_weight)+self.config.alpha*h0; states.append(h)
        coefficients=torch.softmax(self.gamma,0); embedding=sum(c*s for c,s in zip(coefficients,states)); aux={}
        if return_aux: aux={"propagation_coefficients":coefficients,"propagation_depth":self.config.propagation_steps,"alpha":self.config.alpha,"graph_direction":self.config.graph_direction,"final_propagated_embedding":embedding}
        return ModelOutput(self.head(embedding),self.config.target_names,embedding,aux).validate()
__all__=["GPRGNN","weighted_incoming"]
