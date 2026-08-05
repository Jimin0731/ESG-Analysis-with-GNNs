from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def activation(name: str):
    return {"relu":nn.ReLU(),"gelu":nn.GELU(),"tanh":nn.Tanh(),"identity":nn.Identity()}[name]

class DirectedGCNLayer(nn.Module):
    """h'_i = W_self h_i + W_msg weighted_mean_{j -> i}(h_j); stored edges are never symmetrized."""
    def __init__(self, dim: int): super().__init__(); self.self_linear=nn.Linear(dim,dim); self.msg_linear=nn.Linear(dim,dim)
    def forward(self,x,edge_index,edge_weight=None):
        src,dst=edge_index; agg=torch.zeros_like(x)
        if src.numel():
            weights = torch.ones(src.shape[0], device=x.device, dtype=x.dtype) if edge_weight is None else edge_weight.to(x.dtype)
            agg.index_add_(0,dst,x[src] * weights.unsqueeze(-1))
            deg=torch.zeros(x.shape[0],device=x.device,dtype=x.dtype); deg.index_add_(0,dst,weights)
            agg=torch.where(deg.unsqueeze(-1)>0, agg/deg.clamp_min(torch.finfo(x.dtype).tiny).unsqueeze(-1), torch.zeros_like(agg))
        return self.self_linear(x)+self.msg_linear(agg)

class DirectedGATLayer(nn.Module):
    """Pure PyTorch directed multi-head attention. Returned attention has shape [E, H]."""
    def __init__(self, dim:int, heads:int, dropout:float, weighted:bool=False):
        super().__init__(); self.dim=dim; self.heads=heads; self.head_dim=dim//heads; self.weighted=weighted; self.dropout=nn.Dropout(dropout)
        self.linear=nn.Linear(dim,dim,bias=False); self.self_linear=nn.Linear(dim,dim); self.attn_src=nn.Parameter(torch.empty(heads,self.head_dim)); self.attn_dst=nn.Parameter(torch.empty(heads,self.head_dim)); nn.init.xavier_uniform_(self.attn_src); nn.init.xavier_uniform_(self.attn_dst)
    def forward(self,x,edge_index,edge_weight=None):
        N=x.shape[0]; src,dst=edge_index; z=self.linear(x).view(N,self.heads,self.head_dim); out=torch.zeros(N,self.heads,self.head_dim,device=x.device,dtype=x.dtype); alpha=torch.zeros(src.shape[0],self.heads,device=x.device,dtype=x.dtype)
        if src.numel():
            logits=(z[src]*self.attn_src).sum(-1)+(z[dst]*self.attn_dst).sum(-1); logits=F.leaky_relu(logits,0.2)
            for h in range(self.heads):
                for node in torch.unique(dst):
                    mask=dst==node; vals=logits[mask,h]; vals=vals-vals.max(); unnorm=torch.exp(vals)
                    if self.weighted: unnorm=unnorm*edge_weight[mask].to(x.dtype)
                    denom=unnorm.sum(); alpha[mask,h]=torch.where(denom>0, unnorm/denom.clamp_min(torch.finfo(x.dtype).tiny), torch.zeros_like(unnorm))
            msg=z[src]*self.dropout(alpha).unsqueeze(-1); out.index_add_(0,dst,msg)
        return self.self_linear(x)+out.reshape(N,self.dim), alpha
__all__=["activation","DirectedGCNLayer","DirectedGATLayer"]
