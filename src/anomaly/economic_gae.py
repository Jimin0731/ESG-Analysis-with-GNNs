"""Separate unsupervised EconomicGAE; no supervised targets or task head."""
import torch
from torch import nn
from torch.nn import functional as F
from .contracts import AnomalyValidationError,EconomicGAEConfig,EconomicGAEOutput,EconomicGAELoss

def _graph(x,edge_index,edge_weight):
    if edge_index is None: edge_index=torch.empty((2,0),dtype=torch.long,device=x.device)
    if edge_index.ndim!=2 or edge_index.shape[0]!=2 or edge_index.dtype not in (torch.int32,torch.int64): raise AnomalyValidationError("edge_index must have shape [2, E]")
    if edge_index.numel() and (edge_index.min()<0 or edge_index.max()>=x.shape[0]): raise AnomalyValidationError("edge endpoint out of range")
    if edge_weight is not None:
        if edge_weight.ndim!=1 or edge_weight.numel()!=edge_index.shape[1]: raise AnomalyValidationError("edge weight length mismatch")
        if not edge_weight.is_floating_point() or not torch.isfinite(edge_weight).all() or (edge_weight<0).any(): raise AnomalyValidationError("edge weights must be finite and non-negative")
    return edge_index,edge_weight
def _aggregate(h,edges,weights,self_weight):
    source,target=edges; n=h.shape[0]; w=torch.ones(source.numel(),device=h.device,dtype=h.dtype) if weights is None else weights.to(h.dtype)
    total=h.new_full((n,),float(self_weight)); out=h*float(self_weight)
    total.index_add_(0,target,w); out.index_add_(0,target,h[source]*w[:,None])
    return out/total.clamp_min(torch.finfo(h.dtype).tiny)[:,None]
class EconomicGAE(nn.Module):
    def __init__(self,config:EconomicGAEConfig):
        super().__init__(); self.config=config
        self.hidden=nn.Linear(config.input_dim,config.hidden_dim); self.latent_layer=nn.Linear(config.hidden_dim,config.latent_dim); self.dropout=nn.Dropout(config.dropout)
        self.feature_decoder=nn.Sequential(nn.Linear(config.latent_dim,config.hidden_dim),nn.ReLU(),nn.Dropout(config.dropout),nn.Linear(config.hidden_dim,config.input_dim))
        self.source_projection=nn.Linear(config.latent_dim,config.latent_dim,bias=False); self.target_projection=nn.Linear(config.latent_dim,config.latent_dim,bias=False)
    def encode(self,x,edge_index=None,edge_weight=None):
        edges,weights=_graph(x,edge_index,edge_weight)
        if self.config.graph_direction=="reverse": edges=edges.flip(0)
        h=F.relu(self.hidden(x)); h=self.dropout(_aggregate(h,edges,weights,self.config.self_loop_weight))
        return self.latent_layer(_aggregate(h,edges,weights,self.config.self_loop_weight))
    def decode_structure(self,z,edge_pairs):
        if edge_pairs is None: edge_pairs=torch.empty((2,0),dtype=torch.long,device=z.device)
        return (self.source_projection(z)[edge_pairs[0]]*self.target_projection(z)[edge_pairs[1]]).sum(1)
    def forward(self,x,edge_index=None,edge_weight=None,negative_edge_index=None):
        z=self.encode(x,edge_index,edge_weight); reconstructed=self.feature_decoder(z)
        edges,_=_graph(x,edge_index,edge_weight)
        return EconomicGAEOutput(z,reconstructed,self.decode_structure(z,edges),self.decode_structure(z,negative_edge_index))
def reconstruction_loss(x,output,config):
    feature=F.mse_loss(output.reconstructed_x,x)
    logits=torch.cat((output.positive_edge_logits,output.negative_edge_logits)); labels=torch.cat((torch.ones_like(output.positive_edge_logits),torch.zeros_like(output.negative_edge_logits)))
    structure=F.binary_cross_entropy_with_logits(logits,labels) if logits.numel() else logits.sum()
    total=config.feature_loss_weight*feature+config.structure_loss_weight*structure
    return EconomicGAELoss(total,feature,structure,config.feature_loss_weight,config.structure_loss_weight)
__all__=["EconomicGAE","reconstruction_loss"]
