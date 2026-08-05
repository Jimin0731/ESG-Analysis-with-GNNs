from __future__ import annotations
from torch import nn
from .contracts import ModelConfig, ModelOutput
from .validation import validate_features, validate_graph
from .heads import NodeRegressionHead
from .layers import DirectedGATLayer, activation
class DirectedGAT(nn.Module):
    """Unweighted directed GAT; attention coefficients are [E, H] and normalized per target node/head."""
    def __init__(self, config: ModelConfig, *, weighted: bool=False):
        super().__init__(); self.config=config; self.weighted=weighted; heads=int(config.attention_heads or 1); self.input=nn.Linear(config.input_dim,config.hidden_dim); self.layers=nn.ModuleList([DirectedGATLayer(config.hidden_dim,heads,config.dropout,weighted=weighted) for _ in range(config.num_layers)]); self.act=activation(config.activation); self.drop=nn.Dropout(config.dropout); self.head=NodeRegressionHead(config.hidden_dim, config.target_names); self.heads=heads
    def forward(self,x,edge_index=None,edge_weight=None,*,return_aux=False):
        validate_features(x,self.config.input_dim); edge_index,edge_weight=validate_graph(x,edge_index,edge_weight,require_weights=self.weighted,non_negative_weights=self.weighted)
        h=self.act(self.input(x)); alpha=None
        for layer in self.layers: h,alpha=layer(h,edge_index,edge_weight); h=self.drop(self.act(h))
        aux={}
        if return_aux: aux={"edge_index":edge_index,"attention_coefficients":alpha,"attention_heads":self.heads}
        return ModelOutput(self.head(h), self.config.target_names, h, aux).validate()
class WeightedDirectedGAT(DirectedGAT):
    """Weighted GAT: exp(stabilized_logit(edge, head)) * edge_weight(edge), normalized by incoming target/head mass."""
    def __init__(self, config: ModelConfig): super().__init__(config, weighted=True)
__all__=["DirectedGAT","WeightedDirectedGAT"]
