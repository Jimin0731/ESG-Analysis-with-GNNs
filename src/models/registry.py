from __future__ import annotations
from collections import OrderedDict
import torch
from .contracts import CANONICAL_MODEL_NAMES, ModelCapabilities, ModelConfig, ModelValidationError
from .mlp import MLPBaseline
from .gcn import DirectedGCN
from .gat import DirectedGAT, WeightedDirectedGAT
from .directed import BidirectionalGNN

_REGISTRY: OrderedDict[str, tuple[type[torch.nn.Module], ModelCapabilities]] = OrderedDict()
def _register(name, cls, cap):
    if name in _REGISTRY: raise ModelValidationError(f"duplicate model registration: {name}")
    _REGISTRY[name]=(cls,cap)
_register("mlp",MLPBaseline,ModelCapabilities("mlp",False,False,False,False,False,True))
_register("gcn",DirectedGCN,ModelCapabilities("gcn",True,True,False,False,False,True))
_register("gat",DirectedGAT,ModelCapabilities("gat",True,True,False,True,False,True))
_register("weighted_gat",WeightedDirectedGAT,ModelCapabilities("weighted_gat",True,True,True,True,False,True))
_register("bidirectional_gnn",BidirectionalGNN,ModelCapabilities("bidirectional_gnn",True,True,False,False,True,True,"concat_projection"))
assert tuple(_REGISTRY) == CANONICAL_MODEL_NAMES

def available_models() -> tuple[str,...]: return tuple(_REGISTRY)
def get_model_capabilities(model_name: str) -> ModelCapabilities:
    if model_name not in _REGISTRY: raise ModelValidationError(f"unknown model: {model_name}")
    return _REGISTRY[model_name][1]
def build_model(config: ModelConfig):
    if config.name not in _REGISTRY: raise ModelValidationError(f"unknown model: {config.name}")
    cls=_REGISTRY[config.name][0]
    if config.seed is None: return cls(config)
    devices=[]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        return cls(config)
__all__=["available_models","build_model","get_model_capabilities"]
