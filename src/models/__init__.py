"""Model package exposing PR 6 registry contracts plus legacy smoke models."""
from .contracts import CANONICAL_MODEL_NAMES, ModelCapabilities, ModelConfig, ModelOutput, ModelValidationError
from .heads import NodeRegressionHead
from .mlp import MLPBaseline
from .gcn import DirectedGCN
from .gat import DirectedGAT, WeightedDirectedGAT
from .directed import BidirectionalGNN
from .registry import available_models, build_model, get_model_capabilities
from .gnn_models import DenseGraphConv, EconomicESGGNN, ESGChannelGating, MeanScoreBaseline
__all__=["CANONICAL_MODEL_NAMES","ModelCapabilities","ModelConfig","ModelOutput","ModelValidationError","NodeRegressionHead","MLPBaseline","DirectedGCN","DirectedGAT","WeightedDirectedGAT","BidirectionalGNN","available_models","build_model","get_model_capabilities","MeanScoreBaseline","DenseGraphConv","ESGChannelGating","EconomicESGGNN"]
