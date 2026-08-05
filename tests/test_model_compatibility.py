import torch
from src.models.gnn_models import MeanScoreBaseline, DenseGraphConv, ESGChannelGating, EconomicESGGNN
def test_legacy_imports_and_keys():
    assert MeanScoreBaseline().fit([1,2]).predict(2)==[1.5,1.5]
    m=EconomicESGGNN(4,hidden_dim=8,dropout=0.0); out=m(torch.randn(3,4),torch.tensor([[0,1],[1,2]]),torch.ones(2))
    assert {"esg_risk","economic_impact","volatility","node_embedding","esg_channel_scores"} <= set(out)
    assert DenseGraphConv and ESGChannelGating
