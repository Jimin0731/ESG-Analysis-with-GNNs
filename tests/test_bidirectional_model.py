import torch
from src.models import ModelConfig, build_model
T=("target__observed_esg_score",)
def test_bidirectional_aux_separate_params_and_direction():
    c=ModelConfig(name="bidirectional_gnn",input_dim=3,hidden_dim=6,num_layers=1,dropout=0.0,target_names=T,seed=5)
    m=build_model(c); m.eval(); x=torch.eye(3); ei=torch.tensor([[0,1],[1,2]]); ew=torch.tensor([1.,5.])
    assert not any(a is b for a,b in zip(m.forward_layers.parameters(), m.reverse_layers.parameters()))
    out=m(x,ei,ew,return_aux=True); rev=m(x,ei.flip(0),ew,return_aux=True)
    assert out.auxiliary["forward_embedding"].shape==(3,6); assert out.auxiliary["reverse_embedding"].shape==(3,6); assert out.auxiliary["combination"]=="concat_projection"
    assert not torch.allclose(out.auxiliary["forward_embedding"], rev.auxiliary["forward_embedding"])
    assert torch.allclose(out.predictions, m(x,ei,ew,return_aux=True).predictions)
