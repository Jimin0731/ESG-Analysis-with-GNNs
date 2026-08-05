import torch, pytest
from src.models import ModelConfig, build_model, ModelValidationError
T=("target__observed_esg_score",)
def cfg(): return ModelConfig(name="bidirectional_gnn",input_dim=3,hidden_dim=6,num_layers=1,dropout=0.0,target_names=T,seed=5)
def fixture(): return torch.eye(3), torch.tensor([[0,1,2,0],[2,2,0,1]]), torch.tensor([1.,4.,2.,3.])
def test_bidirectional_aux_separate_params_and_direction():
    m=build_model(cfg()); m.eval(); x,ei,ew=fixture(); out=m(x,ei,ew,return_aux=True); rev=m(x,ei.flip(0),ew,return_aux=True)
    assert not any(a is b for a,b in zip(m.forward_layers.parameters(), m.reverse_layers.parameters()))
    assert out.auxiliary["forward_embedding"].shape==(3,6); assert out.auxiliary["reverse_embedding"].shape==(3,6); assert out.auxiliary["combination"]=="concat_projection"
    assert not torch.allclose(out.auxiliary["forward_embedding"], rev.auxiliary["forward_embedding"])
    assert torch.allclose(out.predictions, m(x,ei,ew,return_aux=True).predictions)
def test_bidirectional_uses_aligned_edge_weights_and_stays_finite():
    m=build_model(cfg()); m.eval(); x,ei,ew=fixture(); base=m(x,ei,ew,return_aux=True); changed=m(x,ei,torch.tensor([5.,1.,2.,3.]),return_aux=True)
    assert not torch.allclose(base.auxiliary["forward_embedding"], changed.auxiliary["forward_embedding"])
    assert not torch.allclose(base.predictions, changed.predictions)
    perm=torch.tensor([2,0,3,1]); assert torch.allclose(base.predictions, m(x,ei[:,perm],ew[perm]).predictions, atol=1e-6)
    assert not torch.allclose(base.predictions, m(x,ei,ew[perm]).predictions)
    assert torch.isfinite(m(x,ei,torch.zeros_like(ew)).predictions).all()
    empty=torch.empty((2,0),dtype=torch.long); assert torch.isfinite(m(x,empty,None).predictions).all()
    with pytest.raises(ModelValidationError): m(x,ei,torch.tensor([1.,-1.,1.,1.]))
    with pytest.raises(ModelValidationError): m(x,ei,torch.tensor([1.,float('nan'),1.,1.]))
def test_bidirectional_requires_weights_for_non_empty_graph_and_reverse_alignment():
    m=build_model(cfg()); m.eval(); x,ei,ew=fixture()
    with pytest.raises(ModelValidationError): m(x,ei,None)
    out=m(x,ei,ew,return_aux=True)
    manual_reverse=m._branch(m.act(m.r_in(x)), ei.flip(0), ew, m.reverse_layers)
    assert torch.allclose(out.auxiliary["reverse_embedding"], manual_reverse)
