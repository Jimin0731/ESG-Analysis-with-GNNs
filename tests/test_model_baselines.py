import torch, pytest
from src.models import ModelConfig, build_model, ModelValidationError
T=("target__observed_esg_score",)
def cfg(n):
    d=dict(name=n,input_dim=3,hidden_dim=6,num_layers=2,dropout=0.0,target_names=T,seed=9)
    if n in {"gat","weighted_gat"}: d["attention_heads"]=2
    return ModelConfig(**d)
def data(): return torch.tensor([[1.,0,0],[0,1,0],[0,0,1],[1,1,0]]), torch.tensor([[0,1,2],[1,2,0]]), torch.tensor([1.,2.,3.])
def test_mlp_ignores_edges_and_weights():
    x,ei,ew=data(); m=build_model(cfg("mlp")); m.eval(); a=m(x,ei,ew).predictions; b=m(x,ei.flip(0),ew*9).predictions
    assert torch.allclose(a,b); assert not torch.allclose(a,m(x+0.1,ei,ew).predictions)
def test_gcn_direction_and_not_weights_and_isolated_finite():
    x,ei,ew=data(); m=build_model(cfg("gcn")); m.eval(); a=m(x,ei,ew).predictions; assert torch.allclose(a,m(x,ei,ew*8).predictions); assert not torch.allclose(a,m(x,ei.flip(0),ew).predictions)
    empty=torch.empty((2,0),dtype=torch.long); assert torch.isfinite(m(x,empty,None).predictions).all()
def test_gat_attention_and_unweighted_behavior():
    x,ei,ew=data(); m=build_model(cfg("gat")); m.eval(); out=m(x,ei,ew,return_aux=True); alpha=out.auxiliary["attention_coefficients"]
    assert alpha.shape==(ei.shape[1],2)
    for node in ei[1].unique(): assert torch.allclose(alpha[ei[1]==node].sum(0), torch.ones(2), atol=1e-6)
    assert torch.allclose(out.predictions,m(x,ei,ew*7).predictions)
    assert torch.isfinite(m(x,torch.empty((2,0),dtype=torch.long),None).predictions).all()
def test_weighted_gat_sensitivity_zero_and_reorder():
    x,ei,ew=data(); m=build_model(cfg("weighted_gat")); m.eval(); a=m(x,ei,ew,return_aux=True); b=m(x,ei,torch.tensor([9.,2.,3.]))
    assert not torch.allclose(a.predictions,b.predictions)
    z=m(x,ei,torch.tensor([0.,2.,3.]),return_aux=True).auxiliary["attention_coefficients"]; assert torch.all(z[0]==0)
    assert torch.isfinite(m(x,ei,torch.zeros(3)).predictions).all()
    perm=torch.tensor([2,0,1]); assert torch.allclose(a.predictions, m(x,ei[:,perm],ew[perm]).predictions, atol=1e-6)
    with pytest.raises(ModelValidationError): m(x,ei,torch.tensor([1.,-1.,1.]))
