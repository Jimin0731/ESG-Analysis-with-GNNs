import json, torch, pytest
from src.models import ModelConfig, ModelValidationError, available_models, build_model, get_model_capabilities
T=("target__observed_esg_score","target__observed_real_output_growth")
def cfg(name="mlp", **kw):
    d=dict(name=name,input_dim=3,hidden_dim=6,num_layers=2,dropout=0.0,target_names=T,seed=3)
    if name in {"gat","weighted_gat"}: d["attention_heads"]=2
    d.update(kw); return ModelConfig(**d)
def graph(): return torch.randn(4,3), torch.tensor([[0,1,2,0],[1,2,0,3]]), torch.tensor([1.,2.,0.,3.])
def test_registry_order_and_capabilities_json():
    assert available_models()==("mlp","gcn","gat","weighted_gat","bidirectional_gnn","gpr_gnn")
    for n in available_models(): json.dumps(get_model_capabilities(n).to_report())
    with pytest.raises(ModelValidationError): get_model_capabilities("unknown")
def test_config_validation():
    for bad in [dict(input_dim=0),dict(hidden_dim=True),dict(dropout=1.0),dict(target_names=("target__x","target__x")),dict(target_names=("x",))]:
        with pytest.raises(ModelValidationError): cfg(**bad)
    with pytest.raises(ModelValidationError): cfg("gat", attention_heads=4, hidden_dim=6)
def test_fresh_seeded_models_and_rng_stream():
    torch.manual_seed(123); before=torch.rand(3); m1=build_model(cfg()); after=torch.rand(3)
    torch.manual_seed(123); assert torch.allclose(torch.rand(3), before); _=build_model(cfg()); assert torch.allclose(torch.rand(3), after)
    m2=build_model(cfg()); assert all(torch.equal(a,b) for a,b in zip(m1.parameters(),m2.parameters()))
    m3=build_model(cfg(seed=4)); assert any(not torch.equal(a,b) for a,b in zip(m1.parameters(),m3.parameters()))
def test_every_model_output_and_gradients():
    x,ei,ew=graph()
    for n in available_models():
        m=build_model(cfg(n)); out=m(x,ei,ew,return_aux=False) if n!="weighted_gat" else m(x,ei,ew)
        assert out.predictions.shape==(4,2); assert out.node_embeddings.shape[0]==4; assert out.target_names==T; assert out.auxiliary=={}
        loss=out.predictions.sum(); loss.backward(); assert any(p.grad is not None for p in m.parameters() if p.requires_grad)
def test_input_validation_rejections():
    x,ei,ew=graph(); m=build_model(cfg("weighted_gat"))
    cases=[(x[:,0],ei,ew),(x[:,:2],ei,ew),(x.clone().fill_(float('nan')),ei,ew),(x,ei.float(),ew),(x,torch.tensor([[0],[-1]]),ew[:1]),(x,torch.tensor([[0],[9]]),ew[:1]),(x,ei,ew[:2]),(x,ei,torch.tensor([1.,float('nan'),1.,1.])),(x,ei,torch.tensor([1.,-1.,1.,1.]))]
    for a,b,c in cases:
        with pytest.raises(ModelValidationError): m(a,b,c)
