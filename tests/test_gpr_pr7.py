import torch,pytest
from src.models import ModelConfig,ModelOutput,ModelValidationError,available_models,build_model,get_model_capabilities
from src.models.ablations import build_gpr_ablation_configs
T=("target__a","target__b")
def cfg(**kw):
 d=dict(name="gpr_gnn",input_dim=2,hidden_dim=4,num_layers=1,dropout=0.,target_names=T,seed=7,propagation_steps=2); d.update(kw); return ModelConfig(**d)
def data(): return torch.tensor([[1.,0.],[0.,1.],[2.,1.]]),torch.tensor([[0,0,1],[1,2,2]]),torch.tensor([1.,3.,2.])
def test_registry_contract_and_auxiliary():
 assert available_models()[-1]=="gpr_gnn"; assert get_model_capabilities("gpr_gnn").uses_edge_weight
 x,e,w=data(); o=build_model(cfg())(x,e,w,return_aux=True)
 assert isinstance(o,ModelOutput) and o.predictions.shape==(3,2) and o.node_embeddings.shape==(3,4) and o.target_names==T
 c=o.auxiliary["propagation_coefficients"]; assert (c>=0).all() and torch.allclose(c.sum(),torch.tensor(1.))
def test_k_zero_graph_independent_and_alpha_one_restart_only():
 x,e,w=data(); m=build_model(cfg(propagation_steps=0)); assert torch.equal(m(x,e,w).node_embeddings,m(x,e.flip(0),w.flip(0)).node_embeddings)
 m=build_model(cfg(alpha=1.)); o=m(x,e,w,return_aux=True); assert torch.allclose(o.node_embeddings,o.auxiliary["final_propagated_embedding"])
 assert torch.allclose(o.node_embeddings,m.encoder(x))
def test_direction_alignment_weight_effects_and_finite_edges():
 x,e,w=data(); a=build_model(cfg(graph_direction="stored")); b=build_model(cfg(graph_direction="reverse")); b.load_state_dict(a.state_dict())
 assert not torch.allclose(a(x,e,w).node_embeddings,b(x,e,w).node_embeddings)
 p=torch.tensor([2,0,1]); assert torch.allclose(a(x,e,w).node_embeddings,a(x,e[:,p],w[p]).node_embeddings)
 assert not torch.allclose(a(x,e,w).node_embeddings,a(x,e,w[p]).node_embeddings)
 for edges,weights in [(torch.empty((2,0),dtype=torch.long),torch.empty(0)),(e,torch.zeros(3))]: assert torch.isfinite(a(x,edges,weights).node_embeddings).all()
def test_config_seed_and_gradients():
 for bad in ({"alpha":float("nan")},{"alpha":2},{"propagation_steps":True},{"propagation_steps":-1},{"graph_direction":"both"}):
  with pytest.raises(ModelValidationError): cfg(**bad)
 a,b=build_model(cfg()),build_model(cfg()); assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
 x,e,w=data(); build_model(cfg())(x,e,w).predictions.sum().backward(); m=build_model(cfg()); m(x,e,w).predictions.sum().backward()
 assert m.gamma.grad is not None and m.encoder[0].weight.grad is not None and m.head.projection.weight.grad is not None
def test_ablation_order_validation_and_independence():
 base=cfg(); specs=build_gpr_ablation_configs(base,alphas=[.1,.2],propagation_steps=[0,2],graph_directions=["stored","reverse"])
 assert len(specs)==8 and [(s.alpha,s.propagation_steps,s.graph_direction) for s in specs][:4]==[(.1,0,"stored"),(.1,0,"reverse"),(.1,2,"stored"),(.1,2,"reverse")]
 assert len({s.name for s in specs})==8 and base.propagation_steps==2 and build_model(specs[0].config) is not build_model(specs[0].config)
 for args in [dict(alphas=[],propagation_steps=[1],graph_directions=["stored"]),dict(alphas=[.1,.1],propagation_steps=[1],graph_directions=["stored"]),dict(alphas=[.1],propagation_steps=[-1],graph_directions=["stored"])]:
  with pytest.raises(ModelValidationError): build_gpr_ablation_configs(base,**args)
