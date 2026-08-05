import torch,pytest
from src.anomaly import *
def cfg(**kw): d=dict(input_dim=3,hidden_dim=4,latent_dim=2,dropout=0.); d.update(kw); return EconomicGAEConfig(**d)
def data(): return torch.arange(12,dtype=torch.float).reshape(4,3)/10,torch.tensor([[0,0,1],[1,2,2]]),torch.tensor([1.,3.,2.])
def test_shapes_loss_gradients_and_score_formula():
 x,e,w=data(); n=sample_directed_negative_edges(e,4,3,seed=4); m=EconomicGAE(cfg()); o=m(x,e,w,n)
 assert o.latent.shape==(4,2) and o.reconstructed_x.shape==x.shape and o.positive_edge_logits.shape==(3,) and o.negative_edge_logits.shape==(3,)
 loss=reconstruction_loss(x,o,m.config); assert torch.isfinite(loss.total) and torch.allclose(loss.total,loss.feature_reconstruction+loss.structure_reconstruction)
 loss.total.backward(); assert all(p.grad is not None for p in (m.hidden.weight,m.feature_decoder[-1].weight,m.source_projection.weight,m.target_projection.weight))
 score=score_anomalies(m,x,e,w); assert torch.allclose(score.anomaly_scores,score.feature_reconstruction_error+.5*score.embedding_consistency_error)
 assert m.config.anomaly_structural_coefficient==.5 and not hasattr(score,"threshold")
def test_directed_decoder_and_sampling():
 m=EconomicGAE(cfg()); z=torch.tensor([[1.,2.],[3.,4.]])
 with torch.no_grad(): m.source_projection.weight.copy_(torch.eye(2)); m.target_projection.weight.copy_(torch.tensor([[1.,2.],[0.,1.]]))
 logits=m.decode_structure(z,torch.tensor([[0,1],[1,0]])); assert logits[0]!=logits[1]
 e=torch.tensor([[0,1],[1,0]]); a=sample_directed_negative_edges(e,3,3,seed=9); b=sample_directed_negative_edges(e,3,3,seed=9)
 assert torch.equal(a,b) and len(set(map(tuple,a.t().tolist())))==3 and not set(map(tuple,a.t().tolist()))&set(map(tuple,e.t().tolist()))
 with pytest.raises(AnomalyValidationError): sample_directed_negative_edges(e,2,1)
def test_graph_direction_alignment_empty_and_invalid_weights():
 x,e,w=data(); a=EconomicGAE(cfg(graph_direction="stored")); b=EconomicGAE(cfg(graph_direction="reverse")); b.load_state_dict(a.state_dict())
 assert not torch.allclose(a.encode(x,e,w),b.encode(x,e,w)); p=torch.tensor([2,0,1]); assert torch.allclose(a.encode(x,e,w),a.encode(x,e[:,p],w[p]))
 assert not torch.allclose(a.encode(x,e,w),a.encode(x,e,w[p])); assert torch.isfinite(a.encode(x,torch.empty((2,0),dtype=torch.long))).all()
 for bad in (torch.tensor([-1.,1.,1.]),torch.tensor([float("nan"),1.,1.]),torch.tensor([float("inf"),1.,1.])):
  with pytest.raises(AnomalyValidationError): a.encode(x,e,bad)
