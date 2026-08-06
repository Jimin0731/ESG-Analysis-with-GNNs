import json
import matplotlib
matplotlib.use("Agg")
import pytest,torch
from src.models import ModelConfig,build_model,available_models
from src.experiments.contracts import SupervisedSnapshot
from src.evaluation import (InterpretabilityValidationError,build_shock_explanation,extract_attention_explanation,extract_propagation_explanation,run_edge_ablation,run_feature_ablation,run_node_feature_perturbation,trace_attention_paths)
from src.evaluation.explanation_contracts import AttentionEdgeRecord,AttentionExplanation,NON_CAUSAL_WARNINGS
from src.visualization import export_explanation_bundle,plot_attention_edges

def fixture(model="weighted_gat",weights=(1.,1.,1.)):
    kw={"attention_heads":2} if "gat" in model else {}
    config=ModelConfig(model,2,4,2,0.,("target__a","target__b"),seed=3,**kw)
    snapshot=SupervisedSnapshot(2020,"validation",("a","b","c"),torch.tensor([[1.,2.],[3.,4.],[5.,6.]]),torch.zeros(3,2),torch.ones(3,2,dtype=torch.bool),torch.tensor([[0,0,1],[1,2,2]]),torch.tensor(weights),("safe__x","safe__unused"),("target__a","target__b"))
    return build_model(config),snapshot

def test_gat_auxiliary_layers_and_legacy_keys():
    model,s=fixture(); plain=model(s.x,s.edge_index,s.edge_weight); out=model(s.x,s.edge_index,s.edge_weight,return_aux=True)
    assert not plain.auxiliary and {"edge_index","attention_coefficients","attention_heads"}<=out.auxiliary.keys()
    layers=out.auxiliary["attention_by_layer"]; assert len(layers)==2 and all(x.shape==(3,2) for x in layers); assert torch.equal(out.auxiliary["attention_coefficients"],layers[-1])

def test_attention_extraction_normalization_order_heads_and_top_k():
    model,s=fixture(); explanation=extract_attention_explanation(model,s)
    assert explanation.normalization_valid and {r.edge_position for r in explanation.edges}=={0,1,2}
    with torch.no_grad(): raw=model(s.x,s.edge_index,s.edge_weight,return_aux=True).auxiliary["attention_coefficients"]
    assert explanation.edges[0].aggregated_attention_coefficient==pytest.approx(float(raw[explanation.edges[0].edge_position].mean()))
    assert extract_attention_explanation(model,s,head_aggregation="max").edges
    assert extract_attention_explanation(model,s,head_aggregation=1,top_k=1).edges[0].edge_position in range(3)

def test_invalid_attention_and_unsupported_models():
    _,s=fixture()
    for name in ("mlp","gcn","bidirectional_gnn","gpr_gnn"):
        config=ModelConfig(name,2,4,1,0.,s.target_names,seed=1)
        with pytest.raises(InterpretabilityValidationError): extract_attention_explanation(build_model(config),s)
    model,_=fixture()
    with pytest.raises(InterpretabilityValidationError): extract_attention_explanation(model,s,layer_index=9)
    with pytest.raises(InterpretabilityValidationError): extract_attention_explanation(model,s,head_aggregation=9)

def test_weighted_zero_mass_is_finite():
    model,s=fixture(weights=(1.,0.,0.)); explanation=extract_attention_explanation(model,s)
    assert all(torch.isfinite(torch.tensor(r.attention_coefficient)) for r in explanation.heads)

def test_perturbation_nonmutation_mode_and_target_order():
    model,s=fixture(); model.train(); before=s.x.clone(); params={k:v.clone() for k,v in model.state_dict().items()}
    report=run_node_feature_perturbation(model,s,source_node_id="a",feature_changes={"safe__x":.5},target_names=("target__b","target__a"))
    assert model.training and torch.equal(s.x,before) and all(torch.equal(v,params[k]) for k,v in model.state_dict().items())
    assert report.target_names==("target__b","target__a") and report.feature_changes[0][2:]==(1.,1.5)

def test_feature_and_edge_ablation_nonmutation_and_alignment():
    model,s=fixture(); edge=s.edge_index.clone(); weights=s.edge_weight.clone()
    feature=run_feature_ablation(model,s,feature_names=("safe__x",)); edge_report=run_edge_ablation(model,s,edge_positions=(1,),mode="remove")
    assert feature.label=="one-at-a-time feature ablation sensitivity" and edge_report.records[0].economic_edge_weights==(1.,)
    assert torch.equal(edge,s.edge_index) and torch.equal(weights,s.edge_weight)
    unweighted,_=fixture("gat")
    with pytest.raises(InterpretabilityValidationError): run_edge_ablation(unweighted,s,edge_positions=(0,),mode="zero_weight")

def test_hand_constructed_directed_path_product_and_no_reverse():
    edges=(AttentionEdgeRecord(0,0,"a",1,"b",0,.5,None),AttentionEdgeRecord(1,1,"b",2,"c",0,.4,None),AttentionEdgeRecord(2,0,"a",2,"c",0,.1,None),AttentionEdgeRecord(3,2,"c",0,"a",0,.9,None))
    explanation=AttentionExplanation("gat",2020,"validation","source_to_target",0,"mean","incoming_target_per_head",True,(),edges,"rule",NON_CAUSAL_WARNINGS)
    paths=trace_attention_paths(explanation,source_node_id="a",max_hops=2,top_k=5)
    c=next(r for r in paths.records if r.destination_node_id=="c"); assert c.attention_path_score==pytest.approx(.2) and c.path_edge_positions==(0,1)
    reverse=trace_attention_paths(explanation,source_node_id="b",max_hops=1,top_k=5); assert all(r.destination_node_id!="a" for r in reverse.records)

def test_export_json_csv_figures(tmp_path):
    model,s=fixture(); bundle=build_shock_explanation(model,s,source_node_id="a",feature_changes={"safe__x":.2},feature_ablation=("safe__x",),edge_ablation={"edge_positions":(0,),"mode":"remove"})
    assert list(tmp_path.iterdir())==[] and plot_attention_edges(bundle.attention).__class__.__name__=="Figure"
    manifest=export_explanation_bundle(bundle,tmp_path); assert len(manifest.generated_files)==len(set(manifest.generated_files)); json.loads((tmp_path/"explanation.json").read_text())
    assert all((tmp_path/name).stat().st_size for name in manifest.generated_files)
    with pytest.raises(InterpretabilityValidationError): export_explanation_bundle(bundle,tmp_path/"bad",figure_format="pdf")

def test_registry_unchanged(): assert available_models()==("mlp","gcn","gat","weighted_gat","bidirectional_gnn","gpr_gnn")

@pytest.mark.parametrize("steps",[0,3])
def test_gpr_coefficients_are_distinct_normalized_propagation(steps):
    _,s=fixture(); config=ModelConfig("gpr_gnn",2,4,1,0.,s.target_names,seed=2,propagation_steps=steps)
    model=build_model(config); report=extract_propagation_explanation(model,s)
    assert len(report.records)==steps+1 and sum(r.coefficient for r in report.records)==pytest.approx(1.)
    assert all(r.coefficient>=0 for r in report.records) and "not edge attention" in report.statement

def test_mlp_perturbation_is_node_local():
    model,s=fixture("mlp"); report=run_node_feature_perturbation(model,s,source_node_id="a",feature_changes={"safe__x":1.})
    assert all(r.prediction_delta==pytest.approx(0.) for r in report.records if r.node_id!="a")
