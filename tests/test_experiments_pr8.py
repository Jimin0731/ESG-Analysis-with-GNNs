import math,torch,pytest,numpy as np,pandas as pd
from types import SimpleNamespace
from src.experiments import *
from src.experiments.supervised import _global_loss
from src.graphs import GraphNode,GraphSnapshot
from src.models import ModelConfig,available_models
from src.targets import LeakageAuditReport,TargetPanel,TargetProvenance
def snap(period,split,offset=0.):
    x=torch.tensor([[0.,1.],[1.,0.],[.5,.5]]); y=torch.tensor([[1.+offset,2.],[3.+offset,float("nan")],[2.+offset,4.]])
    return SupervisedSnapshot(period,split,("a","b","c"),x,y,~torch.isnan(y),torch.tensor([[0,1],[1,2]]),torch.tensor([.2,.7]),("safe__a","safe__b"),("target__one","target__two"))
def data(): return PreparedSupervisedData((snap(2018,"train"),snap(2019,"train"),snap(2020,"train")),(snap(2021,"validation"),),(snap(2022,"test"),),("safe__a","safe__b"),("target__one","target__two"))
def test_masked_loss_exact_and_missing_excluded():
    s=snap(1,"train"); pred=torch.zeros_like(s.y); r=masked_multi_target_mse(pred,s.y,s.target_observed_mask,s.target_names,{"target__one":1.,"target__two":3.})
    assert r.normalized_weights=={"target__one":.25,"target__two":.75}; assert r.observed_counts["target__two"]==2; assert r.total.item()==pytest.approx((14/3)*.25+10*.75)
def test_metrics_hand_calculated_and_undefined_r2():
    y=torch.tensor([[1.],[2.],[3.]]); p=torch.tensor([[1.],[3.],[2.]]); m=torch.ones_like(y,dtype=torch.bool); result=regression_metrics(y,p,m,("target__x",))[0]
    assert result.mae==pytest.approx(2/3); assert result.rmse==pytest.approx(math.sqrt(2/3)); assert result.r2==pytest.approx(0.)
    result=regression_metrics(torch.ones((2,1)),torch.ones((2,1)),m[:2],("target__x",))[0]; assert result.r2 is None and result.r2_undefined_reason
def test_baseline_is_train_only():
    d=data(); first=TrainMeanBaseline().fit(d.train); changed=PreparedSupervisedData(d.train,(snap(2021,"validation",1000),),(snap(2022,"test",-1000),),d.feature_names,d.target_names); second=TrainMeanBaseline().fit(changed.train)
    assert first.means==second.means
def test_seeded_training_checkpoint_and_no_test_history():
    d=data(); cfg=ModelConfig("mlp",2,4,1,0.,d.target_names,seed=7); args=(cfg,d,TrainingConfig(4,seed=7),EarlyStoppingConfig(patience=2),SchedulerConfig())
    a=train_supervised_model(*args); b=train_supervised_model(*args)
    assert a.restored_best and a.history==b.history; assert a.evaluations["test"].predictions==b.evaluations["test"].predictions; assert all("test" not in vars(h) for h in a.history)
def test_all_six_models_tiny_cpu_run():
    d=data()
    for name in available_models():
        cfg=ModelConfig(name,2,4,1,0.,d.target_names,attention_heads=1 if name in {"gat","weighted_gat"} else None,seed=3,propagation_steps=1)
        assert train_supervised_model(cfg,d,TrainingConfig(1,seed=3),EarlyStoppingConfig(enabled=False),SchedulerConfig()).completed_epochs==1
def test_threshold_validation_only():
    a=calibrate_anomaly_threshold([1.,2.,3.],"validation_quantile",.5); b=calibrate_anomaly_threshold([1.,2.,3.],"validation_quantile",.5); assert a==b and a["threshold"]==2.; assert calibrate_anomaly_threshold([1.]) is None

def test_sparse_period_uses_global_observed_count_and_ignores_masked_values():
    class Identity(torch.nn.Module):
        def forward(self,x,edge_index,edge_weight):
            class Output: pass
            result=Output(); result.predictions=x; return result
    first=snap(2018,"train"); missing_y=first.y.clone(); missing_y[:,1]=torch.nan
    missing=SupervisedSnapshot(2018,"train",first.node_ids,first.x,missing_y,~torch.isnan(missing_y),first.edge_index,first.edge_weight,first.feature_names,first.target_names)
    second=snap(2019,"train"); total,losses,counts=_global_loss(Identity(),(missing,second),{"target__one":.5,"target__two":.5},require_all=True)
    expected_one=(((missing.x[:,0]-missing.y[:,0])**2).sum()+((second.x[:,0]-second.y[:,0])**2).sum())/6
    expected_two=((second.x[:,1][second.target_observed_mask[:,1]]-second.y[:,1][second.target_observed_mask[:,1]])**2).mean()
    assert counts=={"target__one":6,"target__two":2}; assert losses["target__one"]==pytest.approx(expected_one); assert total==pytest.approx((expected_one+expected_two)/2)
    changed_snapshot=SupervisedSnapshot(2018,"train",first.node_ids,first.x,missing_y.clone(),~torch.isnan(missing_y),first.edge_index,first.edge_weight,first.feature_names,first.target_names)
    changed_snapshot.y[:,1]=torch.tensor([1e20,-1e20,3e20])
    changed_total,_,_=_global_loss(Identity(),(changed_snapshot,second),{"target__one":.5,"target__two":.5},require_all=True); assert changed_total==total

def test_globally_missing_train_target_fails():
    snapshots=[]
    for period in (2018,2019):
        original=snap(period,"train"); y=original.y.clone(); y[:,1]=torch.nan
        snapshots.append(SupervisedSnapshot(period,"train",original.node_ids,original.x,y,~torch.isnan(y),original.edge_index,original.edge_weight,original.feature_names,original.target_names))
    class Identity(torch.nn.Module):
        def forward(self,x,edge_index,edge_weight):
            class Output: pass
            result=Output(); result.predictions=x; return result
    with pytest.raises(ExperimentValidationError,match="no observed labels across split"): _global_loss(Identity(),tuple(snapshots),{"target__one":.5,"target__two":.5},require_all=True)

def test_restore_best_flag_controls_final_state():
    d=data(); config=ModelConfig("mlp",2,4,1,0.,d.target_names,seed=11)
    restored=train_supervised_model(config,d,TrainingConfig(3,learning_rate=.2,seed=11),EarlyStoppingConfig(enabled=False,restore_best=True),SchedulerConfig())
    retained=train_supervised_model(config,d,TrainingConfig(3,learning_rate=.2,seed=11),EarlyStoppingConfig(enabled=False,restore_best=False),SchedulerConfig())
    assert restored.restored_best and all(torch.equal(restored.best_state[k],restored.final_state[k]) for k in restored.best_state)
    assert not retained.restored_best
    if retained.best_epoch!=retained.completed_epochs: assert any(not torch.equal(retained.best_state[k],retained.final_state[k]) for k in retained.best_state)

def test_selected_target_subset_is_canonical_and_ordered():
    nodes=("a","b","c"); periods=(2018,2019,2020); keys=[(node,period) for period in periods for node in nodes]
    feature_panel=SimpleNamespace(feature_names=("safe__a","safe__b"),processed_features=np.tile([[0.,1.],[1.,0.],[.5,.5]],(3,1)),assembly_report=SimpleNamespace(final_period_order=periods,final_node_order=nodes),train_mask=np.array([True]*3+[False]*6),validation_mask=np.array([False]*3+[True]*3+[False]*3),test_mask=np.array([False]*6+[True]*3))
    target_names=("target__one","target__two","target__unused"); frame=pd.DataFrame(keys,columns=("node_id","period"))
    frame["target__one"]=[1.,2.,3.]*3; frame["target__two"]=[4.,5.,6.]*3; frame["target__unused"]=[100.,200.,300.]*3
    provenance=tuple(TargetProvenance(name,"observed","synthetic") for name in target_names); panel=TargetPanel(frame,target_names,provenance)
    graphs={period:GraphSnapshot(tuple(GraphNode(node,node) for node in nodes),np.array([[0,1],[1,2]]),np.array([.2,.4]),np.array([2.,4.]),"synthetic",period) for period in periods}
    prepared=prepare_supervised_data(feature_panel,panel,graphs,safe_feature_names=("safe__a","safe__b"),leakage_audit_report=LeakageAuditReport(()),selected_target_names=("target__two","target__one"))
    baseline=TrainMeanBaseline().fit(prepared.train).evaluate(prepared)
    assert prepared.target_names==("target__two","target__one"); assert tuple(baseline.means)==prepared.target_names
    assert all(record["target_name"]!="target__unused" for evaluation in baseline.evaluations.values() for record in evaluation.predictions)
