import math,torch,pytest
from src.experiments import *
from src.models import ModelConfig,available_models
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
