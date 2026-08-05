#!/usr/bin/env python3
"""Fast, file-free synthetic chronological experiment smoke run."""
from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.experiments import EarlyStoppingConfig,PreparedSupervisedData,SchedulerConfig,SupervisedSnapshot,TrainMeanBaseline,TrainingConfig,train_supervised_model
from src.models import ModelConfig
def snapshot(period,split):
    x=torch.tensor([[period/10,0.],[0.,period/10],[.5,.5]],dtype=torch.float32); y=torch.tensor([[x[0,0],1.],[x[1,1],float("nan")],[.5,.5]])
    return SupervisedSnapshot(period,split,("A","B","C"),x,y,~torch.isnan(y),torch.tensor([[0,1,2],[1,2,0]]),torch.tensor([.2,.4,.6]),("synthetic__level","synthetic__change"),("target__score","target__risk"))
def main():
    data=PreparedSupervisedData(tuple(snapshot(p,"train") for p in (2018,2019,2020)),(snapshot(2021,"validation"),),(snapshot(2022,"test"),),("synthetic__level","synthetic__change"),("target__score","target__risk"))
    config=ModelConfig("mlp",2,4,1,0.,data.target_names,seed=17); result=train_supervised_model(config,data,TrainingConfig(epochs=4,learning_rate=.01,seed=17),EarlyStoppingConfig(patience=2),SchedulerConfig("cosine")); baseline=TrainMeanBaseline().fit(data.train).evaluate(data)
    print(f"experiment_smoke best_epoch={result.best_epoch} validation_rmse={result.evaluations['validation'].macro_rmse:.6f} test_rmse={result.evaluations['test'].macro_rmse:.6f} baseline_test_rmse={baseline.evaluations['test'].macro_rmse:.6f} restored_best={result.restored_best}")
if __name__=="__main__": main()
