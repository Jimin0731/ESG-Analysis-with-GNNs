import torch,numpy as np
from src.anomaly import EconomicGAE,reconstruction_loss,sample_directed_negative_edges,score_anomalies
from .checkpoints import InMemoryBestCheckpoint
from .contracts import AnomalyTrainingResult,ExperimentValidationError
from .supervised import seed_everything
def calibrate_anomaly_threshold(validation_scores,policy="none",quantile=None):
    if policy=="none": return None
    if policy!="validation_quantile" or quantile is None or not np.isfinite(quantile) or not 0<quantile<1: raise ExperimentValidationError("invalid anomaly threshold policy")
    return {"threshold":float(np.quantile(validation_scores,quantile)),"quantile":float(quantile),"calibration_split":"validation","label":"heuristic reconstruction flag"}
def train_economic_gae(config,prepared_graph_data,training_config,early_stopping_config,scheduler_config,negative_sampling_config=None,threshold_policy="none",threshold_quantile=None):
    seed_everything(training_config.seed); model=EconomicGAE(config); opt=torch.optim.AdamW(model.parameters(),lr=training_config.learning_rate,weight_decay=training_config.weight_decay); checkpoint=InMemoryBestCheckpoint(early_stopping_config.minimum_delta); stale=0
    for epoch in range(1,training_config.epochs+1):
        model.train()
        for s in prepared_graph_data.train:
            opt.zero_grad(); neg=sample_directed_negative_edges(s.edge_index,len(s.node_ids),min(s.edge_index.shape[1],len(s.node_ids)*(len(s.node_ids)-1)-s.edge_index.shape[1]),seed=training_config.seed*1000003+s.period*1009+epoch); loss=reconstruction_loss(s.x,model(s.x,s.edge_index,s.edge_weight,neg),config).total; loss.backward();opt.step()
        model.eval(); vals=[]
        with torch.no_grad():
            for s in prepared_graph_data.validation:
                neg=sample_directed_negative_edges(s.edge_index,len(s.node_ids),min(s.edge_index.shape[1],len(s.node_ids)*(len(s.node_ids)-1)-s.edge_index.shape[1]),seed=training_config.seed*1000003+s.period*1009+epoch); vals.append(float(reconstruction_loss(s.x,model(s.x,s.edge_index,s.edge_weight,neg),config).total))
        value=sum(vals)/len(vals); became=checkpoint.update(model,value,{"epoch":epoch}); stale=0 if became else stale+1
        if early_stopping_config.enabled and stale>=early_stopping_config.patience: break
    checkpoint.restore(model);model.eval(); scores={}
    with torch.no_grad():
        for split in ("train","validation","test"):
            records=[]
            for s in getattr(prepared_graph_data,split):
                output=score_anomalies(model,s.x,s.edge_index,s.edge_weight)
                for node,total,feature,consistency in zip(s.node_ids,output.anomaly_scores,output.feature_reconstruction_error,output.embedding_consistency_error): records.append({"node_id":node,"period":s.period,"split":split,"anomaly_score":float(total),"feature_error":float(feature),"consistency_error":float(consistency)})
            scores[split]=tuple(records)
    threshold=calibrate_anomaly_threshold([r["anomaly_score"] for r in scores["validation"]],threshold_policy,threshold_quantile)
    if threshold:
        for split in scores:
            scores[split]=tuple({**r,"heuristic_flag":r["anomaly_score"]>=threshold["threshold"]} for r in scores[split])
    return AnomalyTrainingResult(training_config.seed,checkpoint.metadata["epoch"],checkpoint.metadata["monitored_value"],True,scores,threshold)
__all__=["calibrate_anomaly_threshold","train_economic_gae"]
