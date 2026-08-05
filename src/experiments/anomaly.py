"""Separate, seeded EconomicGAE training and validation-only calibration."""
import numpy as np
import torch
from src.anomaly import EconomicGAE,reconstruction_loss,sample_directed_negative_edges,score_anomalies
from .checkpoints import InMemoryBestCheckpoint
from .contracts import AnomalyTrainingResult,ExperimentValidationError,NegativeSamplingConfig
from .supervised import seed_everything

def calibrate_anomaly_threshold(validation_scores,policy="none",quantile=None):
    if policy=="none": return None
    if policy!="validation_quantile" or quantile is None or not np.isfinite(quantile) or not 0<quantile<1: raise ExperimentValidationError("invalid anomaly threshold policy")
    return {"threshold":float(np.quantile(validation_scores,quantile)),"quantile":float(quantile),"calibration_split":"validation","label":"heuristic reconstruction flag"}

def _negative_edges(snapshot,configuration,seed):
    positives=set(map(tuple,snapshot.edge_index.t().tolist()))
    positive_basis=sum(1 for source,target in positives if source!=target)
    n=len(snapshot.node_ids)
    candidates=sum(1 for source in range(n) for target in range(n) if (not configuration.exclude_self_loops or source!=target) and (source,target) not in positives)
    requested=min(int(np.ceil(positive_basis*configuration.negative_ratio)),candidates)
    return sample_directed_negative_edges(snapshot.edge_index,n,requested,seed=seed,exclude_self_loops=configuration.exclude_self_loops)

def train_economic_gae(config,prepared_graph_data,training_config,early_stopping_config,scheduler_config,negative_sampling_config=None,threshold_policy="none",threshold_quantile=None):
    negative_sampling_config=negative_sampling_config or NegativeSamplingConfig(); seed_everything(training_config.seed); model=EconomicGAE(config)
    optimizer_cls=torch.optim.Adam if training_config.optimizer=="adam" else torch.optim.AdamW
    optimizer=optimizer_cls(model.parameters(),lr=training_config.learning_rate,weight_decay=training_config.weight_decay); scheduler=None
    if scheduler_config.name=="cosine": scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,training_config.epochs,eta_min=scheduler_config.cosine_min_lr)
    elif scheduler_config.name=="reduce_on_plateau": scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=scheduler_config.plateau_factor,patience=scheduler_config.plateau_patience)
    checkpoint=InMemoryBestCheckpoint(early_stopping_config.minimum_delta); stale=0
    for epoch in range(1,training_config.epochs+1):
        model.train(); optimizer.zero_grad(); losses=[]
        for snapshot in prepared_graph_data.train:
            negative=_negative_edges(snapshot,negative_sampling_config,training_config.seed*1000003+snapshot.period*1009+epoch)
            losses.append(reconstruction_loss(snapshot.x,model(snapshot.x,snapshot.edge_index,snapshot.edge_weight,negative),config).total)
        train_loss=sum(losses)/len(losses); train_loss.backward()
        if training_config.gradient_clip_norm is not None: torch.nn.utils.clip_grad_norm_(model.parameters(),training_config.gradient_clip_norm)
        optimizer.step(); model.eval(); validation_losses=[]
        with torch.no_grad():
            for snapshot in prepared_graph_data.validation:
                negative=_negative_edges(snapshot,negative_sampling_config,training_config.seed*1000003+snapshot.period*1009+epoch)
                validation_losses.append(reconstruction_loss(snapshot.x,model(snapshot.x,snapshot.edge_index,snapshot.edge_weight,negative),config).total)
        value=float(sum(validation_losses)/len(validation_losses)); became=checkpoint.update(model,value,{"epoch":epoch}); stale=0 if became else stale+1
        if scheduler_config.name=="cosine": scheduler.step()
        elif scheduler_config.name=="reduce_on_plateau": scheduler.step(value)
        if early_stopping_config.enabled and stale>=early_stopping_config.patience: break
    if early_stopping_config.restore_best: checkpoint.restore(model)
    model.eval(); scores={}
    with torch.no_grad():
        for split in ("train","validation","test"):
            records=[]
            for snapshot in getattr(prepared_graph_data,split):
                output=score_anomalies(model,snapshot.x,snapshot.edge_index,snapshot.edge_weight)
                for node,total,feature,consistency in zip(snapshot.node_ids,output.anomaly_scores,output.feature_reconstruction_error,output.embedding_consistency_error): records.append({"node_id":node,"period":snapshot.period,"split":split,"anomaly_score":float(total),"feature_error":float(feature),"consistency_error":float(consistency)})
            scores[split]=tuple(records)
    threshold=calibrate_anomaly_threshold([record["anomaly_score"] for record in scores["validation"]],threshold_policy,threshold_quantile)
    if threshold:
        for split in scores: scores[split]=tuple({**record,"heuristic_flag":record["anomaly_score"]>=threshold["threshold"]} for record in scores[split])
    return AnomalyTrainingResult(training_config.seed,checkpoint.metadata["epoch"],checkpoint.metadata["monitored_value"],early_stopping_config.restore_best,scores,threshold)
__all__=["calibrate_anomaly_threshold","train_economic_gae"]
