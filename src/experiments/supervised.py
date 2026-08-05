import random,numpy as np,torch
from src.models import build_model
from .checkpoints import InMemoryBestCheckpoint
from .contracts import EpochRecord,ExperimentValidationError,TrainingResult
from .losses import masked_multi_target_mse
from .metrics import evaluate_predictions
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True,warn_only=True)
def _aggregate(model,snapshots,weights,training=False,optimizer=None,clip=None):
    sums={n:0. for n in snapshots[0].target_names}; counts={n:0 for n in sums}
    for s in snapshots:
        if training: optimizer.zero_grad()
        output=model(s.x,s.edge_index,s.edge_weight).predictions
        result=masked_multi_target_mse(output,s.y,s.target_observed_mask,s.target_names,weights)
        if training:
            result.total.backward()
            if clip is not None: torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
            optimizer.step()
        for n in sums: sums[n]+=float(result.per_target[n].detach())*result.observed_counts[n]; counts[n]+=result.observed_counts[n]
    per={n:sums[n]/counts[n] for n in sums}; normalized=result.normalized_weights
    return sum(per[n]*normalized[n] for n in per),per,counts,normalized
def train_supervised_model(model_config,prepared_data,training_config,early_stopping_config,scheduler_config,checkpoint_config=None):
    seed_everything(training_config.seed); model=build_model(model_config)
    if model_config.input_dim!=len(prepared_data.feature_names) or tuple(model_config.target_names)!=prepared_data.target_names: raise ExperimentValidationError("model dimensions/targets do not match prepared data")
    if training_config.target_loss_weights is not None and tuple(training_config.target_loss_weights)!=prepared_data.target_names: raise ExperimentValidationError("target loss weight names must exactly match model targets")
    optimizer_cls=torch.optim.AdamW if training_config.optimizer=="adamw" else torch.optim.Adam
    optimizer=optimizer_cls(model.parameters(),lr=training_config.learning_rate,weight_decay=training_config.weight_decay)
    scheduler=None
    if scheduler_config.name=="cosine": scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,training_config.epochs,eta_min=scheduler_config.cosine_min_lr)
    if scheduler_config.name=="reduce_on_plateau": scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=scheduler_config.plateau_factor,patience=scheduler_config.plateau_patience)
    checkpoint=InMemoryBestCheckpoint(early_stopping_config.minimum_delta,None if checkpoint_config is None else checkpoint_config.path); history=[]; stale=0
    for epoch in range(1,training_config.epochs+1):
        model.train(); train_total,train_per,train_counts,normalized=_aggregate(model,prepared_data.train,training_config.target_loss_weights,True,optimizer,training_config.gradient_clip_norm)
        model.eval()
        with torch.no_grad(): val_total,val_per,val_counts,_=_aggregate(model,prepared_data.validation,training_config.target_loss_weights)
        became=checkpoint.update(model,val_total,{"epoch":epoch,"model_name":model_config.name,"target_names":list(model_config.target_names),"seed":training_config.seed,"model_configuration":model_config.to_report(),"training_configuration":training_config.to_report()})
        stale=0 if became else stale+1
        history.append(EpochRecord(epoch,train_total,val_total,train_per,val_per,train_counts,val_counts,optimizer.param_groups[0]["lr"],became))
        if scheduler_config.name=="cosine": scheduler.step()
        elif scheduler_config.name=="reduce_on_plateau": scheduler.step(val_total)
        if early_stopping_config.enabled and stale>=early_stopping_config.patience: break
    checkpoint.restore(model); model.eval(); evaluations={}
    with torch.no_grad():
        for split in ("train","validation","test"):
            snaps=getattr(prepared_data,split); evaluations[split]=evaluate_predictions(snaps,[model(s.x,s.edge_index,s.edge_weight).predictions for s in snaps])
    stopped=len(history)<training_config.epochs
    return TrainingResult(model_config.name,training_config.seed,training_config.epochs,len(history),checkpoint.metadata["epoch"],stopped,"validation_loss_patience" if stopped else "epochs_completed",checkpoint.metadata["monitored_value"],True,normalized,tuple(history),evaluations,{k:v.clone() for k,v in checkpoint.state.items()})
__all__=["seed_everything","train_supervised_model"]
