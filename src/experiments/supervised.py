"""Seeded supervised training with full-split label-count aggregation."""
import random
import numpy as np
import torch
from src.models import build_model
from .checkpoints import InMemoryBestCheckpoint
from .contracts import EpochRecord,ExperimentValidationError,TrainingResult
from .metrics import evaluate_predictions

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)

def _weights(target_names,configured):
    if configured is not None and tuple(configured)!=tuple(target_names): raise ExperimentValidationError("target loss weight names must exactly match model targets")
    raw={n:float((configured or {}).get(n,1.)) for n in target_names}; denominator=sum(raw.values())
    return {n:v/denominator for n,v in raw.items()}

def _global_loss(model,snapshots,weights,*,require_all):
    names=snapshots[0].target_names
    sums={n:None for n in names}; counts={n:0 for n in names}
    for snapshot in snapshots:
        predictions=model(snapshot.x,snapshot.edge_index,snapshot.edge_weight).predictions
        for column,name in enumerate(names):
            observed=snapshot.target_observed_mask[:,column]; count=int(observed.sum())
            if count:
                value=((predictions[:,column][observed]-snapshot.y[:,column][observed])**2).sum()
                sums[name]=value if sums[name] is None else sums[name]+value; counts[name]+=count
    missing=[name for name in names if not counts[name]]
    if require_all and missing: raise ExperimentValidationError(f"targets have no observed labels across split: {missing}")
    losses={name:(sums[name]/counts[name] if counts[name] else None) for name in names}
    available=[name for name in names if counts[name]]
    if not available: raise ExperimentValidationError("split contains no observed labels")
    total=sum(losses[name]*weights[name] for name in available)
    return total,losses,counts

def train_supervised_model(model_config,prepared_data,training_config,early_stopping_config,scheduler_config,checkpoint_config=None):
    seed_everything(training_config.seed); model=build_model(model_config)
    if model_config.input_dim!=len(prepared_data.feature_names) or tuple(model_config.target_names)!=prepared_data.target_names: raise ExperimentValidationError("model dimensions/targets do not match prepared data")
    normalized=_weights(prepared_data.target_names,training_config.target_loss_weights)
    optimizer_cls=torch.optim.AdamW if training_config.optimizer=="adamw" else torch.optim.Adam
    optimizer=optimizer_cls(model.parameters(),lr=training_config.learning_rate,weight_decay=training_config.weight_decay)
    scheduler=None
    if scheduler_config.name=="cosine": scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,training_config.epochs,eta_min=scheduler_config.cosine_min_lr)
    elif scheduler_config.name=="reduce_on_plateau": scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=scheduler_config.plateau_factor,patience=scheduler_config.plateau_patience)
    checkpoint=InMemoryBestCheckpoint(early_stopping_config.minimum_delta,None if checkpoint_config is None else checkpoint_config.path); history=[]; stale=0
    for epoch in range(1,training_config.epochs+1):
        model.train(); optimizer.zero_grad()
        train_total,train_losses,train_counts=_global_loss(model,prepared_data.train,normalized,require_all=True)
        train_total.backward()
        if training_config.gradient_clip_norm is not None: torch.nn.utils.clip_grad_norm_(model.parameters(),training_config.gradient_clip_norm)
        optimizer.step(); model.eval()
        with torch.no_grad(): validation_total,validation_losses,validation_counts=_global_loss(model,prepared_data.validation,normalized,require_all=False)
        value=float(validation_total); became=checkpoint.update(model,value,{"epoch":epoch,"model_name":model_config.name,"target_names":list(model_config.target_names),"seed":training_config.seed,"model_configuration":model_config.to_report(),"training_configuration":training_config.to_report()})
        stale=0 if became else stale+1
        history.append(EpochRecord(epoch,float(train_total.detach()),value,{n:None if v is None else float(v.detach()) for n,v in train_losses.items()},{n:None if v is None else float(v) for n,v in validation_losses.items()},train_counts,validation_counts,optimizer.param_groups[0]["lr"],became))
        if scheduler_config.name=="cosine": scheduler.step()
        elif scheduler_config.name=="reduce_on_plateau": scheduler.step(value)
        if early_stopping_config.enabled and stale>=early_stopping_config.patience: break
    if early_stopping_config.restore_best: checkpoint.restore(model)
    model.eval(); evaluations={}
    with torch.no_grad():
        for split in ("train","validation","test"):
            snapshots=getattr(prepared_data,split); evaluations[split]=evaluate_predictions(snapshots,[model(s.x,s.edge_index,s.edge_weight).predictions for s in snapshots])
    stopped=len(history)<training_config.epochs; final={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    return TrainingResult(model_config.name,training_config.seed,training_config.epochs,len(history),checkpoint.metadata["epoch"],stopped,"validation_loss_patience" if stopped else "epochs_completed",checkpoint.metadata["monitored_value"],early_stopping_config.restore_best,normalized,tuple(history),evaluations,{k:v.clone() for k,v in checkpoint.state.items()},final)
__all__=["seed_everything","train_supervised_model"]
