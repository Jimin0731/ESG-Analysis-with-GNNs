import platform,torch,numpy as np
from src.models import get_model_capabilities
from .baselines import TrainMeanBaseline
from .contracts import ExperimentReport,ExperimentValidationError
from .preparation import prepare_supervised_data
from .supervised import train_supervised_model
def run_supervised_experiment(experiment_config,feature_panel,target_panel,graph_snapshots_by_period,leakage_audit_report):
    selected=tuple(experiment_config.selected_target_names)
    if tuple(experiment_config.model.target_names)!=selected: raise ExperimentValidationError("model target names must match selected targets")
    provenance_by_target={}
    for provenance in target_panel.provenance:
        provenance_by_target.setdefault(provenance.target_name,[]).append(provenance)
    selected_provenance=[]
    for target_name in selected:
        matches=provenance_by_target.get(target_name,[])
        if not matches: raise ExperimentValidationError(f"selected target provenance is missing: {target_name}")
        if len(matches)>1: raise ExperimentValidationError(f"duplicate provenance for selected target: {target_name}")
        selected_provenance.append(matches[0].to_dict())
    data=prepare_supervised_data(feature_panel,target_panel,graph_snapshots_by_period,safe_feature_names=experiment_config.safe_feature_names,leakage_audit_report=leakage_audit_report,selected_target_names=experiment_config.selected_target_names)
    training=train_supervised_model(experiment_config.model,data,experiment_config.training,experiment_config.early_stopping,experiment_config.scheduler,experiment_config.checkpoint)
    baseline=TrainMeanBaseline().fit(data.train).evaluate(data)
    return ExperimentReport(experiment_config.name,experiment_config.seed,get_model_capabilities(experiment_config.model.name).to_report(),data.feature_names,data.target_names,{s:tuple(x.period for x in getattr(data,s)) for s in ("train","validation","test")},tuple(feature_panel.preprocessing_state.fit_periods),tuple(selected_provenance),leakage_audit_report.to_dict(),training.to_report(),{"means":baseline.means,"evaluations":{k:vars(v) for k,v in baseline.evaluations.items()}},{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__})
__all__=["run_supervised_experiment"]
