import platform,torch,numpy as np
from src.models import get_model_capabilities
from .baselines import TrainMeanBaseline
from .contracts import ExperimentReport
from .preparation import prepare_supervised_data
from .supervised import train_supervised_model
def run_supervised_experiment(experiment_config,feature_panel,target_panel,graph_snapshots_by_period,leakage_audit_report):
    data=prepare_supervised_data(feature_panel,target_panel,graph_snapshots_by_period,safe_feature_names=experiment_config.safe_feature_names,leakage_audit_report=leakage_audit_report)
    training=train_supervised_model(experiment_config.model,data,experiment_config.training,experiment_config.early_stopping,experiment_config.scheduler,experiment_config.checkpoint)
    baseline=TrainMeanBaseline().fit(data.train).evaluate(data)
    return ExperimentReport(experiment_config.name,experiment_config.seed,get_model_capabilities(experiment_config.model.name).to_report(),data.feature_names,data.target_names,{s:tuple(x.period for x in getattr(data,s)) for s in ("train","validation","test")},tuple(feature_panel.preprocessing_state.fit_periods),tuple(p.to_dict() for p in target_panel.provenance),leakage_audit_report.to_dict(),training.to_report(),{"means":baseline.means,"evaluations":{k:vars(v) for k,v in baseline.evaluations.items()}},{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__})
__all__=["run_supervised_experiment"]
