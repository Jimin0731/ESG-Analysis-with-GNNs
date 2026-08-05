import numpy as np,torch,pandas as pd
from .contracts import ExperimentValidationError,PreparedSupervisedData,SupervisedSnapshot
def prepare_supervised_data(feature_panel,target_panel,graph_snapshots_by_period,*,safe_feature_names,leakage_audit_report,selected_target_names=None):
    names=tuple(safe_feature_names)
    if not names or len(names)!=len(set(names)): raise ExperimentValidationError("safe features must be non-empty and unique")
    if leakage_audit_report is None: raise ExperimentValidationError("leakage audit report is required")
    direct={"target_column_in_features","derived_source_feature_in_features","direct_lineage"}
    if any(f.severity=="error" and f.finding_type in direct and f.feature_name not in leakage_audit_report.removed_feature_names for f in leakage_audit_report.findings): raise ExperimentValidationError("unresolved direct-lineage leakage finding")
    if any(n not in feature_panel.feature_names for n in names): raise ExperimentValidationError("safe feature missing from feature panel")
    report=feature_panel.assembly_report; periods=tuple(report.final_period_order); nodes=tuple(report.final_node_order)
    if len(feature_panel.processed_features)!=len(periods)*len(nodes): raise ExperimentValidationError("feature panel row count disagrees with assembly metadata")
    keys=[(node,period) for period in periods for node in nodes]
    frame=pd.DataFrame(feature_panel.processed_features,columns=feature_panel.feature_names,index=pd.MultiIndex.from_tuples(keys,names=["node_id","period"]))
    graph_keys=tuple(graph_snapshots_by_period)
    if len(graph_keys)!=len(set(graph_keys)) or set(graph_keys)!=set(periods): raise ExperimentValidationError("graphs must contain exactly one snapshot per retained period")
    selected=tuple(selected_target_names or ())
    if not selected or len(selected)!=len(set(selected)) or any(not n.startswith("target__") for n in selected): raise ExperimentValidationError("selected target names must be non-empty, unique, and begin with target__")
    if any(n not in target_panel.target_names for n in selected): raise ExperimentValidationError("selected target missing from target panel")
    targets=target_panel.frame.set_index(["node_id","period"]); features=frame
    if set(targets.index)!=set(features.index): raise ExperimentValidationError("feature and target keys must align exactly")
    split_periods={"train":set(),"validation":set(),"test":set()}
    for index,(_,period) in enumerate(keys):
        memberships=[s for s,m in (("train",feature_panel.train_mask),("validation",feature_panel.validation_mask),("test",feature_panel.test_mask)) if m[index]]
        split_periods[memberships[0]].add(period)
    if any(len({s for s,p in split_periods.items() if period in p})!=1 for period in periods): raise ExperimentValidationError("a period cannot span multiple splits")
    out={k:[] for k in split_periods}
    for period in sorted(periods):
        split=next((s for s,p in split_periods.items() if period in p),None)
        if split is None: raise ExperimentValidationError("period has no chronological split")
        keys=[(n,period) for n in nodes]
        try: x=features.loc[keys,list(names)].to_numpy(dtype=np.float32); y=targets.loc[keys,list(selected)].to_numpy(dtype=np.float32)
        except KeyError as exc: raise ExperimentValidationError("feature and target keys must align exactly") from exc
        graph=graph_snapshots_by_period[period]
        if graph.period!=period or tuple(graph.node_ids)!=nodes: raise ExperimentValidationError("graph period/node order mismatch")
        mask=~np.isnan(y)
        snap=SupervisedSnapshot(period,split,nodes,torch.tensor(x),torch.tensor(y),torch.tensor(mask),torch.tensor(graph.edge_index,dtype=torch.long),torch.tensor(graph.edge_weight,dtype=torch.float32),names,selected)
        out[split].append(snap)
    return PreparedSupervisedData(*(tuple(out[x]) for x in ("train","validation","test")),names,selected)
__all__=["prepare_supervised_data"]
