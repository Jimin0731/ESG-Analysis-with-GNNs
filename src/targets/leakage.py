from __future__ import annotations
import pandas as pd
from .contracts import LeakageFinding, LeakageAuditReport, TargetValidationError
POLICIES={'error','drop_declared_sources'}

def audit_direct_leakage(feature_names, target_panel, *, feature_keys:pd.DataFrame|None=None, declared_statistics_metadata=None) -> LeakageAuditReport:
    feats=tuple(feature_names); findings=[]
    if len(feats)!=len(set(feats)): findings.append(LeakageFinding('*','duplicate_feature_names',message='duplicate feature names declared'))
    if len(target_panel.target_names)!=len(set(target_panel.target_names)): findings.append(LeakageFinding('*','duplicate_target_names',message='duplicate target names declared'))
    prov_seen={}
    for p in target_panel.provenance:
        if p.target_name in prov_seen and prov_seen[p.target_name].to_dict()!=p.to_dict(): findings.append(LeakageFinding(p.target_name,'conflicting_target_provenance',message='duplicate or conflicting target provenance'))
        prov_seen[p.target_name]=p
        if p.target_name in feats: findings.append(LeakageFinding(p.target_name,'target_column_in_features',p.target_name,'target column is included directly in feature set'))
        if p.target_kind=='derived':
            for s in p.source_feature_names:
                if s in feats: findings.append(LeakageFinding(p.target_name,'derived_source_feature_in_features',s,'derived target source feature is included in model feature set'))
        if p.metadata.get('fitted_on') in {'validation','test','validation_test'}: findings.append(LeakageFinding(p.target_name,'validation_or_test_fitted_statistics',message='target metadata declares validation/test-derived fitted statistics'))
    if feature_keys is not None:
        f=set(map(tuple,feature_keys[['node_id','period']].to_numpy())); t=set(map(tuple,target_panel.frame[['node_id','period']].to_numpy()))
        if f!=t: findings.append(LeakageFinding('*','target_key_misalignment',message='target keys do not align with feature keys',metadata={'missing_target_keys':list(map(list,f-t)),'extra_target_keys':list(map(list,t-f))}))
    for p in target_panel.provenance:
        if 'observation_period' in target_panel.frame.columns:
            bad=target_panel.frame['observation_period'] != target_panel.frame['period'] + p.forecast_horizon
            if bad.any(): findings.append(LeakageFinding(p.target_name,'forecast_horizon_violation',message='target observation periods violate configured forecast horizon'))
    return LeakageAuditReport(tuple(findings))

def audit_and_filter_features(feature_names, target_panel, policy='error', **kwargs):
    if policy not in POLICIES: raise TargetValidationError('unsupported leakage policy')
    report=audit_direct_leakage(feature_names,target_panel,**kwargs)
    if policy=='error' and report.findings: raise TargetValidationError('direct leakage findings detected')
    remove=[]; reasons={}
    if policy=='drop_declared_sources':
        for f in report.findings:
            if f.finding_type in {'target_column_in_features','derived_source_feature_in_features'} and f.feature_name:
                if f.feature_name not in remove: remove.append(f.feature_name)
                reasons.setdefault(f.feature_name,[]).append(f'{f.finding_type}:{f.target_name}')
    safe=tuple(x for x in feature_names if x not in set(remove))
    report=LeakageAuditReport(report.findings,policy,tuple(remove),reasons)
    return safe, tuple(remove), report, reasons
