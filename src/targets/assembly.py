from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd
from .contracts import TargetPanel, TargetBlock, TargetAssemblyReport, TargetValidationError, LeakageAuditReport

@dataclass(frozen=True)
class SupervisedTargetResult:
    ordered_node_ids: tuple[str,...]
    ordered_periods: tuple[int,...]
    target_matrix: np.ndarray
    target_names: tuple[str,...]
    target_missing_mask: np.ndarray
    target_provenance: tuple
    target_assembly_report: TargetAssemblyReport
    leakage_audit_report: LeakageAuditReport|None
    safe_feature_names: tuple[str,...]

def _ordered_key_dicts(keys):
    return [{'node_id': str(n), 'period': int(p)} for n,p in sorted(keys, key=lambda x: (int(x[1]), str(x[0])))]


def assemble_target_blocks(feature_keys: pd.DataFrame, blocks, *, strict=True) -> TargetPanel:
    if feature_keys.duplicated(['node_id','period']).any(): raise TargetValidationError('feature keys must be unique')
    canonical=feature_keys[['node_id','period']].copy().reset_index(drop=True)
    names=[]; prov=[]; final=canonical.copy(); missing={}; extra={}
    keyset=set(map(tuple, canonical[['node_id','period']].to_numpy()))
    for i,b in enumerate(blocks):
        if b.frame.duplicated(['node_id','period']).any(): raise TargetValidationError('duplicate target keys are not allowed')
        if set(names).intersection(b.target_names): raise TargetValidationError('target-name collisions are not allowed')
        bkeys=set(map(tuple,b.frame[['node_id','period']].to_numpy()))
        label=','.join(b.target_names)
        missing[label]=_ordered_key_dicts(keyset-bkeys)
        extra[label]=_ordered_key_dicts(bkeys-keyset)
        if strict and (missing[label] or extra[label]): raise TargetValidationError('target keys do not align with feature keys')
        final=final.merge(b.frame[['node_id','period',*b.target_names]],on=['node_id','period'],how='left',validate='one_to_one')
        names.extend(b.target_names); prov.extend(b.provenance)
    report=TargetAssemblyReport(strict,len(canonical),len(final),missing,extra)
    return TargetPanel(final,tuple(names),tuple(prov),report)

def make_supervised_target_result(feature_keys, target_panel, *, leakage_audit_report=None, safe_feature_names=()):
    return SupervisedTargetResult(tuple(target_panel.frame.node_id), tuple(int(x) for x in target_panel.frame.period), target_panel.frame[list(target_panel.target_names)].to_numpy(float), target_panel.target_names, target_panel.frame[list(target_panel.target_names)].isna().to_numpy(bool), target_panel.provenance, target_panel.assembly_report, leakage_audit_report, tuple(safe_feature_names))
