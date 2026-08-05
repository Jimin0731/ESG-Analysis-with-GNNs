"""Deterministic, configuration-only GPR ablation specifications."""
from dataclasses import dataclass,replace
from .contracts import ModelConfig,ModelValidationError
@dataclass(frozen=True)
class GPRAblationSpec:
    name:str; alpha:float; propagation_steps:int; graph_direction:str; config:ModelConfig; changed_factors:tuple[str,...]
def build_gpr_ablation_configs(base_config,*,alphas,propagation_steps,graph_directions):
    grids=tuple(map(tuple,(alphas,propagation_steps,graph_directions)))
    if any(not g for g in grids): raise ModelValidationError("ablation grids must not be empty")
    if any(len(set(g))!=len(g) for g in grids): raise ModelValidationError("ablation grids must not contain duplicates")
    out=[]; names=set()
    for alpha in grids[0]:
      for depth in grids[1]:
       for direction in grids[2]:
        cfg=replace(base_config,name="gpr_gnn",alpha=alpha,propagation_steps=depth,graph_direction=direction)
        name=f"gpr_alpha-{float(alpha):g}_k-{depth}_direction-{direction}"
        if name in names: raise ModelValidationError("duplicate generated ablation name")
        names.add(name); changed=tuple(k for k,v in (("alpha",alpha),("propagation_steps",depth),("graph_direction",direction)) if getattr(base_config,k)!=v)
        out.append(GPRAblationSpec(name,alpha,depth,direction,cfg,changed))
    return tuple(out)
__all__=["GPRAblationSpec","build_gpr_ablation_configs"]
