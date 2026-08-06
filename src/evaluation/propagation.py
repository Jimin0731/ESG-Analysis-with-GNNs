from __future__ import annotations
import torch
from .attention_explanations import _infer
from .explanation_contracts import (InterpretabilityValidationError,NON_CAUSAL_WARNINGS,PropagationCoefficientRecord,PropagationExplanation)

def extract_propagation_explanation(model,snapshot):
    if getattr(getattr(model,"config",None),"name",None)!="gpr_gnn": raise InterpretabilityValidationError("model does not provide GPR propagation coefficients")
    aux=_infer(model,snapshot,aux=True).auxiliary; values=aux.get("propagation_coefficients"); depth=aux.get("propagation_depth")
    if not isinstance(values,torch.Tensor) or values.ndim!=1 or len(values)!=depth+1: raise InterpretabilityValidationError("invalid propagation coefficient shape")
    if not torch.isfinite(values).all() or (values<0).any() or not torch.isclose(values.sum(),torch.tensor(1.,device=values.device),atol=1e-5): raise InterpretabilityValidationError("invalid propagation coefficients")
    alpha=float(aux["alpha"]); direction=str(aux["graph_direction"])
    records=tuple(PropagationCoefficientRecord(i,float(v),alpha,direction,"initial_encoded_state" if i==0 else "propagated_state") for i,v in enumerate(values))
    return PropagationExplanation("gpr_gnn",snapshot.period,snapshot.split,records,"These are softmax-normalized propagation-step coefficients, not edge attention weights.",NON_CAUSAL_WARNINGS)
__all__=["extract_propagation_explanation"]
