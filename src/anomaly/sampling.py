"""Deterministic sampling from the directed non-edge set."""
import random,torch
from .contracts import AnomalyValidationError
def sample_directed_negative_edges(edge_index,num_nodes,num_samples,*,seed=0,exclude_self_loops=True):
    if isinstance(num_samples,bool) or num_samples<0: raise AnomalyValidationError("num_samples must be non-negative")
    positives=set(map(tuple,edge_index.t().tolist()))
    candidates=[(u,v) for u in range(num_nodes) for v in range(num_nodes) if (not exclude_self_loops or u!=v) and (u,v) not in positives]
    if num_samples>len(candidates): raise AnomalyValidationError("requested more negatives than available")
    chosen=random.Random(seed).sample(candidates,num_samples)
    return torch.tensor(chosen,dtype=torch.long,device=edge_index.device).t().contiguous() if chosen else torch.empty((2,0),dtype=torch.long,device=edge_index.device)
__all__=["sample_directed_negative_edges"]
