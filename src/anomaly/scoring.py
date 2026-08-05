"""Raw reconstruction indicator scoring; it does not establish ground truth."""
import torch
from .contracts import AnomalyScoreOutput
def score_anomalies(model,x,edge_index=None,edge_weight=None):
    output=model(x,edge_index,edge_weight); z_reconstructed=model.encode(output.reconstructed_x,edge_index,edge_weight)
    feature=((x-output.reconstructed_x)**2).mean(1); consistency=((z_reconstructed-output.latent)**2).mean(1)
    coefficient=model.config.anomaly_structural_coefficient
    return AnomalyScoreOutput(feature+coefficient*consistency,feature,consistency,output.latent,output.reconstructed_x,{"anomaly_structural_coefficient":coefficient,"graph_direction":model.config.graph_direction})
__all__=["score_anomalies"]
