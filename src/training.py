"""Training and evaluation helpers for the ESG GNN pipeline."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data.preprocessing import GraphDataset
from src.models.gnn_models import EconomicESGGNN


def to_tensors(dataset: GraphDataset, device: str | torch.device = "cpu"):
    return (
        torch.tensor(dataset.features, dtype=torch.float32, device=device),
        torch.tensor(dataset.edge_index, dtype=torch.long, device=device),
        torch.tensor(dataset.edge_weight, dtype=torch.float32, device=device),
        torch.tensor(dataset.targets, dtype=torch.float32, device=device),
    )


def train_model(dataset: GraphDataset, epochs: int = 50, hidden_dim: int = 32, lr: float = 0.01, seed: int = 7, device: str = "cpu"):
    torch.manual_seed(seed)
    x, edge_index, edge_weight, y = to_tensors(dataset, device)
    model = EconomicESGGNN(input_dim=x.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for _ in range(epochs):
        model.train(); optimizer.zero_grad()
        pred = model(x, edge_index, edge_weight)["esg_risk"]
        loss = F.mse_loss(pred, y)
        loss.backward(); optimizer.step()
        history.append(float(loss.detach().cpu()))
    return model, history


def evaluate_model(model: EconomicESGGNN, dataset: GraphDataset, device: str = "cpu") -> dict[str, float]:
    model.eval(); x, edge_index, edge_weight, y = to_tensors(dataset, device)
    with torch.no_grad():
        pred = model(x, edge_index, edge_weight)["esg_risk"].detach().cpu().numpy()
    target = y.detach().cpu().numpy()
    return {"mae": float(mean_absolute_error(target, pred)), "rmse": float(np.sqrt(mean_squared_error(target, pred))), "r2": float(r2_score(target, pred))}
