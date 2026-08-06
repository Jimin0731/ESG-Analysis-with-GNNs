"""Single-snapshot training helpers backed by the canonical model registry.

The chronological multi-snapshot workflow lives in :mod:`src.experiments`.
This smaller helper exists for the file-based ``scripts/run_pipeline.py``
entry point and is intentionally not a scientific evaluation protocol.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data.preprocessing import GraphDataset
from src.models import ModelConfig, ModelOutput, build_model


def to_tensors(dataset: GraphDataset, device: str | torch.device = "cpu"):
    return (
        torch.tensor(dataset.features, dtype=torch.float32, device=device),
        torch.tensor(dataset.edge_index, dtype=torch.long, device=device),
        torch.tensor(dataset.edge_weight, dtype=torch.float32, device=device),
        torch.tensor(dataset.targets, dtype=torch.float32, device=device),
    )


def train_model(dataset: GraphDataset, epochs: int = 50, hidden_dim: int = 32, lr: float = 0.01, seed: int = 7, device: str = "cpu"):
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    torch.manual_seed(seed)
    x, edge_index, edge_weight, y = to_tensors(dataset, device)
    model = build_model(ModelConfig(
        name="weighted_gat",
        input_dim=x.shape[1],
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.0,
        target_names=("target__esg_score",),
        attention_heads=2,
        seed=seed,
    )).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for _ in range(epochs):
        model.train(); optimizer.zero_grad()
        pred = model(x, edge_index, edge_weight).predictions[:, 0]
        loss = F.mse_loss(pred, y)
        loss.backward(); optimizer.step()
        history.append(float(loss.detach().cpu()))
    return model, history


def evaluate_model(model: torch.nn.Module, dataset: GraphDataset, device: str = "cpu") -> dict[str, float]:
    model.eval(); x, edge_index, edge_weight, y = to_tensors(dataset, device)
    with torch.no_grad():
        output = model(x, edge_index, edge_weight)
        if not isinstance(output, ModelOutput):
            raise TypeError("canonical models must return ModelOutput")
        pred = output.predictions[:, 0].detach().cpu().numpy()
    target = y.detach().cpu().numpy()
    return {"mae": float(mean_absolute_error(target, pred)), "rmse": float(np.sqrt(mean_squared_error(target, pred))), "r2": float(r2_score(target, pred))}
