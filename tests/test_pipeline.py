import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.preprocessing import build_graph_dataset, graph_from_io_matrix, make_smoke_dataset, targets_from_esg_frame
from src.models.gnn_models import EconomicESGGNN
from src.training import evaluate_model, train_model, to_tensors


def test_smoke_dataset_shapes():
    dataset = make_smoke_dataset()
    assert dataset.features.shape == (4, 6)
    assert dataset.edge_index.shape[0] == 2
    assert dataset.edge_weight.shape[0] == dataset.edge_index.shape[1]
    assert dataset.targets.shape == (4,)


def test_graph_from_io_matrix_rejects_empty_graph():
    with pytest.raises(ValueError, match="positive flows"):
        graph_from_io_matrix(np.zeros((2, 2)))


def test_invalid_inputs_have_clear_errors():
    with pytest.raises(ValueError, match="square"):
        graph_from_io_matrix(np.ones((2, 3)))
    with pytest.raises(ValueError, match="percentile"):
        graph_from_io_matrix(np.ones((2, 2)), percentile=101)
    with pytest.raises(ValueError, match="target length"):
        build_graph_dataset(np.ones((2, 2)), targets=np.array([1.0]))
    with pytest.raises(ValueError, match="at least one row"):
        targets_from_esg_frame(pd.DataFrame(columns=["esg_score"]), ["A"])
    with pytest.raises(ValueError, match="positive integer"):
        train_model(make_smoke_dataset(), epochs=0)


def test_train_and_evaluate_smoke_dataset():
    dataset = make_smoke_dataset()
    model, history = train_model(dataset, epochs=2, hidden_dim=8)
    assert isinstance(model, EconomicESGGNN)
    assert len(history) == 2
    metrics = evaluate_model(model, dataset)
    assert set(metrics) == {"mae", "rmse", "r2"}


def test_real_forward_and_backward_step_completes():
    dataset = make_smoke_dataset()
    x, edge_index, edge_weight, y = to_tensors(dataset)
    model = EconomicESGGNN(input_dim=x.shape[1], hidden_dim=8)
    pred = model(x, edge_index, edge_weight)["esg_risk"]
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    assert loss.item() > 0
    assert any(param.grad is not None for param in model.parameters())


def test_cli_from_repo_root_writes_metrics(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_pipeline.py",
            "--smoke-test",
            "--epochs",
            "2",
            "--hidden-dim",
            "8",
            "--metrics-output",
            str(metrics_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "metrics_json=" in result.stdout
    payload = json.loads(metrics_path.read_text())
    assert {"mae", "rmse", "r2"}.issubset(payload)


def test_cli_invalid_epochs_fails_clearly():
    result = subprocess.run(
        [sys.executable, "scripts/run_pipeline.py", "--smoke-test", "--epochs", "0"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "epochs must be a positive integer" in result.stderr
