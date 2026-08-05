import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch.nn.functional as F

from src.data.preprocessing import build_graph_dataset, graph_from_io_matrix, make_smoke_dataset, targets_from_esg_frame
from src.models.gnn_models import EconomicESGGNN
from src.training import evaluate_model, to_tensors, train_model


def test_smoke_dataset_shapes():
    dataset = make_smoke_dataset()
    assert dataset.features.shape == (4, 6)
    assert dataset.edge_index.shape[0] == 2
    assert dataset.edge_weight.shape[0] == dataset.edge_index.shape[1]
    assert dataset.targets.shape == (4,)


def test_graph_from_io_matrix_rejects_empty_graph():
    with pytest.raises(ValueError, match="positive flows"):
        graph_from_io_matrix(np.zeros((2, 2)))


def test_graph_from_io_matrix_validates_percentile_and_square_shape():
    with pytest.raises(ValueError, match="percentile"):
        graph_from_io_matrix(np.ones((2, 2)), percentile=101)
    with pytest.raises(ValueError, match="square"):
        build_graph_dataset(np.ones((2, 3)))


def test_targets_from_esg_frame_rejects_empty_data():
    with pytest.raises(ValueError, match="at least one row"):
        targets_from_esg_frame(pd.DataFrame(columns=["sector", "esg_score"]), ["Energy"])


def test_targets_from_esg_frame_rejects_unaligned_target_length_mismatch():
    with pytest.raises(ValueError, match="target length"):
        targets_from_esg_frame(pd.DataFrame({"esg_score": [1.0]}), ["Energy", "Tech"])


def test_build_graph_dataset_rejects_target_length_mismatch():
    with pytest.raises(ValueError, match="matching the number"):
        build_graph_dataset(np.ones((2, 2)), targets=np.array([1.0]))


def test_train_and_evaluate_smoke_dataset():
    dataset = make_smoke_dataset()
    model, history = train_model(dataset, epochs=2, hidden_dim=8)
    assert isinstance(model, EconomicESGGNN)
    assert len(history) == 2
    metrics = evaluate_model(model, dataset)
    assert set(metrics) == {"mae", "rmse", "r2"}


def test_train_model_rejects_non_positive_epochs():
    with pytest.raises(ValueError, match="positive"):
        train_model(make_smoke_dataset(), epochs=0)


def test_forward_backward_step():
    dataset = make_smoke_dataset()
    model = EconomicESGGNN(input_dim=dataset.features.shape[1], hidden_dim=8)
    x, edge_index, edge_weight, y = to_tensors(dataset)
    pred = model(x, edge_index, edge_weight)["esg_risk"]
    loss = F.mse_loss(pred, y)
    loss.backward()
    assert any(param.grad is not None for param in model.parameters())


def test_cli_writes_metrics_output(tmp_path):
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
    assert "metrics_output=" in result.stdout
    metrics = json.loads(metrics_path.read_text())
    assert set(metrics) == {"mae", "rmse", "r2"}


def test_cli_rejects_invalid_epochs():
    result = subprocess.run(
        [sys.executable, "scripts/run_pipeline.py", "--smoke-test", "--epochs", "0"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "positive integer" in result.stderr
