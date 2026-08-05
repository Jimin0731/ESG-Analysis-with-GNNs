import numpy as np
from src.data.preprocessing import make_smoke_dataset, graph_from_io_matrix
from src.models.gnn_models import EconomicESGGNN
from src.training import evaluate_model, train_model


def test_smoke_dataset_shapes():
    dataset = make_smoke_dataset()
    assert dataset.features.shape == (4, 6)
    assert dataset.edge_index.shape[0] == 2
    assert dataset.edge_weight.shape[0] == dataset.edge_index.shape[1]
    assert dataset.targets.shape == (4,)


def test_graph_from_io_matrix_rejects_empty_graph():
    with np.testing.assert_raises(ValueError):
        graph_from_io_matrix(np.zeros((2, 2)))


def test_train_and_evaluate_smoke_dataset():
    dataset = make_smoke_dataset()
    model, history = train_model(dataset, epochs=2, hidden_dim=8)
    assert isinstance(model, EconomicESGGNN)
    assert len(history) == 2
    metrics = evaluate_model(model, dataset)
    assert set(metrics) == {"mae", "rmse", "r2"}
