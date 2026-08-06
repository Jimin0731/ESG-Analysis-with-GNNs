"""Deterministic synthetic fixtures and thin calls into canonical public APIs."""
from __future__ import annotations

import torch

from src.evaluation import build_shock_explanation
from src.experiments import (
    EarlyStoppingConfig,
    PreparedSupervisedData,
    SchedulerConfig,
    SupervisedSnapshot,
    TrainMeanBaseline,
    TrainingConfig,
    train_supervised_model,
)
from src.models import ModelConfig, build_model

FEATURE_NAMES = ("safe__activity", "safe__intensity")
TARGET_NAMES = ("target__environment", "target__social")
NODE_IDS = ("sector_a", "sector_b", "sector_c")


def build_synthetic_prepared_data(seed: int = 17) -> PreparedSupervisedData:
    """Return five ordered snapshots containing invented, deterministic values."""
    generator = torch.Generator().manual_seed(seed)
    edge_index = torch.tensor([[0, 0, 2, 1], [1, 2, 1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([2.0, 0.4, 1.7, 0.8])
    snapshots = []
    for offset, (period, split) in enumerate(
        ((2018, "train"), (2019, "train"), (2020, "train"),
         (2021, "validation"), (2022, "test"))
    ):
        x = torch.rand((3, 2), generator=generator) + offset * 0.1
        y = torch.stack((1.2 * x[:, 0] + 0.2, -0.7 * x[:, 1] + 1.0), dim=1)
        mask = torch.ones_like(y, dtype=torch.bool)
        if period == 2019:
            mask[1, 1] = False
            y[1, 1] = torch.nan
        snapshots.append(SupervisedSnapshot(
            period, split, NODE_IDS, x, y, mask, edge_index.clone(),
            edge_weight.clone(), FEATURE_NAMES, TARGET_NAMES,
        ))
    return PreparedSupervisedData(tuple(snapshots[:3]), (snapshots[3],),
                                  (snapshots[4],), FEATURE_NAMES, TARGET_NAMES)


def build_example_model_config(model_name: str = "weighted_gat", seed: int = 17) -> ModelConfig:
    """Create one small canonical model configuration."""
    kwargs = {"attention_heads": 2} if model_name in {"gat", "weighted_gat"} else {}
    if model_name == "gpr_gnn":
        kwargs.update(alpha=0.2, propagation_steps=2)
    return ModelConfig(model_name, 2, 4, 2, 0.0, TARGET_NAMES, seed=seed, **kwargs)


def run_example_training(model_name: str = "weighted_gat", *, epochs: int = 3,
                         seed: int = 17, prepared_data=None):
    """Train through the canonical chronological PR 8 workflow."""
    data = prepared_data or build_synthetic_prepared_data(seed)
    return train_supervised_model(
        build_example_model_config(model_name, seed), data,
        TrainingConfig(epochs=epochs, learning_rate=0.01, seed=seed),
        EarlyStoppingConfig(patience=2, restore_best=True),
        SchedulerConfig(name="reduce_on_plateau", plateau_patience=1),
    )


def run_example_baseline(prepared_data=None):
    """Fit and evaluate the canonical non-neural train-mean baseline."""
    data = prepared_data or build_synthetic_prepared_data()
    return TrainMeanBaseline().fit(data.train).evaluate(data)


def reconstruct_trained_model(result):
    """Reconstruct the exact PR 9 model configuration and restore its best state."""
    config = ModelConfig(**result.model_configuration)
    model = build_model(config)
    model.load_state_dict(result.best_state)
    model.eval()
    return model


def build_example_explanation(model, snapshot):
    """Build the canonical PR 9 explanation bundle without exporting files."""
    return build_shock_explanation(
        model, snapshot, source_node_id="sector_a",
        feature_changes={FEATURE_NAMES[0]: 0.25},
        feature_ablation=(FEATURE_NAMES[1],),
        edge_ablation={"edge_positions": (0,), "mode": "zero_weight"},
        top_k=4,
    )
