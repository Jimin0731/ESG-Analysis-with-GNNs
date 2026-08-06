"""Lightweight orchestration used only by the active synthetic notebooks."""

from .notebook_support import (
    build_example_explanation,
    build_example_model_config,
    build_synthetic_prepared_data,
    reconstruct_trained_model,
    run_example_baseline,
    run_example_training,
)

__all__ = [
    "build_synthetic_prepared_data",
    "build_example_model_config",
    "run_example_training",
    "run_example_baseline",
    "reconstruct_trained_model",
    "build_example_explanation",
]
