"""Experiment-failure model fitting shared by core and tabular workflows."""

from __future__ import annotations

from typing import Any

from .configs import FitConfig, ModelConfig
from .factory import build_model, fit_model
from .observation import ExperimentFailureConfig, ObservationData


def default_failure_model_config(objective_config: ModelConfig) -> ModelConfig:
    """Create a binary GP config sharing only meaningful input-side settings."""

    return ModelConfig(
        task_type="binary",
        model_type="base",
        input_type=objective_config.input_type,
        cat_dims=objective_config.cat_dims,
        input_transform_config=objective_config.input_transform_config,
        outcome_transform=False,
    )


def attach_observation_state(
    optimizer: Any,
    observations: ObservationData,
    *,
    failure_config: ExperimentFailureConfig | None = None,
) -> None:
    """Attach canonical observations and fit the independent success model.

    The objective model is assumed to have already been fitted from the canonical
    objective training rows. This function never modifies or imputes objective Y.
    Failure-model fitting is deliberately independent from the objective model's
    fit configuration unless an explicit failure ``fit_config`` is supplied.
    """

    optimizer.observations = observations
    optimizer.failure_config = failure_config
    if optimizer.bundle is not None:
        optimizer.bundle.metadata["observation"] = observations.report()

    optimizer.failure_bundle = None
    optimizer.failure_model = None
    if failure_config is None:
        if optimizer.bundle is not None:
            optimizer.bundle.metadata["experiment_failure_model"] = {
                "enabled": False,
                "reason": "not_configured",
            }
        return
    if not bool(observations.failed_mask.any()):
        if optimizer.bundle is not None:
            optimizer.bundle.metadata["experiment_failure_model"] = {
                "enabled": False,
                "reason": "no_failed_experiments",
            }
        return

    success_X, success_Y = observations.success_training_data()
    objective_config = optimizer.model_config
    failure_model_config = (
        failure_config.model_config
        or default_failure_model_config(objective_config)
    )
    failure_fit_config = failure_config.fit_config or FitConfig()
    failure_bundle = build_model(
        train_X=success_X,
        train_Y=success_Y,
        config=failure_model_config,
        model_registry=optimizer.model_registry,
    )
    failure_bundle = fit_model(failure_bundle, failure_fit_config)
    optimizer.failure_bundle = failure_bundle
    optimizer.failure_model = failure_bundle.model
    if optimizer.bundle is not None:
        optimizer.bundle.metadata["experiment_failure_model"] = {
            "enabled": True,
            "model_type": str(failure_model_config.model_type),
            "n_completed": int(success_X.shape[0]),
            "n_failed": int(observations.failed_mask.sum().item()),
        }


__all__ = ["attach_observation_state", "default_failure_model_config"]
