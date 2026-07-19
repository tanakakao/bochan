from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from bochan.acquisition.multiclass.bayesian_optimization import (
    qHeteroMultiOutputMulticlassNParEGO,
)
from bochan.acquisition.multiclass.bayesian_optimization.nparego_observed_baseline import (
    configure_hetero_multiclass_nparego_observed_baseline,
)


def test_public_hetero_multiclass_nparego_accepts_observed_baseline_kwargs() -> None:
    parameters = inspect.signature(
        qHeteroMultiOutputMulticlassNParEGO,
    ).parameters

    assert "Y_baseline" in parameters
    assert "train_Y" in parameters
    assert "utility_values" in parameters
    assert "objective_signs" in parameters


def test_hetero_multiclass_nparego_uses_observed_objective_baseline() -> None:
    converted = torch.tensor(
        [[0.2, 0.3], [0.8, 0.9]],
        dtype=torch.double,
    )
    calls = []

    class _Objective:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _Acquisition(torch.nn.Module):
        def __init__(
            self,
            model,
            X_baseline,
            ref_point,
            *,
            target_class=None,
            output_target_classes=None,
            class_reduction="mean",
            weights=None,
            sampler=None,
            objective=None,
            rho=0.05,
            noise_mode="inverse_linear",
            noise_combine="multiply",
            noise_penalty_lambda=1.0,
            noise_min_weight=0.0,
            noise_weight_scale=1.0,
            noise_model_outputs_log_var=True,
            eps=1e-8,
        ) -> None:
            del (
                model,
                X_baseline,
                ref_point,
                target_class,
                output_target_classes,
                class_reduction,
                weights,
                sampler,
                rho,
                noise_mode,
                noise_combine,
                noise_penalty_lambda,
                noise_min_weight,
                noise_weight_scale,
                noise_model_outputs_log_var,
                eps,
            )
            super().__init__()
            self.objective = objective
            self.register_buffer(
                "best_value",
                torch.tensor(-1.0, dtype=torch.double),
            )

        @staticmethod
        def _scalarize(values: torch.Tensor) -> torch.Tensor:
            return values.sum(dim=-1)

    def _compute_observed_multiclass_utility(**kwargs):
        calls.append(kwargs)
        return converted

    module = SimpleNamespace(
        MulticlassTargetProbabilityObjective=_Objective,
        compute_observed_multiclass_utility=_compute_observed_multiclass_utility,
    )
    configure_hetero_multiclass_nparego_observed_baseline(
        module,
        _Acquisition,
    )

    raw_train_Y = torch.tensor(
        [[0.0, 1.0], [2.0, 2.0]],
        dtype=torch.double,
    )
    model = SimpleNamespace(train_Y=raw_train_Y)
    acquisition = _Acquisition(
        model=model,
        X_baseline=torch.rand(2, 3, dtype=torch.double),
        ref_point=torch.zeros(2, dtype=torch.double),
        Y_baseline=raw_train_Y,
        utility_values=[0.0, 0.5, 1.0],
        objective_signs=[1.0, -1.0],
    )

    assert len(calls) == 1
    assert torch.equal(calls[0]["train_Y"], raw_train_Y)
    assert torch.allclose(acquisition.best_value, torch.tensor(1.7, dtype=torch.double))
    assert acquisition.objective.kwargs["utility_values"] == [0.0, 0.5, 1.0]
    assert acquisition.objective.kwargs["objective_signs"] == [1.0, -1.0]
