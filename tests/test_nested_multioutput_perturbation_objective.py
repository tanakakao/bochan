from __future__ import annotations

import torch

from bochan.acquisition.objective.regression import (
    MultiOutputRegressionInputPerturbationObjective,
    RegressionLinearMCObjective,
)
from bochan.api.kronecker_input_perturbation_defaults import (
    install_kronecker_input_perturbation_objective_defaults,
)


def _make_objective(*, n_w: int = 4):
    install_kronecker_input_perturbation_objective_defaults()
    inner = RegressionLinearMCObjective(
        output_indices=[0, 1],
        weights=[1.0, 1.0],
        signs=[1.0, 1.0],
    )
    outer = MultiOutputRegressionInputPerturbationObjective(
        inner_objective=inner,
        n_w=n_w,
        risk_type=None,
    )
    return inner, outer


def test_inner_q_shape_check_is_deferred_to_outer_objective() -> None:
    inner, outer = _make_objective(n_w=4)
    samples = torch.arange(
        3 * 2 * 4 * 2,
        dtype=torch.double,
    ).reshape(3, 2, 4, 2)
    X = torch.rand(2, 1, 2, dtype=torch.double)

    values = outer(samples=samples, X=X)
    expected = samples.reshape(3, 2, 1, 4, 2).mean(dim=-2)

    assert inner._verify_output_shape is False
    assert values.shape == torch.Size([3, 2, 1, 2])
    assert torch.allclose(values, expected)


def test_unexpanded_baseline_is_not_aggregated_accidentally() -> None:
    _, outer = _make_objective(n_w=4)
    samples = torch.rand(3, 6, 2, dtype=torch.double)
    X = torch.rand(6, 2, dtype=torch.double)

    values = outer(samples=samples, X=X)

    assert values.shape == samples.shape
    assert torch.allclose(values, samples)


def test_expanded_baseline_is_aggregated_by_n_w() -> None:
    _, outer = _make_objective(n_w=4)
    samples = torch.arange(
        3 * 24 * 2,
        dtype=torch.double,
    ).reshape(3, 24, 2)
    X = torch.rand(6, 2, dtype=torch.double)

    values = outer(samples=samples, X=X)
    expected = samples.reshape(3, 6, 4, 2).mean(dim=-2)

    assert values.shape == torch.Size([3, 6, 2])
    assert torch.allclose(values, expected)
