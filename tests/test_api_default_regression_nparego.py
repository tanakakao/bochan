from __future__ import annotations

from types import SimpleNamespace

import torch
from botorch.acquisition.monte_carlo import qExpectedImprovement

from bochan.acquisition.objective import RegressionLinearMCObjective
from bochan.acquisition.regression.bayesian_optimization import (
    qMultiOutputRegressionNParEGO,
)
from bochan.api import AcquisitionConfig, DataContext, ModelBundle, ModelConfig
from bochan.api.engine_defaults import resolve_acquisition_defaults


def _make_bundle(train_Y: torch.Tensor) -> ModelBundle:
    train_X = torch.linspace(0.0, 1.0, train_Y.shape[0], dtype=torch.double).unsqueeze(-1)
    return ModelBundle(
        model=SimpleNamespace(),
        train_X=train_X,
        train_Y=train_Y,
        model_config=ModelConfig(
            task_type="regression",
            model_type="base",
            outcome_transform=False,
        ),
        task_type="regression",
        model_type="base",
    )


def test_multi_output_regression_nparego_alias_uses_custom_acquisition() -> None:
    bundle = _make_bundle(
        torch.tensor(
            [[1.0, 3.0], [2.0, 2.0], [0.0, 4.0]],
            dtype=torch.double,
        )
    )
    objective = RegressionLinearMCObjective(
        output_indices=[0, 1],
        weights=[1.0, 1.0],
        signs=[1.0, -1.0],
    )

    resolved, context = resolve_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="nparego",
            acqf_cls=qExpectedImprovement,
            objective=objective,
        ),
        DataContext(X_baseline=bundle.train_X),
    )

    assert resolved.acqf_cls is qMultiOutputRegressionNParEGO
    assert resolved.objective is objective
    assert context.X_baseline is bundle.train_X
    torch.testing.assert_close(
        context.ref_point,
        torch.tensor([-0.1, -4.1], dtype=torch.double),
    )


def test_qnparego_alias_also_uses_custom_acquisition() -> None:
    bundle = _make_bundle(
        torch.tensor(
            [[1.0, 3.0], [2.0, 2.0], [0.0, 4.0]],
            dtype=torch.double,
        )
    )

    resolved, _ = resolve_acquisition_defaults(
        bundle,
        AcquisitionConfig(name="qnparego", acqf_cls=qExpectedImprovement),
        DataContext(X_baseline=bundle.train_X),
    )

    assert resolved.acqf_cls is qMultiOutputRegressionNParEGO


def test_single_output_regression_nparego_keeps_standard_acquisition() -> None:
    bundle = _make_bundle(
        torch.tensor([[1.0], [2.0], [0.0]], dtype=torch.double)
    )

    resolved, _ = resolve_acquisition_defaults(
        bundle,
        AcquisitionConfig(name="nparego", acqf_cls=qExpectedImprovement),
        DataContext(X_baseline=bundle.train_X),
    )

    assert resolved.acqf_cls is qExpectedImprovement
