from __future__ import annotations

import torch
from botorch.models import SingleTaskGP
from botorch.utils.multi_objective.box_decompositions import (
    FastNondominatedPartitioning,
)

from bochan.acquisition.regression.bayesian_optimization import (
    qHeteroMultiOutputRegressionExpectedHypervolumeImprovement,
)


def _make_problem():
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_Y = torch.tensor(
        [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]],
        dtype=torch.double,
    )
    model = SingleTaskGP(train_X, train_Y)
    ref_point = torch.tensor([-0.1, -0.1], dtype=torch.double)
    partitioning = FastNondominatedPartitioning(
        ref_point=ref_point,
        Y=train_Y,
    )
    return model, ref_point, partitioning


def test_hetero_ehvi_without_constraints_keeps_none_and_runs_forward() -> None:
    model, ref_point, partitioning = _make_problem()
    acqf = qHeteroMultiOutputRegressionExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
    )

    assert acqf.constraints is None

    value = acqf(torch.tensor([[0.25]], dtype=torch.double))

    assert torch.isfinite(value).all()


def test_hetero_ehvi_accepts_float_eta_with_constraints() -> None:
    model, ref_point, partitioning = _make_problem()

    acqf = qHeteroMultiOutputRegressionExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        constraints=[lambda samples: samples[..., 0] - 0.8],
        eta=1e-3,
    )

    assert torch.is_tensor(acqf.eta)
    assert acqf.eta.dtype == torch.double
    assert float(acqf.eta.reshape(-1)[0]) == 1e-3

    value = acqf(torch.tensor([[0.25]], dtype=torch.double))

    assert torch.isfinite(value).all()
