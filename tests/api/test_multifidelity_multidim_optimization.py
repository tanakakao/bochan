from __future__ import annotations

import torch
from botorch.acquisition.analytic import PosteriorMean

from bochan.api.configs import OptimizeConfig
from bochan.api.optimizer.dispatch import optimize_candidates
from bochan.models.multifidelity import FidelitySpec
from bochan.models.regression.gaussian.long_multifidelity import GaussianMultiFidelityGP
from bochan.serving.fastapi.schemas.configs import OptimizeConfigSchema
from bochan.serving.fastapi.schemas.requests import CandidateRequest


def _model() -> GaussianMultiFidelityGP:
    train_X = torch.tensor(
        [
            [0.00, 0.25, 0.50],
            [0.20, 0.25, 1.00],
            [0.40, 0.50, 0.50],
            [0.60, 0.50, 1.00],
            [0.80, 1.00, 0.50],
            [1.00, 1.00, 1.00],
        ],
        dtype=torch.double,
    )
    x = train_X[:, :1]
    f1 = train_X[:, 1:2]
    f2 = train_X[:, 2:3]
    train_Y = 1.0 - (x - 0.65).square() + 0.15 * f1 + 0.10 * f2
    return GaussianMultiFidelityGP(
        train_X,
        train_Y,
        fidelity_spec=FidelitySpec(
            fidelity_features=(-2, -1),
            target_fidelities={-2: 1.0, -1: 1.0},
        ),
    )


def _bounds() -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=torch.double,
    )


def test_multidimensional_discrete_fidelity_numerical_optimization():
    torch.manual_seed(0)
    model = _model()
    acqf = PosteriorMean(model)
    candidates, value = optimize_candidates(
        acqf,
        _bounds(),
        OptimizeConfig(
            fidelity_values={-2: [0.25, 1.0], -1: [0.5, 1.0]},
            num_restarts=2,
            raw_samples=16,
            ensure_unique_candidates=False,
        ),
    )

    assert candidates.shape == torch.Size([1, 3])
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(value).all()
    assert float(candidates[0, 1]) in {0.25, 1.0}
    assert float(candidates[0, 2]) in {0.5, 1.0}


def test_multidimensional_explicit_assignment_numerical_optimization():
    torch.manual_seed(0)
    model = _model()
    acqf = PosteriorMean(model)
    candidates, value = optimize_candidates(
        acqf,
        _bounds(),
        OptimizeConfig(
            fidelity_assignments=[{-2: 0.25, -1: 0.5}, {-2: 1.0, -1: 1.0}],
            num_restarts=2,
            raw_samples=16,
            ensure_unique_candidates=False,
        ),
    )

    pair = (float(candidates[0, 1]), float(candidates[0, 2]))
    assert pair in {(0.25, 0.5), (1.0, 1.0)}
    assert torch.isfinite(value).all()


def test_multidimensional_continuous_fidelity_numerical_optimization():
    torch.manual_seed(0)
    model = _model()
    acqf = PosteriorMean(model)
    candidates, value = optimize_candidates(
        acqf,
        _bounds(),
        OptimizeConfig(
            optimize_fidelity=True,
            num_restarts=2,
            raw_samples=16,
            ensure_unique_candidates=False,
        ),
    )

    assert candidates.shape == torch.Size([1, 3])
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(value).all()
    assert bool(((candidates[0, 1:] >= 0.0) & (candidates[0, 1:] <= 1.0)).all())


def test_fastapi_optimize_schema_accepts_multidimensional_modes():
    values = OptimizeConfigSchema(
        fidelity_values={-2: [0.25, 1.0], -1: [0.5, 1.0]},
    )
    assert values.fidelity_values == {-2: [0.25, 1.0], -1: [0.5, 1.0]}

    assignments = OptimizeConfigSchema(
        fidelity_assignments=[{-2: 0.25, -1: 0.5}, {-2: 1.0, -1: 1.0}],
    )
    assert assignments.fidelity_assignments == [
        {-2: 0.25, -1: 0.5},
        {-2: 1.0, -1: 1.0},
    ]


def test_fastapi_candidate_convenience_accepts_json_string_indices():
    request = CandidateRequest.model_validate(
        {
            "acquisition_config": {"name": "qei"},
            "fidelity_values": {"-2": [0.25, 1.0], "-1": [0.5, 1.0]},
        }
    )
    assert request.fidelity_values == {-2: [0.25, 1.0], -1: [0.5, 1.0]}
