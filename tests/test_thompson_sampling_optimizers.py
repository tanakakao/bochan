from __future__ import annotations

import torch
from botorch.models import SingleTaskGP

from bochan.optim import (
    optimize_thompson_sampling,
    optimize_thompson_sampling_mixed,
)


def _make_model() -> SingleTaskGP:
    train_X = torch.tensor(
        [[0.0, 0.0], [0.25, 1.0], [0.75, 0.0], [1.0, 1.0]],
        dtype=torch.double,
    )
    train_Y = (train_X[:, :1] - 0.5).square() + 0.1 * train_X[:, 1:2]
    return SingleTaskGP(train_X=train_X, train_Y=train_Y)


def test_optimize_thompson_sampling_returns_q_candidates() -> None:
    model = _make_model()
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    candidate_set = torch.tensor(
        [[0.1, 0.2], [0.3, 0.4], [0.6, 0.7], [0.9, 0.8]],
        dtype=torch.double,
    )

    candidates, values = optimize_thompson_sampling(
        acq_function=model,
        bounds=bounds,
        q=2,
        options={
            "candidate_set": candidate_set,
            "replacement": False,
        },
    )

    assert candidates.shape == torch.Size([2, 2])
    assert values.shape == torch.Size([2])
    assert all(
        torch.any(torch.all(torch.isclose(candidate_set, candidate), dim=-1))
        for candidate in candidates
    )


def test_optimize_thompson_sampling_applies_fixed_features_and_constraints() -> None:
    model = _make_model()
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    candidate_set = torch.tensor(
        [[0.1, 0.1], [0.3, 0.2], [0.7, 0.3], [0.9, 0.4]],
        dtype=torch.double,
    )

    candidates, _ = optimize_thompson_sampling(
        acq_function=model,
        bounds=bounds,
        q=2,
        fixed_features={1: 0.5},
        inequality_constraints=[
            (
                torch.tensor([0]),
                torch.tensor([1.0], dtype=torch.double),
                0.5,
            )
        ],
        options={"candidate_set": candidate_set},
    )

    assert torch.all(candidates[:, 0] >= 0.5)
    assert torch.allclose(candidates[:, 1], torch.full((2,), 0.5, dtype=torch.double))


def test_optimize_thompson_sampling_mixed_uses_fixed_feature_combinations() -> None:
    model = _make_model()
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    base_candidate_set = torch.tensor(
        [[0.1, 0.25], [0.4, 0.25], [0.7, 0.25], [0.9, 0.25]],
        dtype=torch.double,
    )

    candidates, values = optimize_thompson_sampling_mixed(
        acq_function=model,
        bounds=bounds,
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        q=3,
        options={
            "candidate_set": base_candidate_set,
            "replacement": False,
        },
    )

    assert candidates.shape == torch.Size([3, 2])
    assert values.shape == torch.Size([3])
    assert torch.all((candidates[:, 1] == 0.0) | (candidates[:, 1] == 1.0))
