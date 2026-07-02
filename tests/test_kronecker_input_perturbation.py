from __future__ import annotations

import torch

from bochan.api import InputTransformConfig, ModelConfig
from bochan.api.factory import build_model
from bochan.models.regression.gaussian import (
    PerturbationCompatibleKroneckerMultiTaskGP,
)


def _make_bundle(*, n_w: int = 4):
    train_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.8],
            [0.4, 0.3],
            [0.6, 0.7],
            [0.8, 0.2],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = torch.stack(
        [
            torch.sin(train_X[:, 0] * 2.0),
            torch.cos(train_X[:, 1] * 2.0),
        ],
        dim=-1,
    )
    config = ModelConfig(
        task_type="regression",
        model_type="kronecker",
        input_transform_config=InputTransformConfig(
            normalize=True,
            perturbation=True,
            n_w=n_w,
            std=0.05,
        ),
        outcome_transform=False,
    )
    return build_model(train_X, train_Y, config), train_X


def test_kronecker_registry_builds_perturbation_compatible_model() -> None:
    bundle, _ = _make_bundle()

    assert isinstance(
        bundle.model,
        PerturbationCompatibleKroneckerMultiTaskGP,
    )


def test_kronecker_training_inputs_keep_original_row_count() -> None:
    bundle, train_X = _make_bundle(n_w=4)
    model = bundle.model

    model.eval()

    assert model.train_inputs[0].shape[-2] == len(train_X)
    assert model.transform_inputs(model.train_inputs[0]).shape[-2] == len(train_X)


def test_kronecker_candidate_inputs_are_perturbed() -> None:
    bundle, _ = _make_bundle(n_w=4)
    model = bundle.model
    candidates = torch.tensor(
        [[0.25, 0.75], [0.75, 0.25]],
        dtype=torch.double,
    )

    model.eval()
    transformed = model.transform_inputs(candidates)

    assert transformed.shape == torch.Size([8, 2])


def test_kronecker_posterior_runs_with_input_perturbation() -> None:
    bundle, train_X = _make_bundle(n_w=4)
    model = bundle.model
    candidates = torch.tensor(
        [[0.25, 0.75], [0.75, 0.25]],
        dtype=torch.double,
    )

    model.eval()
    posterior = model.posterior(candidates)

    assert model.train_inputs[0].shape[-2] == len(train_X)
    assert posterior.mean.shape[-2:] == torch.Size([8, 2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
