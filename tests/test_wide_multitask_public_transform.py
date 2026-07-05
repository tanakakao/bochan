from __future__ import annotations

import torch
from botorch.models.transforms.input import Normalize

from bochan.acquisition.regression.active_learning import qMultiOutputRegressionBALD
from bochan.models.wide_multitask_compat import (
    TaskFeatureInputTransform,
    WideMultiTaskGP,
)


def test_task_feature_transform_accepts_public_and_internal_inputs() -> None:
    transform = TaskFeatureInputTransform(
        Normalize(
            d=2,
            bounds=torch.tensor(
                [[0.0, 10.0], [10.0, 30.0]],
                dtype=torch.double,
            ),
        ),
        data_dim=2,
    )
    public_X = torch.tensor(
        [[0.0, 10.0], [10.0, 30.0]],
        dtype=torch.double,
    )
    internal_X = torch.tensor(
        [[0.0, 10.0, 0.0], [10.0, 30.0, 1.0]],
        dtype=torch.double,
    )

    public_transformed = transform(public_X)
    internal_transformed = transform(internal_X)

    expected_public = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0]],
        dtype=torch.double,
    )
    torch.testing.assert_close(public_transformed, expected_public)
    torch.testing.assert_close(internal_transformed[:, :2], expected_public)
    torch.testing.assert_close(internal_transformed[:, -1], internal_X[:, -1])
    torch.testing.assert_close(transform.untransform(public_transformed), public_X)
    torch.testing.assert_close(transform.untransform(internal_transformed), internal_X)


def test_wide_multitask_bald_uses_public_distance_transform() -> None:
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
            train_X[:, 0] + 0.2 * train_X[:, 1],
            1.0 - train_X[:, 0] + 0.1 * train_X[:, 1],
        ],
        dim=-1,
    )
    model = WideMultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=Normalize(
            d=2,
            bounds=torch.tensor(
                [[0.0, 0.0], [1.0, 1.0]],
                dtype=torch.double,
            ),
        ),
    )
    model.eval()
    acquisition = qMultiOutputRegressionBALD(
        model=model,
        same_batch_penalty_weight=0.1,
        observed_penalty_weight=0.1,
        X_observed=train_X,
    )
    Xq = torch.tensor(
        [[[0.15, 0.25], [0.50, 0.50], [0.85, 0.75]]],
        dtype=torch.double,
        requires_grad=True,
    )

    value = acquisition(Xq)
    gradient = torch.autograd.grad(value.sum(), Xq)[0]

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()
    assert gradient.shape == Xq.shape
    assert torch.isfinite(gradient).all()
