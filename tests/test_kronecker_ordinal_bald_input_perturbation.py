from __future__ import annotations

import torch
from botorch.sampling.normal import SobolQMCNormalSampler

from bochan.acquisition.ordinal.active_learning import qMultiOutputOrdinalBALD
from bochan.acquisition.ordinal.active_learning.bald_compat import (
    _align_pointwise_axes,
)
from bochan.models.ordinal.base import KroneckerMultiTaskOrdinalGPModel
from bochan.models.transforms.input import build_input_transform


def _make_model(*, n_w: int = 4) -> KroneckerMultiTaskOrdinalGPModel:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 2],
            [2, 2],
            [2, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=torch.long,
    )
    input_transform = build_input_transform(
        train_X=train_X,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        perturbation=True,
        n_w=n_w,
        std=0.05,
        normalize=True,
    )
    model = KroneckerMultiTaskOrdinalGPModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        rank=2,
        num_inducing_points=4,
        input_transform=input_transform,
    )
    model.eval()
    model.likelihood.eval()
    return model


def test_align_pointwise_axes_uses_permutation_instead_of_reshape() -> None:
    values = torch.arange(28, dtype=torch.double).reshape(4, 7)
    reference = torch.empty(7, 4, dtype=torch.double)

    aligned = _align_pointwise_axes(
        values,
        reference,
        name="test conditional entropy",
    )

    torch.testing.assert_close(aligned, values.transpose(0, 1))


def test_kronecker_ordinal_bald_supports_input_perturbation_t_batches() -> None:
    model = _make_model(n_w=4)
    acquisition = qMultiOutputOrdinalBALD(
        model=model,
        reduction="mean",
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
    )
    # Matches optimize_acqf initial-condition evaluation: raw_samples t-batches,
    # q=1, with InputPerturbation expanding the point axis to q * n_w.
    X = torch.rand(7, 1, 1, dtype=torch.double)

    value = acquisition(X)

    assert value.shape == torch.Size([7])
    assert torch.isfinite(value).all()
