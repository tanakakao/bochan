from __future__ import annotations

import torch

from bochan.api.model_registry import MODEL_REGISTRY
from bochan.models.multitask.mixed import WideMixedMultiTaskGP


def _mixed_partial_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]],
        dtype=torch.double,
    )
    Y = torch.tensor(
        [[1.0, float("nan")], [float("nan"), 2.0], [3.0, 4.0]],
        dtype=torch.double,
    )
    return X, Y


def test_mixed_multitask_registry_uses_mixed_wide_model() -> None:
    assert (
        MODEL_REGISTRY["mixed"]["regression"]["multitask"]
        is WideMixedMultiTaskGP
    )
    assert (
        MODEL_REGISTRY["mixed"]["multi_objective"]["multitask"]
        is WideMixedMultiTaskGP
    )


def test_mixed_wide_multitask_preserves_partial_targets() -> None:
    X, Y = _mixed_partial_data()
    model = WideMixedMultiTaskGP(
        train_X=X,
        train_Y=Y,
        cat_dims=[1],
        outcome_transform=None,
    )
    model.eval()

    assert model.cat_dims == [1]
    assert torch.isnan(model.train_Y_wide).sum().item() == 2
    posterior = model.posterior(
        torch.tensor([[0.25, 0.0], [0.75, 1.0]], dtype=torch.double)
    )
    assert posterior.mean.shape == torch.Size([2, 2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
