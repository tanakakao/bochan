from __future__ import annotations

"""Utilities for binary classification multi-output variant tests."""

from typing import Any, Callable

import torch

from bochan.models.classification.binary.base import MultiOutputBinaryClassificationModel


SubmodelAssertFn = Callable[..., None]


def assert_multi_output_wrapper_training(
    *,
    model: MultiOutputBinaryClassificationModel,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    cat_dims: list[int],
    submodel_assert_fn: SubmodelAssertFn,
    submodel_assert_extra_kwargs: dict[str, Any] | None = None,
) -> None:
    """MultiOutputBinaryClassificationModel と各 submodel の基本状態を確認する。"""
    model.eval()
    n, m = train_y.shape
    submodel_assert_extra_kwargs = submodel_assert_extra_kwargs or {}

    assert model.num_outputs == m
    assert len(model.models) == m
    assert model.batch_shape == torch.Size([])
    assert model.num_classes_list == [2 for _ in range(m)]
    assert list(getattr(model, "cat_dims", [])) == cat_dims
    assert model.train_inputs[0].shape == train_x.shape
    assert torch.allclose(model.train_inputs[0], train_x)
    assert torch.allclose(model.train_targets, train_y)
    assert torch.allclose(model.train_Y, train_y)

    for j, submodel in enumerate(model.models):
        submodel_assert_fn(
            model=submodel,
            train_x=train_x,
            train_y=train_y[:, [j]],
            cat_dims=cat_dims,
            **submodel_assert_extra_kwargs,
        )

    with torch.no_grad():
        posterior = model.posterior(train_x)
        prob_post = model.probability_posterior(train_x)
        latent_post = model.latent_posterior(train_x)
        mean_probability = model.mean_probability(train_x)
        probability_variance = model.probability_variance(train_x)
        class_probs = model.class_probs(train_x)
        pred_class = model.predict_class(train_x)
        subset_post = model.posterior(train_x, output_indices=[0, m - 1])
        subset_latent = model.latent_posterior(train_x, output_indices=[0])

    assert posterior.mean.shape == torch.Size([n, m])
    assert posterior.variance.shape == torch.Size([n, m])
    assert prob_post.mean.shape == torch.Size([n, m])
    assert mean_probability.shape == torch.Size([n, m])
    assert probability_variance.shape == torch.Size([n, m])
    assert class_probs.shape == torch.Size([n, m, 2])
    assert pred_class.shape == torch.Size([n, m])
    assert subset_post.mean.shape == torch.Size([n, 2])
    assert subset_latent.mean.shape == torch.Size([n, 1])

    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert torch.isfinite(latent_post.mean).all()
    assert (posterior.mean >= 0.0).all() and (posterior.mean <= 1.0).all()
    assert torch.allclose(class_probs.sum(dim=-1), torch.ones_like(posterior.mean))
    assert torch.isin(pred_class, torch.tensor([0, 1], device=pred_class.device)).all()
