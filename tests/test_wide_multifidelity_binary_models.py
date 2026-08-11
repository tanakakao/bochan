from __future__ import annotations

import torch
import pytest
from botorch.models.transforms.input import Normalize

from bochan.acquisition.binary.active_learning import (
    qBinaryBALD,
    qBinaryPredictiveEntropy,
)
from bochan.api import InputTransformConfig, ModelConfig
from bochan.api.engine_defaults import resolve_multi_output_model_config
from bochan.api.factory import build_model
from bochan.api.model_registry import MODEL_REGISTRY
from bochan.models.classification.binary.base import (
    WideMixedMultiFidelityBinaryClassificationGPModel,
    WideMultiFidelityBinaryClassificationGPModel,
)
from bochan.models.regression.gaussian.multifidelity import (
    FidelityFeatureInputTransform,
)


def _normal_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]],
        dtype=torch.double,
    )
    Y = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, float("nan"), 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, float("nan"), 1.0],
            [1.0, 1.0, float("nan")],
        ],
        dtype=torch.double,
    )
    return X, Y


def _mixed_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 1.0],
            [0.4, 2.0],
            [0.6, 0.0],
            [0.8, 1.0],
            [1.0, 2.0],
        ],
        dtype=torch.double,
    )
    Y = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, float("nan"), 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, float("nan"), 1.0],
            [1.0, 1.0, float("nan")],
        ],
        dtype=torch.double,
    )
    return X, Y


def test_binary_multifidelity_registry_and_wide_routing() -> None:
    X, Y = _normal_data()
    config = ModelConfig(
        task_type="binary",
        model_type="multifidelity",
        model_kwargs={
            "fidelity_values": [0.25, 0.5, 1.0],
            "target_fidelity": 1.0,
            "num_inducing_points": 8,
        },
    )
    resolved = resolve_multi_output_model_config(config, Y)
    assert resolved is config
    assert resolved.multi_output_config is None

    assert (
        MODEL_REGISTRY["normal"]["binary"]["multifidelity"]
        is WideMultiFidelityBinaryClassificationGPModel
    )
    assert (
        MODEL_REGISTRY["mixed"]["binary"]["multifidelity"]
        is WideMixedMultiFidelityBinaryClassificationGPModel
    )

    bundle = build_model(X, Y, resolved)
    assert isinstance(
        bundle.model,
        WideMultiFidelityBinaryClassificationGPModel,
    )
    assert bundle.train_X is X
    assert bundle.train_Y is Y
    assert bundle.model.num_outputs == 1
    assert bundle.model.num_fidelities == 3


def test_binary_multifidelity_validates_labels_and_observations() -> None:
    X, Y = _normal_data()

    invalid = Y.clone()
    invalid[0, 0] = 2.0
    with pytest.raises(ValueError, match="0 or 1"):
        WideMultiFidelityBinaryClassificationGPModel(
            train_X=X,
            train_Y=invalid,
            fidelity_values=[0.25, 0.5, 1.0],
        )

    missing_fidelity = Y.clone()
    missing_fidelity[:, 1] = float("nan")
    with pytest.raises(ValueError, match="Every fidelity"):
        WideMultiFidelityBinaryClassificationGPModel(
            train_X=X,
            train_Y=missing_fidelity,
            fidelity_values=[0.25, 0.5, 1.0],
        )


def test_binary_multifidelity_target_and_all_fidelity_posteriors() -> None:
    X, Y = _normal_data()
    model = WideMultiFidelityBinaryClassificationGPModel(
        train_X=X,
        train_Y=Y,
        fidelity_values=[0.25, 0.5, 1.0],
        target_fidelity=1.0,
        num_inducing_points=8,
    )
    model.eval()

    X_test = torch.tensor([[0.3], [0.7]], dtype=torch.double)
    target = model.posterior(X_test)
    low = model.posterior_at_fidelity(X_test, 0.25)
    all_fidelities = model.posterior_all_fidelities(X_test)

    assert target.mean.shape == torch.Size([2, 1])
    assert low.mean.shape == torch.Size([2, 1])
    assert all_fidelities.mean.shape == torch.Size([2, 3])
    assert all_fidelities.variance.shape == torch.Size([2, 3])
    assert all_fidelities.rsample(torch.Size([4])).shape == torch.Size([4, 2, 3])
    assert torch.all((target.mean > 0.0) & (target.mean < 1.0))

    target_probs = model.class_probs(X_test)
    all_probs = model.class_probs_all_fidelities(X_test)
    assert target_probs.shape == torch.Size([2, 2])
    assert all_probs.shape == torch.Size([2, 3, 2])
    torch.testing.assert_close(
        target_probs.sum(dim=-1),
        torch.ones(2, dtype=torch.double),
    )
    torch.testing.assert_close(
        all_probs.sum(dim=-1),
        torch.ones(2, 3, dtype=torch.double),
    )

    latent = model.latent_posterior(X_test)
    latent_all = model.latent_posterior_all_fidelities(X_test)
    assert latent.mean.numel() == 2
    assert latent_all.mean.numel() == 6


def test_binary_multifidelity_input_transform_preserves_fidelity() -> None:
    X, Y = _normal_data()
    transform = Normalize(
        d=1,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
    )
    model = WideMultiFidelityBinaryClassificationGPModel(
        train_X=X,
        train_Y=Y,
        fidelity_values=[0.25, 0.5, 1.0],
        input_transform=transform,
        num_inducing_points=8,
    )
    assert isinstance(model.input_transform, FidelityFeatureInputTransform)

    internal = torch.tensor(
        [[0.2, 0.25], [0.8, 1.0]],
        dtype=torch.double,
    )
    transformed = model.input_transform(internal)
    torch.testing.assert_close(transformed[..., -1], internal[..., -1])


def test_binary_multifidelity_existing_acquisitions_use_target_fidelity() -> None:
    X, Y = _normal_data()
    model = WideMultiFidelityBinaryClassificationGPModel(
        train_X=X,
        train_Y=Y,
        fidelity_values=[0.25, 0.5, 1.0],
        target_fidelity=1.0,
        num_inducing_points=8,
    )
    model.eval()

    Xq = torch.tensor(
        [[[0.25], [0.5], [0.75]]],
        dtype=torch.double,
        requires_grad=True,
    )
    for acquisition in (
        qBinaryPredictiveEntropy(model=model),
        qBinaryBALD(model=model, num_samples=8),
    ):
        value = acquisition(Xq)
        gradient = torch.autograd.grad(
            value.sum(),
            Xq,
            retain_graph=True,
        )[0]
        assert value.shape == torch.Size([1])
        assert torch.isfinite(value).all()
        assert gradient.shape == Xq.shape
        assert torch.isfinite(gradient).all()


def test_binary_multifidelity_perturbation_wide_shape() -> None:
    X, Y = _normal_data()
    bundle = build_model(
        X,
        Y,
        ModelConfig(
            task_type="binary",
            model_type="multifidelity",
            model_kwargs={
                "fidelity_values": [0.25, 0.5, 1.0],
                "num_inducing_points": 8,
            },
            input_transform_config=InputTransformConfig(
                normalize=True,
                perturbation=True,
                n_w=2,
                std=0.01,
            ),
        ),
    )
    model = bundle.model
    model.eval()

    X_test = torch.tensor([[0.3], [0.7]], dtype=torch.double)
    target = model.posterior(X_test)
    all_fidelities = model.posterior_all_fidelities(X_test)
    assert target.mean.shape == torch.Size([4, 1])
    assert all_fidelities.mean.shape == torch.Size([4, 3])


def test_mixed_binary_multifidelity_preserves_categories_and_fidelity() -> None:
    X, Y = _mixed_data()
    transform = Normalize(
        d=2,
        bounds=torch.tensor(
            [[0.0, 0.0], [1.0, 2.0]],
            dtype=torch.double,
        ),
        indices=[0],
    )
    model = WideMixedMultiFidelityBinaryClassificationGPModel(
        train_X=X,
        train_Y=Y,
        cat_dims=[1],
        fidelity_values=[0.25, 0.5, 1.0],
        target_fidelity=1.0,
        input_transform=transform,
        num_inducing=8,
    )
    model.eval()

    X_test = torch.tensor(
        [[0.3, 1.0], [0.7, 2.0]],
        dtype=torch.double,
    )
    target = model.posterior(X_test)
    all_fidelities = model.posterior_all_fidelities(X_test)
    assert target.mean.shape == torch.Size([2, 1])
    assert all_fidelities.mean.shape == torch.Size([2, 3])
    assert model.cat_dims == [1]
    assert model.cont_dims == [0]

    internal = torch.cat(
        [
            X_test,
            torch.tensor([[0.25], [1.0]], dtype=torch.double),
        ],
        dim=-1,
    )
    transformed = model.input_transform(internal)
    torch.testing.assert_close(transformed[:, 1], internal[:, 1])
    torch.testing.assert_close(transformed[:, -1], internal[:, -1])


def test_binary_multifidelity_conditioning_public_and_explicit_fidelity() -> None:
    X, Y = _normal_data()
    model = WideMultiFidelityBinaryClassificationGPModel(
        train_X=X,
        train_Y=Y,
        fidelity_values=[0.25, 0.5, 1.0],
        target_fidelity=1.0,
        num_inducing_points=8,
    )

    conditioned = model.condition_on_observations(
        X=torch.tensor([[0.15], [0.85]], dtype=torch.double),
        Y=torch.tensor([0.0, 1.0], dtype=torch.double),
    )
    assert conditioned.train_X_wide.shape[0] == X.shape[0] + 2
    torch.testing.assert_close(
        conditioned.train_Y_wide[-2:, -1],
        torch.tensor([0.0, 1.0], dtype=torch.double),
    )

    conditioned_low = conditioned.condition_on_observations(
        X=torch.tensor([[0.55, 0.25]], dtype=torch.double),
        Y=torch.tensor([1.0], dtype=torch.double),
    )
    assert conditioned_low.train_X_wide.shape[0] == X.shape[0] + 3
    assert conditioned_low.train_Y_wide[-1, 0].item() == 1.0
    assert torch.isnan(conditioned_low.train_Y_wide[-1, 1:]).all()
    assert conditioned_low.posterior(
        torch.tensor([[0.5]], dtype=torch.double)
    ).mean.shape == torch.Size([1, 1])
