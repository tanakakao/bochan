from __future__ import annotations

import torch
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from bochan.api import (
    AcquisitionConfig,
    AutoStandardizeOutcomeTransform,
    DataContext,
    ModelConfig,
)
from bochan.api.acquisition.defaults import resolve_multi_output_model_config
from bochan.api.automatic_best_f import compute_best_f
from bochan.api.configs import ModelBundle
from bochan.api.factory import build_model
from bochan.api.registry.model import MODEL_REGISTRY
from bochan.models.regression.gaussian import (
    FidelityFeatureInputTransform,
    WideMixedMultiFidelityGP,
    WideMultiFidelityGP,
    wide_fidelity_to_long,
)


def _wide_data() -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    X = torch.tensor(
        [[0.0, 0.1], [0.5, 0.4], [1.0, 0.8]],
        dtype=torch.double,
    )
    Y = torch.tensor(
        [
            [0.0, 0.2, 0.4],
            [0.5, float("nan"), 0.9],
            [0.8, 1.0, float("nan")],
        ],
        dtype=torch.double,
    )
    return X, Y, [0.25, 0.5, 1.0]


def test_wide_fidelity_to_long_omits_missing_cells() -> None:
    X, Y, fidelity_values = _wide_data()
    X_long, Y_long, Yvar_long = wide_fidelity_to_long(
        X,
        Y,
        fidelity_values,
    )

    assert X_long.shape == torch.Size([7, 3])
    assert Y_long.shape == torch.Size([7, 1])
    assert Yvar_long is None
    torch.testing.assert_close(
        X_long[:, -1],
        torch.tensor(
            [0.25, 0.5, 1.0, 0.25, 1.0, 0.25, 0.5],
            dtype=torch.double,
        ),
    )


def test_wide_fidelity_to_long_converts_known_noise() -> None:
    X, Y, fidelity_values = _wide_data()
    Yvar = torch.full_like(Y, 0.01)
    Yvar[torch.isnan(Y)] = float("nan")

    _, _, Yvar_long = wide_fidelity_to_long(
        X,
        Y,
        fidelity_values,
        train_Yvar=Yvar,
    )

    assert Yvar_long is not None
    assert Yvar_long.shape == torch.Size([7, 1])
    torch.testing.assert_close(Yvar_long, torch.full((7, 1), 0.01, dtype=torch.double))


def test_fidelity_feature_is_not_normalized() -> None:
    transform = FidelityFeatureInputTransform(
        Normalize(
            d=2,
            bounds=torch.tensor(
                [[0.0, 10.0], [10.0, 30.0]],
                dtype=torch.double,
            ),
        ),
        data_dim=2,
    )
    X = torch.tensor(
        [[0.0, 10.0, 0.25], [10.0, 30.0, 1.0]],
        dtype=torch.double,
    )

    transformed = transform(X)

    torch.testing.assert_close(
        transformed[:, :2],
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
    )
    torch.testing.assert_close(transformed[:, -1], X[:, -1])


def test_normal_model_predicts_target_and_all_fidelities() -> None:
    X, Y, fidelity_values = _wide_data()
    model = WideMultiFidelityGP(
        train_X=X,
        train_Y=Y,
        fidelity_values=fidelity_values,
        target_fidelity=1.0,
    )
    model.eval()
    X_test = torch.tensor([[0.25, 0.2], [0.75, 0.6]], dtype=torch.double)

    target_posterior = model.posterior(X_test)
    all_posterior = model.posterior_all_fidelities(X_test)
    explicit_target = model.posterior(
        torch.cat(
            [X_test, torch.ones(X_test.shape[0], 1, dtype=X_test.dtype)],
            dim=-1,
        )
    )

    assert model.num_outputs == 1
    assert model.num_fidelities == 3
    assert model.target_fidelity_index == 2
    assert target_posterior.mean.shape == torch.Size([2, 1])
    assert all_posterior.mean.shape == torch.Size([2, 3])
    assert all_posterior.rsample(torch.Size([4])).shape == torch.Size([4, 2, 3])
    torch.testing.assert_close(target_posterior.mean, explicit_target.mean)

    conditioned = model.condition_on_observations(
        X_test[:1],
        torch.tensor([[0.6]], dtype=torch.double),
    )
    assert conditioned.train_inputs[0].shape[-1] == X.shape[-1] + 1


def test_auto_standardize_uses_one_long_output() -> None:
    X, Y, fidelity_values = _wide_data()
    model = WideMultiFidelityGP(
        train_X=X,
        train_Y=Y,
        fidelity_values=fidelity_values,
        outcome_transform=AutoStandardizeOutcomeTransform(),
    )

    assert isinstance(model.outcome_transform, Standardize)
    assert model.outcome_transform.means.shape[-1] == 1


def test_mixed_model_uses_public_category_indices() -> None:
    X = torch.tensor(
        [
            [0.0, 0.0, 0.1],
            [0.4, 1.0, 0.5],
            [0.8, 2.0, 0.9],
            [1.0, 1.0, 0.2],
        ],
        dtype=torch.double,
    )
    Y = torch.tensor(
        [
            [0.0, 0.2, 0.3],
            [0.4, float("nan"), 0.8],
            [0.7, 0.9, 1.1],
            [0.8, 1.0, float("nan")],
        ],
        dtype=torch.double,
    )
    bounds = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=torch.double,
    )
    model = WideMixedMultiFidelityGP(
        train_X=X,
        train_Y=Y,
        cat_dims=[1],
        fidelity_values=[0.25, 0.5, 1.0],
        input_transform=Normalize(
            d=3,
            bounds=bounds,
            indices=[0, 2],
        ),
    )
    model.eval()

    posterior = model.posterior(X[:2])
    all_posterior = model.posterior_all_fidelities(X[:2])

    assert model.cat_dims == [1]
    assert model.cont_dims == [0, 2]
    assert posterior.mean.shape == torch.Size([2, 1])
    assert all_posterior.mean.shape == torch.Size([2, 3])


def test_registry_and_factory_select_normal_and_mixed_models() -> None:
    X, Y, fidelity_values = _wide_data()
    assert MODEL_REGISTRY["normal"]["regression"]["multifidelity"] is WideMultiFidelityGP
    assert MODEL_REGISTRY["mixed"]["regression"]["multifidelity"] is WideMixedMultiFidelityGP

    normal_bundle = build_model(
        X,
        Y,
        ModelConfig(
            task_type="regression",
            model_type="multifidelity",
            outcome_transform=False,
            model_kwargs={"fidelity_values": fidelity_values},
        ),
    )
    mixed_bundle = build_model(
        torch.cat([X[:, :1], torch.tensor([[0.0], [1.0], [0.0]], dtype=X.dtype), X[:, 1:]], dim=-1),
        Y,
        ModelConfig(
            task_type="regression",
            model_type="multifidelity",
            cat_dims=[1],
            outcome_transform=False,
            model_kwargs={"fidelity_values": fidelity_values},
        ),
    )

    assert isinstance(normal_bundle.model, WideMultiFidelityGP)
    assert isinstance(mixed_bundle.model, WideMixedMultiFidelityGP)


def test_multifidelity_wide_targets_are_not_split_into_model_list() -> None:
    config = ModelConfig(
        task_type="regression",
        model_type="multifidelity",
        outcome_transform=False,
        model_kwargs={"fidelity_values": [0.25, 0.5, 1.0]},
    )
    resolved = resolve_multi_output_model_config(
        config,
        torch.zeros(4, 3, dtype=torch.double),
    )

    assert resolved is config
    assert resolved.multi_output_config is None


def test_automatic_best_f_uses_target_fidelity_only() -> None:
    X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    Y = torch.tensor(
        [[100.0, 0.2], [50.0, 0.7], [25.0, 0.9]],
        dtype=torch.double,
    )
    model = WideMultiFidelityGP(
        train_X=X,
        train_Y=Y,
        fidelity_values=[0.25, 1.0],
        target_fidelity=1.0,
    )
    bundle = ModelBundle(
        model=model,
        train_X=X,
        train_Y=Y,
        model_config=ModelConfig(
            task_type="regression",
            model_type="multifidelity",
            outcome_transform=False,
            model_kwargs={"fidelity_values": [0.25, 1.0]},
        ),
        task_type="regression",
        model_type="multifidelity",
    )

    best_f = compute_best_f(
        bundle,
        AcquisitionConfig(name="EI"),
        DataContext(),
    )

    torch.testing.assert_close(best_f, torch.tensor(0.9, dtype=torch.double))
