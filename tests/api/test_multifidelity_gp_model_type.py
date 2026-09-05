from __future__ import annotations

import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from bochan.api import BayesianOptimizer, FitConfig, ModelConfig
from bochan.api.modeling.build import build_model, resolve_model_cls
from bochan.api.modeling.fit import fit_model
from bochan.models.multifidelity import FidelitySpec
from bochan.models.regression.gaussian import (
    GaussianMixedMultiFidelityGP,
    GaussianMultiFidelityGP,
)


def _normal_data():
    train_X = torch.tensor(
        [
            [0.0, 0.25],
            [0.0, 1.00],
            [0.5, 0.50],
            [0.5, 1.00],
            [1.0, 0.25],
            [1.0, 1.00],
        ],
        dtype=torch.double,
    )
    train_Y = torch.sin(train_X[:, :1] * 2.0) + 0.2 * train_X[:, 1:2]
    return train_X, train_Y


def _mixed_data():
    train_X = torch.tensor(
        [
            [0.0, 0.0, 0.25],
            [0.0, 1.0, 1.00],
            [0.5, 0.0, 0.50],
            [0.5, 1.0, 1.00],
            [1.0, 0.0, 0.25],
            [1.0, 1.0, 1.00],
        ],
        dtype=torch.double,
    )
    train_Y = (
        torch.sin(train_X[:, :1] * 2.0)
        + train_X[:, 1:2]
        + 0.2 * train_X[:, 2:3]
    )
    return train_X, train_Y


def _identity_fit(target, **kwargs):
    return target


def test_registry_resolves_multifidelity_gp_for_normal_and_mixed():
    normal = resolve_model_cls(
        ModelConfig(
            task_type="regression",
            model_type="multifidelity_gp",
            input_type="normal",
            outcome_transform=False,
        )
    )
    mixed = resolve_model_cls(
        ModelConfig(
            task_type="regression",
            model_type="multifidelity_gp",
            input_type="mixed",
            cat_dims=[1],
            outcome_transform=False,
        )
    )

    assert normal.__name__ == "create_configured_fidelity_surrogate"
    assert mixed is normal


def test_build_model_normal_uses_fidelity_feature_shorthand():
    train_X, train_Y = _normal_data()
    config = ModelConfig(
        task_type="regression",
        model_type="multifidelity_gp",
        input_type="normal",
        outcome_transform=False,
        model_kwargs={
            "fidelity_features": [-1],
            "target_fidelities": {-1: 1.0},
        },
    )

    bundle = build_model(train_X, train_Y, config)

    assert isinstance(bundle.model, GaussianMultiFidelityGP)
    assert bundle.model.fidelity_features == (1,)
    assert bundle.model.target_fidelities == {1: 1.0}
    assert bundle.model.fidelity_metadata["input_mode"] == "continuous"
    assert bundle.model_type == "multifidelity_gp"


def test_build_model_mixed_dispatches_with_cat_dims():
    train_X, train_Y = _mixed_data()
    config = ModelConfig(
        task_type="regression",
        model_type="multifidelity_gp",
        input_type="mixed",
        cat_dims=[1],
        outcome_transform=False,
        model_kwargs={"fidelity_features": [-1]},
    )

    bundle = build_model(train_X, train_Y, config)

    assert isinstance(bundle.model, GaussianMixedMultiFidelityGP)
    assert bundle.model.cat_dims == (1,)
    assert bundle.model.cont_dims == (0,)
    assert bundle.model.fidelity_features == (2,)
    assert bundle.input_type == "mixed"


def test_build_model_accepts_explicit_fidelity_spec():
    train_X, train_Y = _normal_data()
    config = ModelConfig(
        task_type="regression",
        model_type="multifidelity_gp",
        input_type="normal",
        outcome_transform=False,
        model_kwargs={
            "fidelity_spec": FidelitySpec(
                fidelity_features=(-1,),
                target_fidelities={-1: 1.0},
            )
        },
    )

    bundle = build_model(train_X, train_Y, config)
    assert bundle.model.target_fidelities == {1: 1.0}


def test_rejects_missing_or_conflicting_fidelity_config():
    train_X, train_Y = _normal_data()

    with pytest.raises(ValueError, match="requires model_kwargs"):
        build_model(
            train_X,
            train_Y,
            ModelConfig(
                task_type="regression",
                model_type="multifidelity_gp",
                input_type="normal",
                outcome_transform=False,
            ),
        )

    with pytest.raises(ValueError, match="either fidelity_spec"):
        build_model(
            train_X,
            train_Y,
            ModelConfig(
                task_type="regression",
                model_type="multifidelity_gp",
                input_type="normal",
                outcome_transform=False,
                model_kwargs={
                    "fidelity_spec": FidelitySpec(fidelity_features=(-1,)),
                    "fidelity_features": [-1],
                },
            ),
        )


def test_fit_model_uses_exact_mll_protocol():
    train_X, train_Y = _normal_data()
    config = ModelConfig(
        task_type="regression",
        model_type="multifidelity_gp",
        input_type="normal",
        outcome_transform=False,
        model_kwargs={"fidelity_features": [-1]},
    )
    bundle = build_model(train_X, train_Y, config)

    fitted = fit_model(bundle, FitConfig(fit_func=_identity_fit))

    assert isinstance(fitted.mll, ExactMarginalLogLikelihood)
    assert fitted.mll.model is fitted.model
    assert fitted.fit_result is fitted.mll


def test_bayesian_optimizer_fit_and_predict_normal_multifidelity_gp():
    train_X, train_Y = _normal_data()
    optimizer = BayesianOptimizer(
        ModelConfig(
            task_type="regression",
            model_type="multifidelity_gp",
            input_type="normal",
            outcome_transform=False,
            model_kwargs={"fidelity_features": [-1]},
        ),
        fit_config=FitConfig(fit_func=_identity_fit),
    )

    optimizer.fit(train_X, train_Y)
    mean, variance = optimizer.predict(
        torch.tensor([[0.25, 1.0], [0.75, 0.5]], dtype=torch.double),
        return_type="mean_variance",
    )

    assert isinstance(optimizer.model, GaussianMultiFidelityGP)
    assert mean.shape == torch.Size([2, 1])
    assert variance.shape == torch.Size([2, 1])
    assert torch.isfinite(mean).all()
    assert torch.isfinite(variance).all()
    assert bool((variance >= 0).all())


def test_bayesian_optimizer_fit_and_predict_mixed_multifidelity_gp():
    train_X, train_Y = _mixed_data()
    optimizer = BayesianOptimizer(
        ModelConfig(
            task_type="regression",
            model_type="multifidelity_gp",
            input_type="mixed",
            cat_dims=[1],
            outcome_transform=False,
            model_kwargs={"fidelity_features": [-1]},
        ),
        fit_config=FitConfig(fit_func=_identity_fit),
    )

    optimizer.fit(train_X, train_Y)
    mean = optimizer.predict(
        torch.tensor([[0.25, 0.0, 1.0], [0.75, 1.0, 0.5]], dtype=torch.double),
        return_type="mean",
    )

    assert isinstance(optimizer.model, GaussianMixedMultiFidelityGP)
    assert mean.shape == torch.Size([2, 1])
    assert torch.isfinite(mean).all()


def test_legacy_multifidelity_registry_entry_is_unchanged():
    resolved = resolve_model_cls(
        ModelConfig(
            task_type="regression",
            model_type="multifidelity",
            input_type="normal",
            outcome_transform=False,
        )
    )
    assert resolved.__name__ == "WideMultiFidelityGP"
