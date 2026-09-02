from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from bochan.api import ModelConfig
from bochan.api.observation.service import build_objective_bundle
from bochan.fit import fit_deepkernel_mll
from bochan.models.regression.gaussian.deep.deepkernel_configurable import (
    DeepKernelGaussianGPModel,
)
from bochan.models.regression.gaussian.deep.multitask_fixed_noise import (
    MultitaskFixedNoiseGaussianLikelihood,
)


def _partial_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [[0.0, 0.0], [0.3, 0.2], [0.6, 0.8], [1.0, 1.0]],
        dtype=torch.double,
    )
    Y = torch.tensor(
        [[0.0, 1.0], [0.4, float("nan")], [float("nan"), 1.8], [1.1, 2.2]],
        dtype=torch.double,
    )
    Yvar = torch.tensor(
        [[0.01, 0.02], [0.03, float("nan")], [float("nan"), 0.06], [0.07, 0.08]],
        dtype=torch.double,
    )
    return X, Y, Yvar


def test_multitask_fixed_noise_missing_mode_keeps_public_nan_contract() -> None:
    _, _, Yvar = _partial_data()
    with pytest.raises(ValueError, match="finite variances"):
        MultitaskFixedNoiseGaussianLikelihood(Yvar)

    likelihood = MultitaskFixedNoiseGaussianLikelihood(
        Yvar,
        allow_missing=True,
    )
    torch.testing.assert_close(
        likelihood.task_noise,
        Yvar,
        equal_nan=True,
    )
    torch.testing.assert_close(
        likelihood.missing_noise_mask,
        torch.isnan(Yvar),
    )
    assert torch.isfinite(likelihood.noise_covar.noise).all()


def test_partial_correlated_deepkernel_known_noise_fits_and_predicts() -> None:
    X, Y, Yvar = _partial_data()
    model = DeepKernelGaussianGPModel(
        X,
        Y,
        Yvar,
        feature_extractor=nn.Identity(),
        latent_dim=2,
        input_transform=None,
        outcome_transform=None,
    ).double()

    assert model._uses_observation_nan_mask is True
    assert isinstance(model.likelihood, MultitaskFixedNoiseGaussianLikelihood)
    torch.testing.assert_close(model.likelihood.task_noise, Yvar, equal_nan=True)

    mll = model.make_mll()
    model.train()
    output = model.deepkernel(model.transform_inputs(X))
    value = mll(output, Y)
    assert torch.isfinite(value)
    (-value).backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    fit_deepkernel_mll(model.make_mll(), num_epochs=1, lr=1e-3)
    posterior = model.posterior(X[:2], observation_noise=False)
    assert posterior.mean.shape == torch.Size([2, 2])
    assert posterior.variance.shape == torch.Size([2, 2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert bool((posterior.variance >= 0).all())


def test_default_standardize_preserves_partial_variance_alignment() -> None:
    X, Y, Yvar = _partial_data()
    model = DeepKernelGaussianGPModel(
        X,
        Y,
        Yvar,
        feature_extractor=nn.Identity(),
        latent_dim=2,
        input_transform=None,
    ).double()

    assert torch.equal(torch.isnan(model.train_Y), torch.isnan(Y))
    assert model.train_Yvar is not None
    assert torch.equal(torch.isnan(model.train_Yvar), torch.isnan(Yvar))
    observed = ~torch.isnan(model.train_Y)
    assert torch.isfinite(model.train_Y[observed]).all()
    assert torch.isfinite(model.train_Yvar[observed]).all()
    assert bool((model.train_Yvar[observed] > 0).all())


def test_partial_correlated_contract_rejects_misaligned_or_empty_outputs() -> None:
    X, Y, Yvar = _partial_data()
    bad_yvar = Yvar.clone()
    bad_yvar[1, 1] = 0.04
    with pytest.raises(ValueError, match="NaN positions must exactly match"):
        DeepKernelGaussianGPModel(
            X,
            Y,
            bad_yvar,
            feature_extractor=nn.Identity(),
            latent_dim=2,
            input_transform=None,
            outcome_transform=None,
        )

    empty_output = Y.clone()
    empty_output[:, 1] = float("nan")
    empty_yvar = Yvar.clone()
    empty_yvar[:, 1] = float("nan")
    with pytest.raises(ValueError, match="at least one observed target"):
        DeepKernelGaussianGPModel(
            X,
            empty_output,
            empty_yvar,
            feature_extractor=nn.Identity(),
            latent_dim=2,
            input_transform=None,
            outcome_transform=None,
        )


def test_observation_builder_routes_marker_model_as_correlated_wide() -> None:
    X, Y, Yvar = _partial_data()
    config = ModelConfig(
        task_type="multi_objective",
        model_type="phase6_correlated_test",
        model_cls=DeepKernelGaussianGPModel,
        outcome_transform=False,
        model_kwargs={
            "feature_extractor": nn.Identity(),
            "latent_dim": 2,
            "input_transform": None,
            "outcome_transform": None,
        },
    )
    bundle = build_objective_bundle(
        train_X=X,
        train_Y=Y,
        train_Yvar=Yvar,
        config=config,
    )

    assert isinstance(bundle.model, DeepKernelGaussianGPModel)
    assert bundle.model.num_outputs == 2
    assert bundle.model._uses_observation_nan_mask is True
    torch.testing.assert_close(bundle.model.train_Yvar, Yvar, equal_nan=True)


@pytest.mark.parametrize(
    "filename",
    [
        "mace_multitask.py",
        "chgnet_multitask.py",
        "m3gnet_multitask.py",
        "alignn_multitask.py",
        "crabnet_multitask.py",
    ],
)
def test_material_correlated_families_inherit_shared_partial_contract(
    filename: str,
) -> None:
    source = (
        Path("src/bochan/models/regression/gaussian/deep") / filename
    ).read_text(encoding="utf-8")
    assert "DeepKernelGaussianGPModel" in source
    assert "train_Yvar=train_Yvar" in source
    assert DeepKernelGaussianGPModel._supports_partial_multitask_targets is True
