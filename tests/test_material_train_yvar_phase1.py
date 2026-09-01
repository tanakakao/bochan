from __future__ import annotations

import pytest
import torch
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood

from bochan.models.regression.gaussian.deep import (
    ALIGNNDKLModel,
    ALIGNNGPModel,
    CHGNetDKLModel,
    CHGNetGPModel,
    CrabNetDKLModel,
    CrabNetGPModel,
    DeepKernelGaussianGPModel,
    DeepKernelGaussianMixedGPModel,
    M3GNetDKLModel,
    M3GNetGPModel,
    MACEDKLModel,
    MACEGPModel,
    RoostDKLModel,
    RoostGPModel,
)


def _known_variance(train_Y: torch.Tensor) -> torch.Tensor:
    return torch.linspace(
        0.01,
        0.04,
        train_Y.shape[0],
        dtype=train_Y.dtype,
        device=train_Y.device,
    ).unsqueeze(-1)


def _assert_fixed_noise(model, expected_yvar: torch.Tensor) -> None:
    assert isinstance(model.likelihood, FixedNoiseGaussianLikelihood)
    assert model.train_Yvar is not None
    torch.testing.assert_close(model.train_Yvar, expected_yvar)
    torch.testing.assert_close(model.likelihood.noise, expected_yvar.squeeze(-1))
    posterior = model.posterior(model.train_X[:2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_shared_deepkernel_selects_fixed_noise_only_when_yvar_is_supplied() -> None:
    train_X = torch.rand(8, 2, dtype=torch.double)
    train_Y = train_X[:, :1] - 0.3 * train_X[:, 1:2]
    train_Yvar = _known_variance(train_Y)

    fixed = DeepKernelGaussianGPModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        outcome_transform=None,
    )
    learned = DeepKernelGaussianGPModel(
        train_X,
        train_Y,
        outcome_transform=None,
    )

    _assert_fixed_noise(fixed, train_Yvar)
    assert isinstance(learned.likelihood, GaussianLikelihood)
    assert not isinstance(learned.likelihood, FixedNoiseGaussianLikelihood)


def test_shared_mixed_deepkernel_selects_fixed_noise() -> None:
    continuous = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    categorical = torch.tensor(
        [0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.double
    ).unsqueeze(-1)
    train_X = torch.cat((continuous, categorical), dim=-1)
    train_Y = continuous + 0.2 * categorical
    train_Yvar = _known_variance(train_Y)
    model = DeepKernelGaussianMixedGPModel(
        train_X,
        train_Y,
        cat_dims=[1],
        train_Yvar=train_Yvar,
        outcome_transform=None,
    )
    _assert_fixed_noise(model, train_Yvar)


def test_mace_gp_and_dkl_accept_known_observation_variance() -> None:
    pytest.importorskip('mace')
    from tests.test_mace_gp import _data, _structures, _wrapped_encoder

    train_X, train_Y = _data(with_process=True)
    train_Yvar = _known_variance(train_Y)
    encoder, _ = _wrapped_encoder()
    gp = MACEGPModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        structures=_structures(),
        encoder=encoder,
        latent_dim=3,
        outcome_transform=None,
    )
    encoder, _ = _wrapped_encoder()
    dkl = MACEDKLModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        structures=_structures(),
        encoder=encoder,
        latent_dim=3,
        trainable_encoder_layers=1,
        outcome_transform=None,
    )
    _assert_fixed_noise(gp, train_Yvar)
    _assert_fixed_noise(dkl, train_Yvar)


def test_chgnet_gp_and_dkl_accept_known_observation_variance() -> None:
    pytest.importorskip('pymatgen')
    from tests.test_chgnet_gp import FakeCHGNet, _data, _structures

    train_X, train_Y = _data(with_process=True)
    train_Yvar = _known_variance(train_Y)
    gp = CHGNetGPModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        structures=_structures(),
        encoder=FakeCHGNet(),
        latent_dim=3,
        outcome_transform=None,
    )
    dkl = CHGNetDKLModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        structures=_structures(),
        encoder=FakeCHGNet(),
        latent_dim=3,
        trainable_encoder_layers=1,
        outcome_transform=None,
    )
    _assert_fixed_noise(gp, train_Yvar)
    _assert_fixed_noise(dkl, train_Yvar)


def test_m3gnet_gp_and_dkl_accept_known_observation_variance() -> None:
    pytest.importorskip('pymatgen')
    from tests.test_m3gnet_gp import _data, _structures, _wrapped_encoder

    train_X, train_Y = _data(with_process=True)
    train_Yvar = _known_variance(train_Y)
    gp = M3GNetGPModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        structures=_structures(),
        encoder=_wrapped_encoder(),
        latent_dim=3,
        outcome_transform=None,
    )
    dkl = M3GNetDKLModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        structures=_structures(),
        encoder=_wrapped_encoder(),
        latent_dim=3,
        trainable_encoder_layers=1,
        outcome_transform=None,
    )
    _assert_fixed_noise(gp, train_Yvar)
    _assert_fixed_noise(dkl, train_Yvar)


def test_alignn_gp_and_dkl_accept_known_observation_variance() -> None:
    from tests.test_alignn_gp import FakeALIGNN, _data, _graphs

    train_X, train_Y = _data(with_process=True)
    train_Yvar = _known_variance(train_Y)
    gp = ALIGNNGPModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        structure_graphs=_graphs(),
        encoder=FakeALIGNN(),
        latent_dim=3,
        outcome_transform=None,
    )
    dkl = ALIGNNDKLModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        structure_graphs=_graphs(),
        encoder=FakeALIGNN(),
        latent_dim=3,
        trainable_encoder_layers=1,
        outcome_transform=None,
    )
    _assert_fixed_noise(gp, train_Yvar)
    _assert_fixed_noise(dkl, train_Yvar)


def test_crabnet_gp_and_dkl_accept_known_observation_variance() -> None:
    from tests.test_crabnet_gp import (
        FakeCrabNet,
        LayeredFakeCrabNet,
        _data,
        _element_ids,
    )

    train_X, train_Y = _data(with_process=True)
    train_Yvar = _known_variance(train_Y)
    gp = CrabNetGPModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        element_ids=_element_ids(),
        encoder=FakeCrabNet(),
        latent_dim=3,
        outcome_transform=None,
    )
    dkl = CrabNetDKLModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        element_ids=_element_ids(),
        encoder=LayeredFakeCrabNet(),
        latent_dim=3,
        trainable_encoder_layers=1,
        outcome_transform=None,
    )
    _assert_fixed_noise(gp, train_Yvar)
    _assert_fixed_noise(dkl, train_Yvar)


def test_roost_gp_and_dkl_accept_known_observation_variance() -> None:
    from tests.test_roost_gp import (
        FakeRoostBackbone,
        LayeredFakeRoostBackbone,
        _data,
        _element_ids,
    )

    train_X, train_Y = _data(with_process=True)
    train_Yvar = _known_variance(train_Y)
    gp = RoostGPModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        element_ids=_element_ids(),
        encoder=FakeRoostBackbone(),
        latent_dim=3,
        outcome_transform=None,
    )
    dkl = RoostDKLModel(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        element_ids=_element_ids(),
        encoder=LayeredFakeRoostBackbone(),
        latent_dim=3,
        encoder_training='partial',
        trainable_encoder_layers=1,
        outcome_transform=None,
    )
    _assert_fixed_noise(gp, train_Yvar)
    _assert_fixed_noise(dkl, train_Yvar)
