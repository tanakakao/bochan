from __future__ import annotations

import pytest
import torch

from bochan.api import ModelConfig
from bochan.api.modeling.build import build_model
from bochan.models.multifidelity import (
    FidelitySpec,
    GaussianCorrelatedMultiFidelityGP,
    create_configured_fidelity_surrogate,
)


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [
            [0.05, 0.25],
            [0.20, 0.50],
            [0.40, 0.75],
            [0.65, 1.00],
            [0.85, 0.50],
            [0.95, 1.00],
        ],
        dtype=torch.double,
    )
    y1 = torch.sin(2.0 * X[:, 0]) + 0.4 * X[:, 1]
    y2 = 1.5 * y1 + 0.1 * torch.cos(3.0 * X[:, 0])
    return X, torch.stack([y1, y2], dim=-1)


def test_correlated_multifidelity_model_exposes_shared_contract() -> None:
    X, Y = _data()
    model = GaussianCorrelatedMultiFidelityGP(
        X,
        Y,
        fidelity_spec=FidelitySpec(
            fidelity_features=(-1,),
            target_fidelities={-1: 1.0},
        ),
    )

    assert model.num_outputs == 2
    assert model.fidelity_features == (1,)
    assert model.target_fidelities == {1: 1.0}
    assert model.multi_output_fidelity == "correlated"
    assert model.is_multifidelity_model is True
    assert model.fidelity_metadata()["num_fidelity_outputs"] == 2

    posterior = model.posterior(X[:2])
    assert posterior.mean.shape == (2, 2)


def test_correlated_model_has_nontrivial_task_covariance() -> None:
    X, Y = _data()
    model = GaussianCorrelatedMultiFidelityGP(
        X,
        Y,
        fidelity_spec=FidelitySpec(fidelity_features=(1,)),
        rank=2,
    )
    task_kernel = model.covar_module.task_covar_module
    task_covar = task_kernel.covar_matrix.to_dense()

    assert task_covar.shape == (2, 2)
    assert torch.isfinite(task_covar).all()
    assert task_covar[0, 1].abs() > 0


def test_standard_multifidelity_model_type_can_select_correlated_outputs() -> None:
    X, Y = _data()
    model = create_configured_fidelity_surrogate(
        X,
        Y,
        fidelity_features=(-1,),
        target_fidelities={-1: 1.0},
        correlated_outputs=True,
    )

    assert isinstance(model, GaussianCorrelatedMultiFidelityGP)
    assert model.num_outputs == 2


def test_model_config_alias_builds_correlated_multifidelity_gp() -> None:
    X, Y = _data()
    bundle = build_model(
        X,
        Y,
        ModelConfig(
            task_type="regression",
            model_type="correlated_multifidelity_gp",
            input_type="normal",
            model_kwargs={
                "fidelity_features": [-1],
                "target_fidelities": {-1: 1.0},
            },
        ),
    )

    assert isinstance(bundle.model, GaussianCorrelatedMultiFidelityGP)
    assert bundle.model.num_outputs == 2
    assert bundle.model.multi_output_fidelity == "correlated"


def test_correlated_multifidelity_rejects_partial_outputs_known_noise_and_mixed() -> None:
    X, Y = _data()
    Y_missing = Y.clone()
    Y_missing[0, 1] = float("nan")

    with pytest.raises(ValueError, match="fully observed"):
        GaussianCorrelatedMultiFidelityGP(
            X,
            Y_missing,
            fidelity_spec=FidelitySpec(fidelity_features=(1,)),
        )

    with pytest.raises(NotImplementedError, match="train_Yvar"):
        GaussianCorrelatedMultiFidelityGP(
            X,
            Y,
            train_Yvar=torch.full_like(Y, 1e-4),
            fidelity_spec=FidelitySpec(fidelity_features=(1,)),
        )

    with pytest.raises(NotImplementedError, match="continuous inputs only"):
        create_configured_fidelity_surrogate(
            X,
            Y,
            cat_dims=[0],
            fidelity_features=(1,),
            correlated_outputs=True,
        )
