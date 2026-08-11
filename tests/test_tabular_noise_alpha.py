from __future__ import annotations

import pytest
import torch

from bochan.api import ModelConfig, MultiOutputConfig, OutputConfig
from bochan.models.regression.gaussian.likelihood import build_single_task_likelihood
from bochan.tabular import TabularBayesianOptimizer
from bochan.tabular import optimizer_api


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_Y = train_X.square()
    return train_X, train_Y


def _noise_lower_bound(likelihood) -> float:
    constraint = likelihood.noise_covar.raw_noise_constraint
    return float(constraint.lower_bound.detach().cpu())


def test_likelihood_builder_accepts_alpha_and_uses_valid_initial_value() -> None:
    train_X, train_Y = _training_data()

    likelihood = build_single_task_likelihood(
        train_X=train_X,
        train_Y=train_Y,
        alpha=0.1,
    )

    assert _noise_lower_bound(likelihood) == pytest.approx(0.1)
    assert float(likelihood.noise.detach().cpu()) > 0.1


def test_noise_prior_constraint_stays_positive_for_gradient_optimizers() -> None:
    train_X, train_Y = _training_data()
    alpha = 1e-6
    likelihood = build_single_task_likelihood(
        train_X=train_X,
        train_Y=train_Y,
        alpha=alpha,
    ).to(train_X)

    constraint = likelihood.noise_covar.raw_noise_constraint
    assert constraint._transform is not None

    with torch.no_grad():
        likelihood.noise_covar.raw_noise.fill_(-20.0)

    noise = likelihood.noise
    prior_log_prob = likelihood.noise_covar.noise_prior.log_prob(noise)
    assert torch.all(noise > alpha)
    assert torch.isfinite(prior_log_prob).all()


@pytest.mark.parametrize("alpha", [0.0, -1e-4, float("inf"), float("nan")])
def test_likelihood_builder_rejects_invalid_alpha(alpha: float) -> None:
    train_X, train_Y = _training_data()

    with pytest.raises(ValueError, match="finite positive"):
        build_single_task_likelihood(
            train_X=train_X,
            train_Y=train_Y,
            alpha=alpha,
        )


def test_tabular_constructor_accepts_alpha() -> None:
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        model_type="base",
        alpha=1e-6,
        input_cols=["x"],
        target_cols="y",
    )

    assert optimizer.alpha == pytest.approx(1e-6)


def test_tabular_alpha_builds_likelihood_for_supported_regression() -> None:
    train_X, train_Y = _training_data()
    config = ModelConfig(
        task_type="regression",
        model_type="base",
        outcome_transform=False,
    )

    resolved = optimizer_api._apply_alpha_to_model_config(
        config,
        train_X=train_X,
        train_Y=train_Y,
        explicit_alpha=1e-6,
    )

    likelihood = resolved.model_kwargs["likelihood"]
    assert _noise_lower_bound(likelihood) == pytest.approx(1e-6)
    assert next(likelihood.parameters()).dtype == torch.double


def test_tabular_alpha_applies_only_to_hybrid_regression_outputs() -> None:
    train_X, train_Y = _training_data()
    binary_Y = (train_Y > train_Y.mean()).to(train_Y)
    combined_Y = torch.cat([train_Y, binary_Y], dim=-1)
    private_key = optimizer_api._TABULAR_NOISE_ALPHA_KEY
    config = ModelConfig(
        task_type="hybrid",
        model_type="base",
        model_kwargs={},
        multi_output_config=MultiOutputConfig(
            output_configs=[
                OutputConfig(
                    task_type="regression",
                    model_type="base",
                    model_kwargs={private_key: 1e-5},
                ),
                OutputConfig(
                    task_type="binary",
                    model_type="base",
                    model_kwargs={private_key: 1e-5},
                ),
            ],
            use_hybrid=True,
        ),
    )

    resolved = optimizer_api._apply_alpha_to_model_config(
        config,
        train_X=train_X,
        train_Y=combined_Y,
        explicit_alpha=None,
    )

    multi_output = resolved.multi_output_config
    assert multi_output is not None
    assert multi_output.output_configs is not None
    regression_config, binary_config = multi_output.output_configs
    assert _noise_lower_bound(
        regression_config.model_kwargs["likelihood"]
    ) == pytest.approx(1e-5)
    assert private_key not in regression_config.model_kwargs
    assert private_key not in binary_config.model_kwargs
    assert "likelihood" not in binary_config.model_kwargs


def test_tabular_alpha_rejects_unsupported_gaussian_model() -> None:
    train_X, train_Y = _training_data()
    config = ModelConfig(
        task_type="regression",
        model_type="saas",
        outcome_transform=False,
    )

    with pytest.raises(ValueError, match="recommended Gaussian regression models"):
        optimizer_api._apply_alpha_to_model_config(
            config,
            train_X=train_X,
            train_Y=train_Y,
            explicit_alpha=1e-6,
        )


def test_fastapi_tabular_schema_forwards_alpha_to_model_kwargs() -> None:
    pytest.importorskip("fastapi")
    from pydantic import ValidationError

    from bochan.serving.fastapi.schemas.tabular import TabularFitModelRequest

    request = TabularFitModelRequest.model_validate(
        {
            "data": [{"x": 0.0, "y": 1.0}, {"x": 1.0, "y": 2.0}],
            "model_config": {
                "task_type": "regression",
                "model_type": "base",
            },
            "alpha": 1e-6,
            "input_cols": ["x"],
            "target_cols": "y",
        }
    )

    assert request.alpha == pytest.approx(1e-6)
    assert request.bo_model_config.model_kwargs[
        optimizer_api._TABULAR_NOISE_ALPHA_KEY
    ] == pytest.approx(1e-6)

    with pytest.raises(ValidationError):
        TabularFitModelRequest.model_validate(
            {
                "data": [{"x": 0.0, "y": 1.0}],
                "model_config": {
                    "task_type": "regression",
                    "model_type": "base",
                },
                "alpha": 0.0,
                "input_cols": ["x"],
                "target_cols": "y",
            }
        )
