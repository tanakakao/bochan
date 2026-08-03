from __future__ import annotations

import pytest
import torch

from bochan.api import ModelConfig
from bochan.api.factory import build_model, resolve_model_cls
from bochan.models.regression.non_gaussian.beta.base import WideBetaMultiTaskGPModel
from bochan.models.regression.non_gaussian.gamma.base import WideGammaMultiTaskGPModel
from bochan.models.regression.non_gaussian.negative_binomial.base import (
    WideNegativeBinomialMultiTaskGPModel,
)
from bochan.models.regression.non_gaussian.poisson.base import (
    WidePoissonMultiTaskGPModel,
)
from bochan.models.transforms.outcome import PositiveScaleOutcomeTransform
from bochan.serving.webapp import workflows_tabular


@pytest.mark.parametrize(
    ("web_model_type", "internal_model_type"),
    [
        ("multitask", "multitask"),
        ("beta_multitask", "beta_wide_multitask"),
        ("gamma_multitask", "gamma_wide_multitask"),
        ("poisson_multitask", "poisson_wide_multitask"),
        (
            "negative_binomial_multitask",
            "negative_binomial_wide_multitask",
        ),
    ],
)
def test_web_multitask_aliases_resolve_to_shared_design_models(
    web_model_type: str,
    internal_model_type: str,
) -> None:
    assert workflows_tabular._is_direct_multitask_model(web_model_type)
    assert (
        workflows_tabular._resolve_direct_multitask_model_type(web_model_type)
        == internal_model_type
    )


def test_non_multitask_model_is_not_routed_directly() -> None:
    assert not workflows_tabular._is_direct_multitask_model("gamma_base")
    assert workflows_tabular._resolve_direct_multitask_model_type("gamma_base") is None


def _config(model_type: str) -> ModelConfig:
    return ModelConfig(
        **workflows_tabular._direct_multitask_model_config_kwargs(
            model_type=model_type,
            input_transform_config=None,
            outcome_transform=True,
            model_kwargs={"rank": 1, "num_inducing_points": 4},
        )
    )


def test_gamma_web_alias_builds_wide_correlated_model() -> None:
    train_x = torch.linspace(0.1, 0.9, 4, dtype=torch.double).unsqueeze(-1)
    train_y = torch.cat((1.0 + train_x, 2.0 + train_x), dim=-1)
    config = _config("gamma_wide_multitask")
    bundle = build_model(train_x, train_y, config)

    assert config.task_type == "multi_objective"
    assert config.model_type == "gamma_wide_multitask"
    assert config.multi_output_config is None
    assert config.cat_dims is None
    assert isinstance(config.outcome_transform, PositiveScaleOutcomeTransform)
    assert config.pass_outcome_transform is True
    assert resolve_model_cls(config) is WideGammaMultiTaskGPModel
    assert isinstance(bundle.model, WideGammaMultiTaskGPModel)
    assert bundle.model.num_tasks == train_y.shape[-1]


@pytest.mark.parametrize(
    ("model_type", "model_cls", "train_y"),
    [
        (
            "beta_wide_multitask",
            WideBetaMultiTaskGPModel,
            torch.tensor(
                [[0.2, 0.7], [0.3, 0.6], [0.4, 0.5], [0.5, 0.4]],
                dtype=torch.double,
            ),
        ),
        (
            "poisson_wide_multitask",
            WidePoissonMultiTaskGPModel,
            torch.tensor(
                [[0.0, 1.0], [1.0, 2.0], [2.0, 1.0], [3.0, 2.0]],
                dtype=torch.double,
            ),
        ),
        (
            "negative_binomial_wide_multitask",
            WideNegativeBinomialMultiTaskGPModel,
            torch.tensor(
                [[0.0, 2.0], [1.0, 3.0], [2.0, 4.0], [3.0, 5.0]],
                dtype=torch.double,
            ),
        ),
    ],
)
def test_other_non_gaussian_web_aliases_build_wide_correlated_models(
    model_type: str,
    model_cls: type,
    train_y: torch.Tensor,
) -> None:
    train_x = torch.linspace(0.1, 0.9, 4, dtype=torch.double).unsqueeze(-1)
    config = _config(model_type)
    bundle = build_model(train_x, train_y, config)

    assert resolve_model_cls(config) is model_cls
    assert isinstance(bundle.model, model_cls)
    assert bundle.model.num_tasks == train_y.shape[-1]
