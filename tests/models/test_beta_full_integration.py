"""Focused Beta distribution, registry, capability, and heteroscedastic tests."""

import pytest
import torch

from bochan.api.model_capabilities import BETA_MODEL_TYPES, model_capability
from bochan.api.registry.model import DEFAULT_MODEL_REGISTRY
from bochan.models.regression.beta._components import BetaLogLikelihood, prepare_beta_targets
from bochan.models.regression.beta.robust import HeteroscedasticBetaGPModel


def test_beta_target_clipping_warns_and_rejects_invalid_values() -> None:
    """Clipping accepts only the closed unit interval and reports boundaries."""
    ref = torch.zeros(3, dtype=torch.double)
    with pytest.warns(UserWarning, match="exact 0 or 1"):
        clipped = prepare_beta_targets(torch.tensor([0.0, 0.5, 1.0]), ref, clip=True)
    assert torch.all((clipped > 0) & (clipped < 1))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        prepare_beta_targets(torch.tensor([0.2, 100.0]), ref, clip=True)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"link": "logit"}, "link"),
        ({"eps": 0.5}, "eps"),
        ({"init_concentration": 0.0}, "init_concentration"),
        ({"min_concentration": float("inf")}, "min_concentration"),
    ],
)
def test_beta_likelihood_validates_distribution_parameters(kwargs: dict, message: str) -> None:
    """Invalid link, epsilon, and concentration settings fail at construction."""
    with pytest.raises(ValueError, match=message):
        BetaLogLikelihood(**kwargs)


def test_beta_registry_capabilities_and_web_are_consistent() -> None:
    """All eight normal/mixed variants are catalogued without native multitask."""
    raw = DEFAULT_MODEL_REGISTRY.raw()
    assert "beta_multitask" not in raw["normal"]["regression"]
    assert "beta_multitask" not in raw["normal"]["multi_objective"]
    for model_type in BETA_MODEL_TYPES:
        assert model_type in raw["normal"]["regression"]
        assert model_type in raw["mixed"]["regression"]
        capability = model_capability(model_type)
        assert capability is not None
        assert capability.supports_independent_multi_output
        assert not capability.supports_native_multi_output
    assert model_capability("beta_hetero").supports_noise_importance


def test_heteroscedastic_beta_registers_auxiliary_noise_model() -> None:
    """Parent initialization precedes auxiliary module registration."""
    train_x = torch.rand(5, 2, dtype=torch.double)
    train_y = torch.full((5, 1), 0.5, dtype=torch.double)
    model = HeteroscedasticBetaGPModel(
        train_x,
        train_y,
        train_Yvar=torch.full_like(train_y, 0.01),
        aux_num_epochs=0,
    )
    assert "noise_model" in dict(model.named_modules())
    assert any(key.startswith("noise_model.") for key in model.state_dict())
    model.to(dtype=torch.float)
    assert next(model.noise_model.parameters()).dtype == torch.float
