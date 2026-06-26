from __future__ import annotations

import pytest

from bochan.api import OptimizeConfig
from bochan.api.optimizer_api import resolve_optimizer_from_cat_dims


@pytest.mark.parametrize("method", ["ga", "pso", "sa", "cmaes"])
def test_evolutionary_shorthand_is_normalized(method: str) -> None:
    config = OptimizeConfig(optimizer=method)

    assert config.optimizer == "evo"
    assert config.evo_method == method
    assert config.optimizer_kwargs["method"] == method


def test_explicit_evo_method_is_forwarded() -> None:
    config = OptimizeConfig(optimizer="evo", evo_method="pso")

    assert config.optimizer == "evo"
    assert config.optimizer_kwargs["method"] == "pso"


def test_optimizer_kwargs_method_has_priority() -> None:
    config = OptimizeConfig(
        optimizer="evo",
        evo_method="ga",
        optimizer_kwargs={"method": "sa"},
    )

    assert config.evo_method == "sa"
    assert config.optimizer_kwargs["method"] == "sa"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"optimizer": "cmaes", "q": 3},
        {
            "optimizer": "evo",
            "q": 3,
            "optimizer_kwargs": {"method": "cmaes"},
        },
    ],
)
def test_cmaes_q_greater_than_one_enables_sequential(kwargs: dict) -> None:
    config = OptimizeConfig(**kwargs)

    assert config.optimizer == "evo"
    assert config.evo_method == "cmaes"
    assert config.sequential is True


def test_cmaes_q_one_preserves_sequential_false() -> None:
    config = OptimizeConfig(optimizer="cmaes", q=1)

    assert config.sequential is False


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [
        ("optimize_acqf_mixed", "optimize_acqf"),
        ("optimize_acqf_evo_mixed", "evo"),
        ("optimize_acqf_torch_mixed", "torch"),
        ("optimize_acqf_nsgaii", "nsgaii"),
        ("optimize_thompson_sampling_mixed", "thompson_sampling"),
    ],
)
def test_legacy_optimizer_names_are_normalized(legacy: str, canonical: str) -> None:
    config = OptimizeConfig(optimizer=legacy)

    assert config.optimizer == canonical


@pytest.mark.parametrize(
    ("optimizer", "mixed_optimizer"),
    [
        ("optimize_acqf", "optimize_acqf_mixed"),
        ("evo", "evo_mixed"),
        ("torch", "torch_mixed"),
        ("thompson_sampling", "thompson_sampling_mixed"),
    ],
)
def test_mixed_backend_is_selected_from_cat_dims(
    optimizer: str,
    mixed_optimizer: str,
) -> None:
    config = OptimizeConfig(optimizer=optimizer)

    resolved = resolve_optimizer_from_cat_dims(opt_config=config, cat_dims=[1])

    assert resolved.optimizer == mixed_optimizer
    assert config.optimizer == optimizer


def test_unknown_optimizer_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown optimizer"):
        OptimizeConfig(optimizer="unknown")


def test_unknown_evolutionary_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown evolutionary method"):
        OptimizeConfig(optimizer="evo", optimizer_kwargs={"method": "unknown"})
