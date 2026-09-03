from __future__ import annotations

import pytest

from bochan.models.regression.gaussian.materials import (
    PretrainedMaterialCapabilities,
    PretrainedMaterialSpec,
    resolve_pretrained_loading_mode,
)


def test_pretrained_capabilities_expose_gp_dkl_and_residual_semantics() -> None:
    capabilities = PretrainedMaterialCapabilities(
        representation=True,
        direct_prediction=True,
        loading_modes=frozenset({"model_name", "injected"}),
        fine_tuning=True,
        residual_gp=True,
    )
    spec = PretrainedMaterialSpec(
        family="example_structure_model",
        domain="structure",
        capabilities=capabilities,
        default_model_name="default-model",
    )

    assert spec.supports_gp is True
    assert spec.supports_dkl is True
    assert spec.supports_residual_gp is True
    assert capabilities.supports_loading("model_name") is True
    assert resolve_pretrained_loading_mode(spec) == "model_name"
    assert resolve_pretrained_loading_mode(spec, injected_model=object()) == "injected"


def test_residual_gp_requires_direct_prediction() -> None:
    with pytest.raises(ValueError, match="direct_prediction"):
        PretrainedMaterialCapabilities(
            representation=True,
            direct_prediction=False,
            residual_gp=True,
        )


def test_fine_tuning_requires_representation() -> None:
    with pytest.raises(ValueError, match="representation"):
        PretrainedMaterialCapabilities(
            representation=False,
            fine_tuning=True,
        )


def test_default_model_name_requires_model_name_loading() -> None:
    capabilities = PretrainedMaterialCapabilities(
        representation=True,
        loading_modes=frozenset({"injected"}),
    )
    with pytest.raises(ValueError, match="model_name"):
        PretrainedMaterialSpec(
            family="example",
            domain="composition",
            capabilities=capabilities,
            default_model_name="not-supported",
        )


def test_loading_route_must_be_unambiguous_and_supported() -> None:
    spec = PretrainedMaterialSpec(
        family="example",
        domain="structure",
        capabilities=PretrainedMaterialCapabilities(
            representation=True,
            loading_modes=frozenset({"checkpoint", "injected"}),
        ),
    )

    with pytest.raises(ValueError, match="Exactly one"):
        resolve_pretrained_loading_mode(spec)
    with pytest.raises(ValueError, match="Exactly one"):
        resolve_pretrained_loading_mode(
            spec,
            checkpoint=object(),
            injected_model=object(),
        )
    with pytest.raises(ValueError, match="does not support"):
        resolve_pretrained_loading_mode(spec, model_name="unsupported")


def test_capability_require_helpers_fail_early() -> None:
    capabilities = PretrainedMaterialCapabilities(
        representation=False,
        direct_prediction=False,
        loading_modes=frozenset({"injected"}),
    )

    with pytest.raises(ValueError, match="representations"):
        capabilities.require_representation()
    with pytest.raises(ValueError, match="direct predictions"):
        capabilities.require_direct_prediction()
    with pytest.raises(ValueError, match="residual-GP"):
        capabilities.require_residual_gp()
