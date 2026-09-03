"""Capability contracts for pretrained material backends.

The contracts in this module describe what a pretrained material family can do
without importing the optional backend that implements it.  They deliberately
separate representation extraction from direct property prediction so GP/DKL
and future residual-GP paths can reason about capabilities before loading a
large third-party model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MaterialDomain = Literal["composition", "structure"]
PretrainedLoadingMode = Literal["checkpoint", "model_name", "injected"]


@dataclass(frozen=True)
class PretrainedMaterialCapabilities:
    """Declare backend-independent capabilities of one pretrained model family.

    Args:
        representation: Whether the backend can return a fixed-width material
            representation suitable for a GP/DKL feature extractor.
        direct_prediction: Whether the pretrained backend can return one or
            more physical/property predictions without fitting a Bochan GP.
        loading_modes: Supported ways to obtain the pretrained backend.
        device_aware: Whether the backend can be moved between torch devices
            through its supported adapter contract.
        dtype_aware: Whether floating-point dtype can be propagated safely
            through the supported adapter contract.
        fine_tuning: Whether representation parameters may participate in DKL
            training through Bochan's encoder training policy.
        residual_gp: Whether direct pretrained predictions are intended to be
            usable as the baseline of a future residual GP.
    """

    representation: bool
    direct_prediction: bool = False
    loading_modes: frozenset[PretrainedLoadingMode] = frozenset({"injected"})
    device_aware: bool = True
    dtype_aware: bool = True
    fine_tuning: bool = False
    residual_gp: bool = False

    def __post_init__(self) -> None:
        valid_loading_modes = {"checkpoint", "model_name", "injected"}
        if not self.loading_modes:
            raise ValueError("loading_modes must contain at least one loading mode.")
        if not self.loading_modes.issubset(valid_loading_modes):
            invalid = sorted(set(self.loading_modes) - valid_loading_modes)
            raise ValueError(f"Unsupported pretrained loading modes: {invalid!r}.")
        if self.fine_tuning and not self.representation:
            raise ValueError("fine_tuning requires representation=True.")
        if self.residual_gp and not self.direct_prediction:
            raise ValueError("residual_gp requires direct_prediction=True.")

    def supports_loading(self, mode: PretrainedLoadingMode) -> bool:
        """Return whether the family supports one loading route."""

        return mode in self.loading_modes

    def require_representation(self) -> None:
        """Raise when the family cannot supply GP/DKL representation features."""

        if not self.representation:
            raise ValueError("This pretrained material backend does not expose representations.")

    def require_direct_prediction(self) -> None:
        """Raise when the family cannot make direct pretrained predictions."""

        if not self.direct_prediction:
            raise ValueError("This pretrained material backend does not expose direct predictions.")

    def require_residual_gp(self) -> None:
        """Raise when the family is not declared suitable for residual-GP use."""

        if not self.residual_gp:
            raise ValueError("This pretrained material backend is not residual-GP capable.")


@dataclass(frozen=True)
class PretrainedMaterialSpec:
    """Identify one pretrained family without importing its optional backend.

    ``family`` is an internal stable family identifier such as ``"mace"`` or
    ``"crabnet"``.  ``default_model_name`` is metadata only; loading remains the
    responsibility of the family adapter so this neutral module stays free of
    optional dependencies.
    """

    family: str
    domain: MaterialDomain
    capabilities: PretrainedMaterialCapabilities
    default_model_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family.strip():
            raise ValueError("family must be a non-empty string.")
        if self.domain not in {"composition", "structure"}:
            raise ValueError("domain must be 'composition' or 'structure'.")
        if not isinstance(self.capabilities, PretrainedMaterialCapabilities):
            raise TypeError("capabilities must be PretrainedMaterialCapabilities.")
        if self.default_model_name is not None and (
            not isinstance(self.default_model_name, str) or not self.default_model_name.strip()
        ):
            raise ValueError("default_model_name must be a non-empty string when provided.")
        if self.default_model_name is not None and not self.capabilities.supports_loading(
            "model_name"
        ):
            raise ValueError(
                "default_model_name requires 'model_name' in capabilities.loading_modes."
            )

    @property
    def supports_gp(self) -> bool:
        """Return whether a frozen-representation GP can be constructed."""

        return self.capabilities.representation

    @property
    def supports_dkl(self) -> bool:
        """Return whether the representation is declared trainable for DKL."""

        return self.capabilities.representation and self.capabilities.fine_tuning

    @property
    def supports_residual_gp(self) -> bool:
        """Return whether direct predictions may seed a residual GP."""

        return self.capabilities.residual_gp


def resolve_pretrained_loading_mode(
    spec: PretrainedMaterialSpec,
    *,
    checkpoint: object | None = None,
    model_name: str | None = None,
    injected_model: object | None = None,
) -> PretrainedLoadingMode:
    """Resolve one unambiguous loading route and validate family capabilities.

    This helper performs only routing validation.  It never imports or loads a
    third-party model.
    """

    if not isinstance(spec, PretrainedMaterialSpec):
        raise TypeError("spec must be a PretrainedMaterialSpec.")

    requested: list[PretrainedLoadingMode] = []
    if checkpoint is not None:
        requested.append("checkpoint")
    if model_name is not None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string when provided.")
        requested.append("model_name")
    if injected_model is not None:
        requested.append("injected")

    if not requested and spec.default_model_name is not None:
        requested.append("model_name")
    if len(requested) != 1:
        raise ValueError(
            "Exactly one pretrained loading route must be selected: checkpoint, "
            "model_name, or injected_model."
        )

    mode = requested[0]
    if not spec.capabilities.supports_loading(mode):
        raise ValueError(
            f"Pretrained family {spec.family!r} does not support loading mode {mode!r}."
        )
    return mode


__all__ = [
    "MaterialDomain",
    "PretrainedLoadingMode",
    "PretrainedMaterialCapabilities",
    "PretrainedMaterialSpec",
    "resolve_pretrained_loading_mode",
]
