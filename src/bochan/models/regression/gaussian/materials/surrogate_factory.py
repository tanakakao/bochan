"""Unified factory for registered material Gaussian surrogate variants.

This module lifts the existing backend-specific GP / DKL / mixed / correlated
multi-output classes behind one registry-driven construction API. Concrete
classes remain in their historical modules so saved-model import paths and
class identity stay backward compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from botorch.models.model_list_gp_regression import ModelListGP
from torch import Tensor

from .common import MaterialModelVariant, get_material_family

MaterialInputMode = Literal["continuous", "mixed"]
MaterialOutputMode = Literal["scalar", "independent", "correlated"]
MaterialGaussianKind = Literal["gp", "dkl"]

SUPPORTED_MATERIAL_INPUT_MODES: tuple[MaterialInputMode, ...] = ("continuous", "mixed")
SUPPORTED_MATERIAL_OUTPUT_MODES: tuple[MaterialOutputMode, ...] = (
    "scalar",
    "independent",
    "correlated",
)
SUPPORTED_MATERIAL_GAUSSIAN_KINDS: tuple[MaterialGaussianKind, ...] = ("gp", "dkl")

_INPUT_ALIASES = {
    "continuous": "continuous",
    "numeric": "continuous",
    "standard": "continuous",
    "mixed": "mixed",
    "mixed-input": "mixed",
    "mixed_input": "mixed",
}
_OUTPUT_ALIASES = {
    "scalar": "scalar",
    "single": "scalar",
    "single-output": "scalar",
    "single_output": "scalar",
    "independent": "independent",
    "independent-output": "independent",
    "independent_output": "independent",
    "independent-multi-output": "independent",
    "independent_multi_output": "independent",
    "model-list": "independent",
    "model_list": "independent",
    "modellist": "independent",
    "correlated": "correlated",
    "multi-output": "correlated",
    "multi_output": "correlated",
    "multioutput": "correlated",
    "multitask": "correlated",
    "multi-task": "correlated",
}
_KIND_ALIASES = {
    "gp": "gp",
    "gaussian_process": "gp",
    "gaussian-process": "gp",
    "dkl": "dkl",
    "deep_kernel": "dkl",
    "deep-kernel": "dkl",
}


def _normalize(value: str, aliases: dict[str, str], name: str, supported: tuple[str, ...]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    resolved = aliases.get(value.strip().lower())
    if resolved is None:
        raise ValueError(
            f"Unsupported {name} {value!r}. Supported values: {', '.join(supported)}."
        )
    return resolved


def normalize_material_input_mode(mode: str) -> MaterialInputMode:
    """Normalize the material surrogate input-space mode."""

    return cast(
        MaterialInputMode,
        _normalize(mode, _INPUT_ALIASES, "material input mode", SUPPORTED_MATERIAL_INPUT_MODES),
    )


def normalize_material_output_mode(mode: str) -> MaterialOutputMode:
    """Normalize scalar, independent, or correlated wide-output semantics.

    ``independent`` constructs one scalar material surrogate per output and
    combines them with :class:`botorch.models.ModelListGP`. Historical backend
    classes use ``MultiTask`` for correlated wide outputs; ``multitask`` and
    ``multi-output`` therefore remain aliases for ``correlated``.
    """

    return cast(
        MaterialOutputMode,
        _normalize(mode, _OUTPUT_ALIASES, "material output mode", SUPPORTED_MATERIAL_OUTPUT_MODES),
    )


def normalize_material_gaussian_kind(kind: str) -> MaterialGaussianKind:
    """Normalize GP versus DKL construction semantics."""

    return cast(
        MaterialGaussianKind,
        _normalize(kind, _KIND_ALIASES, "material Gaussian kind", SUPPORTED_MATERIAL_GAUSSIAN_KINDS),
    )


def material_model_variant(
    *,
    kind: str = "gp",
    input_mode: str = "continuous",
    output_mode: str = "scalar",
) -> MaterialModelVariant:
    """Resolve surrogate settings to the historical backend registry variant.

    Independent multi-output uses the same scalar backend variant repeatedly,
    once per output. Correlated multi-output resolves to the historical
    ``multitask_*`` variants.
    """

    normalized_kind = normalize_material_gaussian_kind(kind)
    normalized_input = normalize_material_input_mode(input_mode)
    normalized_output = normalize_material_output_mode(output_mode)

    prefix = "mixed_" if normalized_input == "mixed" else ""
    output = "multitask_" if normalized_output == "correlated" else ""
    return cast(MaterialModelVariant, f"{prefix}{output}{normalized_kind}")


@dataclass(frozen=True, slots=True)
class RegisteredMaterialSurrogateSpec:
    """Serializable identity for one registry-backed material Gaussian model."""

    family: str
    kind: MaterialGaussianKind | str = "gp"
    input_mode: MaterialInputMode | str = "continuous"
    output_mode: MaterialOutputMode | str = "scalar"

    def __post_init__(self) -> None:
        registration = get_material_family(self.family)
        object.__setattr__(self, "family", registration.family)
        object.__setattr__(self, "kind", normalize_material_gaussian_kind(self.kind))
        object.__setattr__(self, "input_mode", normalize_material_input_mode(self.input_mode))
        object.__setattr__(self, "output_mode", normalize_material_output_mode(self.output_mode))
        if not registration.supports(self.variant):
            raise ValueError(
                f"Material family {registration.family!r} does not support "
                f"{self.variant!r}. Supported variants: {sorted(registration.variants)!r}."
            )

    @property
    def variant(self) -> MaterialModelVariant:
        """Return the backend registry variant selected by this spec."""

        return material_model_variant(
            kind=cast(str, self.kind),
            input_mode=cast(str, self.input_mode),
            output_mode=cast(str, self.output_mode),
        )

    @property
    def domain(self) -> str:
        """Return ``composition`` or ``structure`` for the selected family."""

        return get_material_family(self.family).domain

    def as_dict(self) -> dict[str, str]:
        return {
            "family": self.family,
            "domain": self.domain,
            "kind": cast(str, self.kind),
            "input_mode": cast(str, self.input_mode),
            "output_mode": cast(str, self.output_mode),
            "variant": self.variant,
        }


def material_surrogate_capabilities(family: str) -> dict[str, Any]:
    """Return normalized GP/DKL input/output capabilities for one family."""

    registration = get_material_family(family)
    configurations: list[dict[str, str]] = []
    for kind in SUPPORTED_MATERIAL_GAUSSIAN_KINDS:
        for input_mode in SUPPORTED_MATERIAL_INPUT_MODES:
            for output_mode in SUPPORTED_MATERIAL_OUTPUT_MODES:
                variant = material_model_variant(
                    kind=kind,
                    input_mode=input_mode,
                    output_mode=output_mode,
                )
                if registration.supports(variant):
                    configurations.append(
                        {
                            "kind": kind,
                            "input_mode": input_mode,
                            "output_mode": output_mode,
                            "variant": variant,
                        }
                    )
    return {
        "family": registration.family,
        "domain": registration.domain,
        "configurations": configurations,
    }


def _select_output_noise(train_Yvar: Tensor | None, index: int, train_Y: Tensor) -> Tensor | None:
    if train_Yvar is None:
        return None
    if train_Yvar.shape != train_Y.shape:
        raise ValueError(
            "Independent multi-output requires train_Yvar to have the same shape as train_Y."
        )
    return train_Yvar[..., index : index + 1]


def _instantiate_registered_model(
    model_class: type[Any],
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None,
    backend_kwargs: dict[str, Any],
) -> Any:
    """Instantiate a registered backend without assuming positional optional args."""

    return model_class(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        **backend_kwargs,
    )


def _create_independent_material_surrogate(
    *,
    model_class: type[Any],
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None,
    backend_kwargs: dict[str, Any],
) -> ModelListGP:
    if train_Y.ndim != 2 or train_Y.shape[-1] < 2:
        raise ValueError(
            "Independent multi-output requires train_Y with shape [n, m] and at least two outputs."
        )

    models = []
    for output_index in range(train_Y.shape[-1]):
        output_Y = train_Y[..., output_index : output_index + 1]
        output_Yvar = _select_output_noise(train_Yvar, output_index, train_Y)
        models.append(
            _instantiate_registered_model(
                model_class,
                train_X,
                output_Y,
                output_Yvar,
                backend_kwargs,
            )
        )
    return ModelListGP(*models)


def create_material_surrogate(
    family: str,
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None = None,
    /,
    *,
    kind: str = "gp",
    input_mode: str = "continuous",
    output_mode: str = "scalar",
    **backend_kwargs: Any,
) -> Any:
    """Construct one registered material GP/DKL model through a common API.

    ``output_mode="independent"`` creates one scalar surrogate per output and
    returns a :class:`ModelListGP`. Other constructor arguments are delegated
    unchanged to each selected backend model.
    """

    if not isinstance(train_X, Tensor) or not isinstance(train_Y, Tensor):
        raise TypeError("train_X and train_Y must be torch.Tensor instances.")

    spec = RegisteredMaterialSurrogateSpec(
        family=family,
        kind=kind,
        input_mode=input_mode,
        output_mode=output_mode,
    )
    model_class = get_material_family(spec.family).resolve_model_class(spec.variant)

    if spec.output_mode == "independent":
        return _create_independent_material_surrogate(
            model_class=model_class,
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            backend_kwargs=backend_kwargs,
        )

    return _instantiate_registered_model(
        model_class,
        train_X,
        train_Y,
        train_Yvar,
        backend_kwargs,
    )


__all__ = [
    "MaterialGaussianKind",
    "MaterialInputMode",
    "MaterialOutputMode",
    "RegisteredMaterialSurrogateSpec",
    "SUPPORTED_MATERIAL_GAUSSIAN_KINDS",
    "SUPPORTED_MATERIAL_INPUT_MODES",
    "SUPPORTED_MATERIAL_OUTPUT_MODES",
    "create_material_surrogate",
    "material_model_variant",
    "material_surrogate_capabilities",
    "normalize_material_gaussian_kind",
    "normalize_material_input_mode",
    "normalize_material_output_mode",
]
