"""Registry-driven explicit-task Gaussian surrogate construction for materials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from botorch.models import MultiTaskGP
from torch import Tensor, nn

from .common import (
    MaterialExplicitTaskSpec,
    build_material_explicit_task_surrogate,
    get_material_family,
    split_material_task_feature,
    validate_explicit_material_task_data,
)
from .surrogate_factory import (
    MaterialGaussianKind,
    MaterialInputMode,
    normalize_material_gaussian_kind,
    normalize_material_input_mode,
)


class RegisteredMaterialFeatureExtractor(nn.Module):
    """Reuse a registered continuous backend's input transform and representation."""

    def __init__(
        self,
        *,
        input_transform: nn.Module | None,
        feature_extractor: nn.Module,
        output_dim: int,
    ) -> None:
        super().__init__()
        if input_transform is not None and not isinstance(input_transform, nn.Module):
            raise TypeError("input_transform must be a torch.nn.Module or None.")
        if not isinstance(feature_extractor, nn.Module):
            raise TypeError("feature_extractor must be a torch.nn.Module.")
        if isinstance(output_dim, bool) or not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError("output_dim must be a positive integer.")

        self.input_transform = input_transform
        self.feature_extractor = feature_extractor
        self.output_dim = output_dim

    def forward(self, X: Tensor) -> Tensor:
        transformed = self.input_transform(X) if self.input_transform is not None else X
        return cast(Tensor, self.feature_extractor(transformed))


class RegisteredMixedMaterialFeatureExtractor(nn.Module):
    """Preserve a mixed backend's continuous representation and categorical coordinates.

    The configured mixed scalar backend already knows which raw columns are
    continuous versus categorical. This adapter reuses its input transform,
    continuous feature extractor, latent scaling, and exact output layout so
    the backend's mixed covariance module can be transferred unchanged to the
    explicit-task ``MultiTaskGP``.
    """

    def __init__(
        self,
        *,
        input_transform: nn.Module | None,
        feature_extractor: nn.Module,
        scale_to_bounds: nn.Module,
        ord_dims: tuple[int, ...],
        cat_dims: tuple[int, ...],
        latent_dim: int,
        preserve_input_layout: bool,
        kernel_ord_dims: tuple[int, ...],
        kernel_cat_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        if input_transform is not None and not isinstance(input_transform, nn.Module):
            raise TypeError("input_transform must be a torch.nn.Module or None.")
        if not isinstance(feature_extractor, nn.Module):
            raise TypeError("feature_extractor must be a torch.nn.Module.")
        if not isinstance(scale_to_bounds, nn.Module):
            raise TypeError("scale_to_bounds must be a torch.nn.Module.")
        if not cat_dims:
            raise ValueError("cat_dims must not be empty for a mixed material representation.")
        if latent_dim < 0:
            raise ValueError("latent_dim must be non-negative.")

        self.input_transform = input_transform
        self.feature_extractor = feature_extractor
        self.scale_to_bounds = scale_to_bounds
        self.ord_dims = ord_dims
        self.cat_dims = cat_dims
        self.latent_dim = latent_dim
        self.preserve_input_layout = preserve_input_layout
        self.kernel_ord_dims = kernel_ord_dims
        self.kernel_cat_dims = kernel_cat_dims
        self.output_dim = latent_dim + len(cat_dims)

    def forward(self, X: Tensor) -> Tensor:
        transformed = self.input_transform(X) if self.input_transform is not None else X
        if not self.ord_dims:
            return transformed[..., list(self.cat_dims)]

        cont_x = transformed[..., list(self.ord_dims)]
        cat_x = transformed[..., list(self.cat_dims)]
        projected = cast(Tensor, self.feature_extractor(cont_x))
        projected = cast(Tensor, self.scale_to_bounds(projected))
        if projected.shape[-1] != self.latent_dim:
            raise ValueError(
                "mixed feature extractor output width does not match latent_dim: "
                f"{projected.shape[-1]} != {self.latent_dim}."
            )

        if self.preserve_input_layout:
            out = torch.empty(
                *transformed.shape[:-1],
                self.output_dim,
                device=projected.device,
                dtype=projected.dtype,
            )
            out[..., list(self.kernel_ord_dims)] = projected
            out[..., list(self.kernel_cat_dims)] = cat_x.to(dtype=projected.dtype)
            return out
        return torch.cat((projected, cat_x.to(dtype=projected.dtype)), dim=-1)


@dataclass(frozen=True, slots=True)
class RegisteredMaterialExplicitTaskSpec:
    """Serializable identity for one registry-backed explicit-task surrogate."""

    family: str
    kind: MaterialGaussianKind | str = "gp"
    input_mode: MaterialInputMode | str = "continuous"

    def __post_init__(self) -> None:
        registration = get_material_family(self.family)
        normalized_kind = normalize_material_gaussian_kind(cast(str, self.kind))
        normalized_input = normalize_material_input_mode(cast(str, self.input_mode))
        prefix = "mixed_" if normalized_input == "mixed" else ""
        variant = f"{prefix}{normalized_kind}"
        if not registration.supports(cast(Any, variant)):
            raise ValueError(
                f"Material family {registration.family!r} does not support "
                f"explicit-task base variant {variant!r}."
            )
        object.__setattr__(self, "family", registration.family)
        object.__setattr__(self, "kind", normalized_kind)
        object.__setattr__(self, "input_mode", normalized_input)

    @property
    def domain(self) -> str:
        return get_material_family(self.family).domain

    @property
    def base_variant(self) -> str:
        prefix = "mixed_" if self.input_mode == "mixed" else ""
        return f"{prefix}{self.kind}"

    def as_dict(self) -> dict[str, str]:
        return {
            "family": self.family,
            "domain": self.domain,
            "kind": cast(str, self.kind),
            "input_mode": cast(str, self.input_mode),
            "task_mode": "explicit",
            "base_variant": self.base_variant,
        }


def _normalize_scalar_target(train_Y: Tensor) -> Tensor:
    if train_Y.ndim == 1:
        return train_Y.unsqueeze(-1)
    if train_Y.ndim == 2 and train_Y.shape[-1] == 1:
        return train_Y
    raise ValueError("Explicit-task train_Y must have shape [n] or [n, 1].")


def _extract_registered_representation(model: Any) -> RegisteredMaterialFeatureExtractor:
    deepkernel = getattr(model, "deepkernel", None)
    feature_extractor = getattr(deepkernel, "feature_extractor", None)
    if not isinstance(feature_extractor, nn.Module):
        raise RuntimeError(
            "Registered scalar material backend must expose deepkernel.feature_extractor."
        )

    latent_dim = getattr(model, "latent_dim", None)
    if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
        latent_dim = getattr(feature_extractor, "output_dim", None)
    if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
        raise RuntimeError("Registered scalar material backend does not expose a valid latent_dim.")

    input_transform = getattr(model, "input_transform", None)
    if input_transform is not None and not isinstance(input_transform, nn.Module):
        raise RuntimeError("Registered scalar material backend input_transform is not a module.")

    return RegisteredMaterialFeatureExtractor(
        input_transform=input_transform,
        feature_extractor=feature_extractor,
        output_dim=latent_dim,
    )


def _extract_registered_mixed_representation(
    model: Any,
) -> tuple[RegisteredMixedMaterialFeatureExtractor, nn.Module]:
    deepkernel = getattr(model, "deepkernel", None)
    feature_extractor = getattr(deepkernel, "feature_extractor", None)
    scale_to_bounds = getattr(deepkernel, "scale_to_bounds", None)
    covar_module = getattr(deepkernel, "covar_module", None)
    if not isinstance(feature_extractor, nn.Module):
        raise RuntimeError("Registered mixed backend must expose deepkernel.feature_extractor.")
    if not isinstance(scale_to_bounds, nn.Module):
        raise RuntimeError("Registered mixed backend must expose deepkernel.scale_to_bounds.")
    if not isinstance(covar_module, nn.Module):
        raise RuntimeError("Registered mixed backend must expose deepkernel.covar_module.")

    ord_dims = tuple(int(value) for value in getattr(deepkernel, "ord_dims", ()))
    cat_dims = tuple(int(value) for value in getattr(deepkernel, "cat_dims", ()))
    kernel_ord_dims = tuple(int(value) for value in getattr(deepkernel, "kernel_ord_dims", ()))
    kernel_cat_dims = tuple(int(value) for value in getattr(deepkernel, "kernel_cat_dims", ()))
    latent_dim = getattr(deepkernel, "latent_dim", None)
    preserve = bool(getattr(deepkernel, "_preserve_input_layout", False))
    if not cat_dims:
        raise RuntimeError("Registered mixed backend does not expose categorical dimensions.")
    if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim < 0:
        raise RuntimeError("Registered mixed backend does not expose a valid latent_dim.")

    input_transform = getattr(model, "input_transform", None)
    if input_transform is not None and not isinstance(input_transform, nn.Module):
        raise RuntimeError("Registered mixed backend input_transform is not a module.")

    representation = RegisteredMixedMaterialFeatureExtractor(
        input_transform=input_transform,
        feature_extractor=feature_extractor,
        scale_to_bounds=scale_to_bounds,
        ord_dims=ord_dims,
        cat_dims=cat_dims,
        latent_dim=latent_dim,
        preserve_input_layout=preserve,
        kernel_ord_dims=kernel_ord_dims,
        kernel_cat_dims=kernel_cat_dims,
    )
    return representation, covar_module


def registered_material_explicit_task_capabilities(family: str) -> dict[str, Any]:
    """Return explicit-task GP/DKL support for one registered material family."""

    registration = get_material_family(family)
    continuous_kinds = [
        kind for kind in ("gp", "dkl") if registration.supports(cast(Any, kind))
    ]
    mixed_kinds = [
        kind
        for kind in ("gp", "dkl")
        if registration.supports(cast(Any, f"mixed_{kind}"))
    ]
    input_modes = []
    if continuous_kinds:
        input_modes.append("continuous")
    if mixed_kinds:
        input_modes.append("mixed")
    return {
        "family": registration.family,
        "domain": registration.domain,
        "task_mode": "explicit",
        "input_modes": input_modes,
        "gaussian_kinds": continuous_kinds,
        "mixed_gaussian_kinds": mixed_kinds,
        "mixed_explicit_task": bool(mixed_kinds),
    }


def create_registered_material_explicit_task_surrogate(
    family: str,
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None = None,
    /,
    *,
    kind: str = "gp",
    input_mode: str = "continuous",
    task_spec: MaterialExplicitTaskSpec | None = None,
    rank: int | None = None,
    likelihood: Any = None,
    outcome_transform: Any = None,
    covar_module: Any = None,
    mean_module: Any = None,
    task_covar_prior: Any = None,
    validate_task_values: bool = True,
    **backend_kwargs: Any,
) -> MultiTaskGP:
    """Build a registry-backed material ``f(x, task)`` surrogate.

    For ``input_mode="mixed"``, the selected registered mixed GP/DKL backend is
    first configured without the explicit task column. Its continuous material
    representation, categorical coordinates, latent scaling, and mixed data
    covariance are then transferred to the explicit-task ``MultiTaskGP``. The
    task covariance remains the separate BoTorch task-index kernel.
    """

    if not isinstance(train_X, Tensor) or not isinstance(train_Y, Tensor):
        raise TypeError("train_X and train_Y must be torch.Tensor instances.")
    if task_spec is None:
        task_spec = MaterialExplicitTaskSpec()
    if not isinstance(task_spec, MaterialExplicitTaskSpec):
        raise TypeError("task_spec must be a MaterialExplicitTaskSpec.")

    spec = RegisteredMaterialExplicitTaskSpec(
        family=family,
        kind=kind,
        input_mode=input_mode,
    )
    normalized_task_feature, _ = validate_explicit_material_task_data(
        train_X,
        train_Y,
        train_Yvar,
        task_feature=task_spec.task_feature,
        all_tasks=task_spec.all_tasks,
        model_name=f"{spec.family} explicit-task surrogate",
    )
    material_X, _, _ = split_material_task_feature(
        train_X,
        task_feature=normalized_task_feature,
    )
    scalar_Y = _normalize_scalar_target(train_Y)
    scalar_Yvar = _normalize_scalar_target(train_Yvar) if train_Yvar is not None else None

    registration = get_material_family(spec.family)
    model_class = registration.resolve_model_class(cast(Any, spec.base_variant))
    scalar_model = model_class(
        material_X,
        scalar_Y,
        train_Yvar=scalar_Yvar,
        **backend_kwargs,
    )

    resolved_covar_module = covar_module
    if spec.input_mode == "mixed":
        representation, mixed_covar_module = _extract_registered_mixed_representation(
            scalar_model
        )
        if resolved_covar_module is None:
            resolved_covar_module = mixed_covar_module
    else:
        representation = _extract_registered_representation(scalar_model)

    model = build_material_explicit_task_surrogate(
        train_X=train_X,
        train_Y=scalar_Y,
        train_Yvar=scalar_Yvar,
        feature_extractor=representation,
        task_spec=task_spec,
        latent_dim=representation.output_dim,
        rank=rank,
        likelihood=likelihood,
        outcome_transform=outcome_transform,
        covar_module=resolved_covar_module,
        mean_module=mean_module,
        task_covar_prior=task_covar_prior,
        validate_task_values=validate_task_values,
    )
    model.material_family = spec.family
    model.material_domain = spec.domain
    model.material_gaussian_kind = cast(str, spec.kind)
    model.material_input_mode = cast(str, spec.input_mode)
    model.material_task_mode = "explicit"
    model.material_base_model_class = model_class.__name__
    return model


__all__ = [
    "RegisteredMaterialExplicitTaskSpec",
    "RegisteredMaterialFeatureExtractor",
    "RegisteredMixedMaterialFeatureExtractor",
    "create_registered_material_explicit_task_surrogate",
    "registered_material_explicit_task_capabilities",
]
