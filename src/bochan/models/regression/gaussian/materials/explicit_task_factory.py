"""Registry-driven explicit-task Gaussian surrogate construction for materials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

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
    normalize_material_gaussian_kind,
)


class RegisteredMaterialFeatureExtractor(nn.Module):
    """Reuse a registered scalar backend's raw-input transform and representation.

    The adapter owns only the representation path required by the explicit-task
    surrogate. The scalar GP used to configure the backend-specific material
    encoder is intentionally discarded after construction.
    """

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
        """Apply the registered backend's input transform and latent encoder."""

        transformed = self.input_transform(X) if self.input_transform is not None else X
        return cast(Tensor, self.feature_extractor(transformed))


@dataclass(frozen=True, slots=True)
class RegisteredMaterialExplicitTaskSpec:
    """Serializable identity for one registry-backed explicit-task surrogate."""

    family: str
    kind: MaterialGaussianKind | str = "gp"

    def __post_init__(self) -> None:
        registration = get_material_family(self.family)
        normalized_kind = normalize_material_gaussian_kind(cast(str, self.kind))
        variant = cast(str, normalized_kind)
        if not registration.supports(cast(Any, variant)):
            raise ValueError(
                f"Material family {registration.family!r} does not support "
                f"explicit-task base variant {variant!r}."
            )
        object.__setattr__(self, "family", registration.family)
        object.__setattr__(self, "kind", normalized_kind)

    @property
    def domain(self) -> str:
        return get_material_family(self.family).domain

    @property
    def base_variant(self) -> str:
        return cast(str, self.kind)

    def as_dict(self) -> dict[str, str]:
        return {
            "family": self.family,
            "domain": self.domain,
            "kind": cast(str, self.kind),
            "input_mode": "continuous",
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


def registered_material_explicit_task_capabilities(family: str) -> dict[str, Any]:
    """Return explicit-task GP/DKL support for one registered material family."""

    registration = get_material_family(family)
    kinds = [kind for kind in ("gp", "dkl") if registration.supports(cast(Any, kind))]
    return {
        "family": registration.family,
        "domain": registration.domain,
        "task_mode": "explicit",
        "input_modes": ["continuous"],
        "gaussian_kinds": kinds,
        "mixed_explicit_task": False,
    }


def create_registered_material_explicit_task_surrogate(
    family: str,
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None = None,
    /,
    *,
    kind: str = "gp",
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

    ``train_X`` follows the Phase 35 long-format contract and includes one task
    feature. The task column is removed before constructing the registered
    scalar GP/DKL backend, so composition/structure encoders never receive task
    ids. The backend's configured raw-input transform and material feature
    extractor are then reused by the Phase 36 ``MultiTaskGP`` builder.

    Phase 37 intentionally supports continuous material inputs only. Mixed
    categorical explicit-task models require preserving the mixed categorical
    kernel in addition to the material representation and are handled as a
    separate design axis.
    """

    if not isinstance(train_X, Tensor) or not isinstance(train_Y, Tensor):
        raise TypeError("train_X and train_Y must be torch.Tensor instances.")
    if task_spec is None:
        task_spec = MaterialExplicitTaskSpec()
    if not isinstance(task_spec, MaterialExplicitTaskSpec):
        raise TypeError("task_spec must be a MaterialExplicitTaskSpec.")

    spec = RegisteredMaterialExplicitTaskSpec(family=family, kind=kind)
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
        covar_module=covar_module,
        mean_module=mean_module,
        task_covar_prior=task_covar_prior,
        validate_task_values=validate_task_values,
    )
    model.material_family = spec.family
    model.material_domain = spec.domain
    model.material_gaussian_kind = cast(str, spec.kind)
    model.material_input_mode = "continuous"
    model.material_task_mode = "explicit"
    model.material_base_model_class = model_class.__name__
    return model


__all__ = [
    "RegisteredMaterialExplicitTaskSpec",
    "RegisteredMaterialFeatureExtractor",
    "create_registered_material_explicit_task_surrogate",
    "registered_material_explicit_task_capabilities",
]
