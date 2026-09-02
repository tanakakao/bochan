"""Shared GP / DKL surrogate construction contracts for material models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from torch import Tensor, nn

MaterialSurrogateKind = Literal["gp", "dkl"]


@dataclass(frozen=True)
class MaterialSurrogateSpec:
    """Describe the Gaussian surrogate wrapped around a material representation.

    ``gp`` and ``dkl`` intentionally share the same exact-GP backend.  Their
    behavioral difference is the encoder training policy established before the
    surrogate is built: GP keeps the encoder frozen, while DKL exposes selected
    encoder parameters for joint optimization.
    """

    kind: MaterialSurrogateKind = "gp"
    mixed: bool = False
    latent_dim: int | None = None
    ext_type: str = "DEFAULT"

    def __post_init__(self) -> None:
        if self.kind not in {"gp", "dkl"}:
            raise ValueError("kind must be 'gp' or 'dkl'.")
        if self.latent_dim is not None and (
            isinstance(self.latent_dim, bool)
            or not isinstance(self.latent_dim, int)
            or self.latent_dim <= 0
        ):
            raise ValueError("latent_dim must be a positive integer or None.")
        if not isinstance(self.ext_type, str) or not self.ext_type:
            raise ValueError("ext_type must be a non-empty string.")


def resolve_material_latent_dim(
    feature_extractor: nn.Module,
    latent_dim: int | None = None,
) -> int:
    """Resolve and validate the GP-kernel feature width for a material encoder."""

    if not isinstance(feature_extractor, nn.Module):
        raise TypeError("feature_extractor must be a torch.nn.Module.")
    declared = getattr(feature_extractor, "output_dim", None)

    if latent_dim is None:
        if declared is None:
            raise ValueError(
                "latent_dim is required when feature_extractor does not expose output_dim."
            )
        latent_dim = int(declared)
    elif isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
        raise ValueError("latent_dim must be a positive integer.")

    if declared is not None and int(declared) != latent_dim:
        raise ValueError(
            "feature_extractor.output_dim does not match latent_dim: "
            f"{int(declared)} != {latent_dim}."
        )
    return latent_dim


def build_material_gaussian_surrogate(
    *,
    train_X: Tensor,
    train_Y: Tensor,
    feature_extractor: nn.Module,
    spec: MaterialSurrogateSpec,
    train_Yvar: Tensor | None = None,
    cat_dims: Sequence[int] = (),
    likelihood: Any = None,
    input_transform: Any = "DEFAULT",
    outcome_transform: Any = "DEFAULT",
):
    """Build the established exact-GP backend around a material feature extractor.

    This function deliberately delegates observation-aware behavior, known-noise
    handling, input/outcome transforms, and correlated wide-output construction
    to the existing configurable deep-kernel Gaussian wrappers.  It therefore
    does not introduce a second missing-target or likelihood path.

    ``spec.kind`` records GP versus DKL semantics but does not select a different
    Gaussian class: DKL is represented by a feature extractor whose encoder
    parameters have already been made trainable by the shared training policy.
    """

    if not isinstance(spec, MaterialSurrogateSpec):
        raise TypeError("spec must be a MaterialSurrogateSpec.")
    if not isinstance(train_X, Tensor) or not isinstance(train_Y, Tensor):
        raise TypeError("train_X and train_Y must be torch.Tensor instances.")
    latent_dim = resolve_material_latent_dim(feature_extractor, spec.latent_dim)

    # Local import keeps the canonical materials package independent from the
    # historical deep package during ordinary contract-only imports.
    from bochan.models.regression.gaussian.deep.deepkernel_configurable import (
        DeepKernelGaussianGPModel,
        DeepKernelGaussianMixedGPModel,
    )

    common_kwargs = dict(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        likelihood=likelihood,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
        ext_type=spec.ext_type,
        feature_extractor=feature_extractor,
        latent_dim=latent_dim,
    )

    if spec.mixed:
        if not cat_dims:
            raise ValueError("cat_dims must not be empty for a mixed material surrogate.")
        return DeepKernelGaussianMixedGPModel(
            cat_dims=tuple(int(index) for index in cat_dims),
            **common_kwargs,
        )
    if cat_dims:
        raise ValueError("cat_dims must be empty for a non-mixed material surrogate.")
    return DeepKernelGaussianGPModel(**common_kwargs)


__all__ = [
    "MaterialSurrogateKind",
    "MaterialSurrogateSpec",
    "build_material_gaussian_surrogate",
    "resolve_material_latent_dim",
]
