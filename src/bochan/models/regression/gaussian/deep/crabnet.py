"""Frozen CrabNet feature extraction with an exact Gaussian process."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

import torch
from botorch.models.transforms.input import Normalize
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import (
    CrabNetEncoder,
    MaterialProcessFusion,
    build_material_process_fusion,
)
from bochan.composition.encoders.crabnet import Checkpoint

from .deepkernel import InputTransformArg, OutcomeTransformArg
from .deepkernel_configurable import DeepKernelGaussianGPModel


def _validate_element_ids(element_ids: Tensor) -> Tensor:
    """Validate and clone the fixed element vocabulary for one composition site."""

    if not torch.is_tensor(element_ids):
        raise TypeError("element_ids must be a Tensor.")
    if element_ids.ndim != 1 or element_ids.numel() == 0:
        raise ValueError("element_ids must be a non-empty one-dimensional Tensor.")
    if element_ids.dtype != torch.long:
        raise TypeError("element_ids must have dtype torch.long.")
    if (element_ids <= 0).any():
        raise ValueError("element_ids must contain positive atomic numbers.")
    if element_ids.unique().numel() != element_ids.numel():
        raise ValueError("element_ids must not contain duplicate atomic numbers.")
    return element_ids.detach().clone()


def _validate_model_inputs(X: Tensor, *, composition_dim: int, input_dim: int) -> None:
    """Validate fraction and continuous-process columns in a model input tensor."""

    if not torch.is_tensor(X):
        raise TypeError("X must be a Tensor.")
    if X.ndim == 0 or X.shape[-1] != input_dim:
        raise ValueError(
            f"X width must equal composition_dim + process_dim: {X.shape[-1] if X.ndim else 0} != {input_dim}."
        )
    if not X.is_floating_point():
        raise TypeError("X must have a floating-point dtype.")
    if not torch.isfinite(X).all():
        raise ValueError("X must contain only finite values.")

    fractions = X[..., :composition_dim]
    if (fractions < 0).any():
        raise ValueError("Composition fractions must be non-negative.")
    if not torch.allclose(
        fractions.sum(dim=-1),
        torch.ones_like(fractions[..., 0]),
        rtol=1e-4,
        atol=1e-6,
    ):
        raise ValueError("Composition fractions must sum to one.")


def _resolve_material_encoder(
    encoder: CrabNetEncoder | nn.Module | None,
    checkpoint: Checkpoint | None,
    *,
    strict_checkpoint: bool,
) -> CrabNetEncoder:
    """Build or reuse a CrabNet adapter and apply an optional checkpoint."""

    if isinstance(encoder, CrabNetEncoder):
        material_encoder = encoder
        if checkpoint is not None:
            material_encoder.load_checkpoint(checkpoint, strict=strict_checkpoint)
    else:
        material_encoder = CrabNetEncoder(
            encoder=encoder,
            checkpoint=checkpoint,
            strict_checkpoint=strict_checkpoint,
        )

    for parameter in material_encoder.parameters():
        parameter.requires_grad_(False)
    material_encoder.eval()
    return material_encoder


class _CrabNetGPFeatureExtractor(nn.Module):
    """Map fixed-vocabulary fractions and process features to a GP latent space."""

    element_ids: Tensor

    def __init__(
        self,
        *,
        material_encoder: CrabNetEncoder,
        element_ids: Tensor,
        process_dim: int,
        latent_dim: int,
        fusion: Literal["concat"] | MaterialProcessFusion,
        projection: nn.Module | None,
    ) -> None:
        super().__init__()
        if process_dim < 0:
            raise ValueError("process_dim must be non-negative.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        self.material_encoder = material_encoder
        self.register_buffer("element_ids", element_ids)
        self.composition_dim = int(element_ids.numel())
        self.process_dim = int(process_dim)
        self.input_dim = self.composition_dim + self.process_dim
        self.fusion = build_material_process_fusion(
            fusion,
            material_dim=material_encoder.output_dim,
            process_dim=self.process_dim,
        )
        self.output_dim = int(latent_dim)

        if projection is None:
            projection = nn.Linear(self.fusion.output_dim, self.output_dim)
        elif not isinstance(projection, nn.Module):
            raise TypeError("projection must be a torch.nn.Module.")

        declared_output_dim = getattr(projection, "output_dim", None)
        if declared_output_dim is not None and int(declared_output_dim) != self.output_dim:
            raise ValueError(
                f"projection.output_dim does not match latent_dim: {int(declared_output_dim)} != {self.output_dim}."
            )
        if isinstance(projection, nn.Linear):
            if projection.in_features != self.fusion.output_dim:
                raise ValueError(
                    "projection.in_features does not match the fused feature width: "
                    f"{projection.in_features} != {self.fusion.output_dim}."
                )
            if projection.out_features != self.output_dim:
                raise ValueError(
                    "projection.out_features does not match latent_dim: "
                    f"{projection.out_features} != {self.output_dim}."
                )
        self.projection = projection

    def train(self, mode: bool = True) -> _CrabNetGPFeatureExtractor:
        """Set train mode while keeping the frozen CrabNet encoder deterministic."""

        super().train(mode)
        self.material_encoder.eval()
        return self

    def validate_input(self, X: Tensor) -> None:
        """Validate the public packed-tensor contract without running CrabNet."""

        _validate_model_inputs(
            X,
            composition_dim=self.composition_dim,
            input_dim=self.input_dim,
        )

    def forward(self, X: Tensor) -> Tensor:
        """Return projected material/process features for the exact GP kernel."""

        self.validate_input(X)
        fractions = X[..., : self.composition_dim]
        expanded_ids = self.element_ids.view(
            *((1,) * (fractions.ndim - 1)),
            self.composition_dim,
        ).expand_as(fractions)
        active_element_ids = expanded_ids.masked_fill(fractions == 0, 0)
        material_features = self.material_encoder(active_element_ids, fractions)
        process_features = X[..., self.composition_dim :] if self.process_dim else None
        fused_features = self.fusion(material_features, process_features)
        projected_features = self.projection(fused_features)

        if not torch.is_tensor(projected_features):
            raise TypeError("projection must return a Tensor.")
        expected_shape = (*X.shape[:-1], self.output_dim)
        if projected_features.shape != expected_shape:
            raise ValueError(
                "projection must preserve leading dimensions and return latent_dim features: "
                f"{tuple(projected_features.shape)} != {expected_shape}."
            )
        if projected_features.device != X.device or projected_features.dtype != X.dtype:
            raise ValueError("projection output must match X's device and dtype.")
        if not torch.isfinite(projected_features).all():
            raise FloatingPointError("CrabNet material/process projection produced non-finite values.")
        return projected_features


def _resolve_input_transform(
    train_X: Tensor,
    *,
    composition_dim: int,
    input_transform: InputTransformArg,
) -> InputTransformArg:
    """Resolve DEFAULT to process-only normalization, preserving fractions."""

    if not isinstance(input_transform, str) or input_transform.upper() != "DEFAULT":
        return input_transform

    process_dims = list(range(composition_dim, train_X.shape[-1]))
    if not process_dims:
        return None
    return Normalize(d=train_X.shape[-1], indices=process_dims)


class CrabNetGPModel(DeepKernelGaussianGPModel):
    """Exact Gaussian process over frozen CrabNet material representations.

    ``train_X`` is a standard floating-point BoTorch tensor.  Its first
    ``len(element_ids)`` columns contain fractions in the same order as the
    fixed atomic-number vocabulary ``element_ids``; any remaining columns are
    continuous process features.  This is the low-level tensor contract after
    the canonical ``composition_sites`` preprocessing, not a second formula
    API.

    The CrabNet encoder is frozen and always kept in evaluation mode.  Its
    forward pass is not wrapped in ``torch.no_grad()``, so gradients with
    respect to composition fractions remain available to BoTorch.  The latent
    projection, optional custom fusion, exact GP, and likelihood remain
    trainable.  Categorical process variables are not supported in this first
    model; encode only continuous process features here.

    When optimizing the packed fraction representation directly with
    :func:`botorch.optim.optimize_acqf`, pass an intra-point linear equality
    constraint over the first ``len(element_ids)`` columns so that every
    candidate remains on the unit simplex.  Process columns stay unconstrained
    by that equality and are optimized jointly within their box bounds.

    Args:
        train_X: ``[n, composition_dim + process_dim]`` training inputs.
        train_Y: Single-output targets with shape ``[n]`` or ``[n, 1]``.
        train_Yvar: Reserved for future fixed-noise support.  It must currently
            be omitted.
        element_ids: One-dimensional atomic-number vocabulary matching the
            fraction columns in ``train_X``.
        encoder: Optional :class:`CrabNetEncoder` or raw upstream encoder.
        checkpoint: Optional upstream or adapter checkpoint forwarded to the
            material encoder.
        latent_dim: Width of the trainable projection passed to the GP kernel.
        fusion: Material/process fusion strategy.  Phase 4 supports concat.
        projection: Optional projection module.  A linear projection is used
            by default.
        strict_checkpoint: Require a complete encoder checkpoint state.
        likelihood: Optional GPyTorch likelihood.
        input_transform: ``"DEFAULT"`` normalizes process columns only.
        outcome_transform: Outcome transform forwarded to the Gaussian
            DeepKernel wrapper.
    """

    _supports_cache_root = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        element_ids: Tensor,
        encoder: CrabNetEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        strict_checkpoint: bool = True,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
    ) -> None:
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape [n, d].")
        if train_Y.ndim > 1 and train_Y.shape[-1] != 1:
            raise ValueError("CrabNetGPModel currently supports single-output train_Y only.")
        if train_Yvar is not None:
            raise NotImplementedError("CrabNetGPModel does not yet support train_Yvar.")
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")

        validated_element_ids = _validate_element_ids(element_ids)
        composition_dim = int(validated_element_ids.numel())
        if train_X.shape[-1] < composition_dim:
            raise ValueError(
                "train_X must contain one fraction column for every element_id. "
                f"Got {train_X.shape[-1]} columns for {composition_dim} elements."
            )
        _validate_model_inputs(
            train_X,
            composition_dim=composition_dim,
            input_dim=train_X.shape[-1],
        )

        material_encoder = _resolve_material_encoder(
            encoder,
            checkpoint,
            strict_checkpoint=strict_checkpoint,
        )
        feature_extractor = _CrabNetGPFeatureExtractor(
            material_encoder=material_encoder,
            element_ids=validated_element_ids,
            process_dim=train_X.shape[-1] - composition_dim,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
        )
        resolved_input_transform = _resolve_input_transform(
            train_X,
            composition_dim=composition_dim,
            input_transform=input_transform,
        )

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=None,
            likelihood=likelihood,
            input_transform=resolved_input_transform,
            outcome_transform=outcome_transform,
            feature_extractor=feature_extractor,
            latent_dim=latent_dim,
        )

        transformed_train_X = cast(tuple[Tensor, ...], self.deepkernel.train_inputs)[0]
        self.crabnet_feature_extractor.validate_input(transformed_train_X)

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> CrabNetGPModel:
        """Move every model component and refresh the wrapper dtype/device contract."""

        module = super()._apply(fn, recurse=recurse)
        reference = next(
            (value for value in (*self.parameters(), *self.buffers()) if value.is_floating_point()),
            None,
        )
        if reference is not None:
            self._model_dtype = reference.dtype
            self._model_device = reference.device
        return cast(CrabNetGPModel, module)

    @property
    def crabnet_feature_extractor(self) -> _CrabNetGPFeatureExtractor:
        """Return the material/process feature extractor owned by the inner GP."""

        return cast(_CrabNetGPFeatureExtractor, self.deepkernel.feature_extractor)

    @property
    def material_encoder(self) -> CrabNetEncoder:
        """Return the frozen CrabNet material encoder."""

        return self.crabnet_feature_extractor.material_encoder

    @property
    def projection(self) -> nn.Module:
        """Return the trainable latent projection."""

        return self.crabnet_feature_extractor.projection

    @property
    def fusion(self) -> MaterialProcessFusion:
        """Return the material/process fusion module."""

        return self.crabnet_feature_extractor.fusion

    @property
    def element_ids(self) -> Tensor:
        """Return the fixed atomic-number vocabulary buffer."""

        return self.crabnet_feature_extractor.element_ids

    @property
    def composition_dim(self) -> int:
        """Return the number of fraction columns."""

        return self.crabnet_feature_extractor.composition_dim

    @property
    def process_dim(self) -> int:
        """Return the number of continuous process columns."""

        return self.crabnet_feature_extractor.process_dim


__all__ = ["CrabNetGPModel"]
