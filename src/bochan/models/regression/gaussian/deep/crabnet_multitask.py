"""Correlated multi-output Gaussian process over a shared CrabNet representation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

from gpytorch.kernels import MultitaskKernel
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import CrabNetEncoder, MaterialProcessFusion
from bochan.composition.encoders.crabnet import Checkpoint

from .crabnet import (
    CrabNetInputTransform,
    _CrabNetGPFeatureExtractor,
    _resolve_input_transform,
    _resolve_material_encoder,
    _validate_element_ids,
    _validate_model_inputs,
)
from .deepkernel import InputTransformArg, OutcomeTransformArg
from .deepkernel_configurable import DeepKernelGaussianGPModel


class CrabNetMultiTaskGPModel(DeepKernelGaussianGPModel):
    """Correlated multi-output GP with one shared frozen CrabNet encoder.

    ``train_Y`` is a wide ``[n, m]`` tensor with ``m >= 2`` continuous material
    properties observed at the same design points.  Composition and continuous
    process variables are mapped to one shared latent representation, and the
    exact GP uses a :class:`gpytorch.kernels.MultitaskKernel` to learn
    cross-property covariance.  This is intentionally different from bochan's
    independent CrabNet ``ModelListGP`` path, where every output owns a separate
    encoder, projection, and GP.

    The CrabNet encoder stays frozen in this first multitask model.  The shared
    material/process projection, data kernel, task covariance, likelihood, and
    task-specific observation noise remain trainable.  A future multitask-DKL
    variant can add encoder fine-tuning without changing this model's contract.

    Args:
        train_X: ``[n, composition/search + process]`` raw model coordinates.
        train_Y: Wide continuous targets with shape ``[n, m]`` and ``m >= 2``.
        train_Yvar: Reserved for future fixed-noise support.
        element_ids: Fixed atomic-number vocabulary for the composition site.
        encoder: Optional CrabNet adapter or raw upstream encoder.
        checkpoint: Optional encoder checkpoint.
        latent_dim: Width of the shared material/process projection.
        fusion: Material/process fusion strategy. ``"concat"`` is supported.
        projection: Optional shared projection module.
        strict_checkpoint: Require a complete checkpoint state.
        likelihood: Optional multitask Gaussian likelihood.
        input_transform: Optional :class:`CrabNetInputTransform`; ``"DEFAULT"``
            preserves composition fractions and normalizes process columns.
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
        if train_Y.ndim != 2 or train_Y.shape[-1] < 2:
            raise ValueError(
                "CrabNetMultiTaskGPModel requires wide train_Y with shape [n, m] "
                "and at least two target columns."
            )
        if train_Yvar is not None:
            raise NotImplementedError(
                "CrabNetMultiTaskGPModel does not yet support train_Yvar."
            )
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")

        validated_element_ids = _validate_element_ids(element_ids)
        composition_dim = int(validated_element_ids.numel())
        if isinstance(input_transform, CrabNetInputTransform):
            if input_transform.composition_dim != composition_dim:
                raise ValueError(
                    "CrabNetInputTransform.n_components must match element_ids: "
                    f"{input_transform.composition_dim} != {composition_dim}."
                )
            if train_X.shape[-1] != input_transform.input_dim:
                raise ValueError(
                    "train_X width must match CrabNetInputTransform.input_dim: "
                    f"{train_X.shape[-1]} != {input_transform.input_dim}."
                )
            process_dim = input_transform.process_dim
        else:
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
            process_dim = train_X.shape[-1] - composition_dim

        material_encoder = _resolve_material_encoder(
            encoder,
            checkpoint,
            strict_checkpoint=strict_checkpoint,
        )
        feature_extractor = _CrabNetGPFeatureExtractor(
            material_encoder=material_encoder,
            element_ids=validated_element_ids,
            process_dim=process_dim,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
        )
        resolved_input_transform = (
            input_transform
            if isinstance(input_transform, CrabNetInputTransform)
            else _resolve_input_transform(
                train_X,
                composition_dim=composition_dim,
                input_transform=input_transform,
            )
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

        if not isinstance(self.deepkernel.covar_module, MultitaskKernel):
            raise RuntimeError(
                "CrabNetMultiTaskGPModel requires a correlated MultitaskKernel."
            )
        transformed_train_X = cast(tuple[Tensor, ...], self.deepkernel.train_inputs)[0]
        self.crabnet_feature_extractor.validate_input(transformed_train_X)

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> CrabNetMultiTaskGPModel:
        """Move model components while preserving the wrapper dtype/device contract."""

        module = super()._apply(fn, recurse=recurse)
        reference = next(
            (
                value
                for value in (*self.parameters(), *self.buffers())
                if value.is_floating_point()
            ),
            None,
        )
        if reference is not None:
            self._model_dtype = reference.dtype
            self._model_device = reference.device
        return cast(CrabNetMultiTaskGPModel, module)

    @property
    def crabnet_feature_extractor(self) -> _CrabNetGPFeatureExtractor:
        """Return the shared CrabNet material/process feature extractor."""

        return cast(_CrabNetGPFeatureExtractor, self.deepkernel.feature_extractor)

    @property
    def material_encoder(self) -> CrabNetEncoder:
        """Return the single shared frozen CrabNet encoder."""

        return self.crabnet_feature_extractor.material_encoder

    @property
    def projection(self) -> nn.Module:
        """Return the shared trainable latent projection."""

        return self.crabnet_feature_extractor.projection

    @property
    def fusion(self) -> MaterialProcessFusion:
        """Return the shared material/process fusion module."""

        return self.crabnet_feature_extractor.fusion

    @property
    def element_ids(self) -> Tensor:
        """Return the fixed atomic-number vocabulary."""

        return self.crabnet_feature_extractor.element_ids

    @property
    def composition_dim(self) -> int:
        """Return the number of material-fraction components."""

        return self.crabnet_feature_extractor.composition_dim

    @property
    def process_dim(self) -> int:
        """Return the number of continuous process columns."""

        return self.crabnet_feature_extractor.process_dim

    @property
    def num_tasks(self) -> int:
        """Return the number of correlated material-property tasks."""

        return int(self.num_outputs)

    @property
    def task_covar_module(self) -> nn.Module:
        """Expose the learned task covariance module for diagnostics."""

        return self.deepkernel.covar_module.task_covar_module

    @property
    def task_kernel(self) -> nn.Module:
        """Alias the task covariance module to the common multitask interface."""

        return self.task_covar_module


__all__ = ["CrabNetMultiTaskGPModel"]
