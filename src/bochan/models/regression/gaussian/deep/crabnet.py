"""CrabNet feature extraction with exact Gaussian processes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

from botorch.models.transforms.input import Normalize
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import (
    CrabNetEncoder,
    MaterialProcessFusion,
)
from bochan.composition.encoders.crabnet import Checkpoint

from .deepkernel import InputTransformArg, OutcomeTransformArg
from .deepkernel_configurable import DeepKernelGaussianGPModel
from .material import (
    CompositionMaterialInputTransform,
    EncoderTrainingMode,
    MaterialGPFeatureExtractor,
    _validate_composition_element_ids,
    _validate_composition_model_inputs,
)

_TrainableEncoderLayers = int | Literal["all"]


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


def _validate_trainable_encoder_layers(
    trainable_encoder_layers: _TrainableEncoderLayers,
) -> _TrainableEncoderLayers:
    """Validate a partial/full CrabNet fine-tuning configuration."""

    if trainable_encoder_layers == "all":
        return "all"
    if (
        isinstance(trainable_encoder_layers, bool)
        or not isinstance(trainable_encoder_layers, int)
        or trainable_encoder_layers <= 0
    ):
        raise ValueError("trainable_encoder_layers must be a positive integer or 'all'.")
    return trainable_encoder_layers


def _crabnet_transformer_layers(
    material_encoder: CrabNetEncoder,
) -> tuple[nn.Module, ...]:
    """Return the ordered upstream Transformer layers for partial unfreezing."""

    transformer = getattr(material_encoder.encoder, "transformer_encoder", None)
    layers = getattr(transformer, "layers", None)
    if not isinstance(layers, (nn.ModuleList, nn.Sequential)) or len(layers) == 0:
        raise ValueError(
            "Partial CrabNet unfreezing requires "
            "encoder.transformer_encoder.layers to be a non-empty "
            "torch.nn.ModuleList or torch.nn.Sequential. Use "
            "trainable_encoder_layers='all' for an injected encoder without "
            "the upstream CrabNet Transformer structure."
        )
    return tuple(layers)


def _configure_dkl_encoder(
    material_encoder: CrabNetEncoder,
    trainable_encoder_layers: _TrainableEncoderLayers,
) -> tuple[EncoderTrainingMode, tuple[nn.Module, ...]]:
    """Unfreeze the requested encoder parameters and return its train policy."""

    for parameter in material_encoder.parameters():
        parameter.requires_grad_(False)

    if trainable_encoder_layers == "all":
        parameters = tuple(material_encoder.parameters())
        if not parameters:
            raise ValueError("The CrabNet encoder exposes no parameters to fine-tune.")
        for parameter in parameters:
            parameter.requires_grad_(True)
        return "full", ()

    layers = _crabnet_transformer_layers(material_encoder)
    if trainable_encoder_layers > len(layers):
        raise ValueError(
            "trainable_encoder_layers exceeds the number of CrabNet "
            f"Transformer layers: {trainable_encoder_layers} > {len(layers)}."
        )
    trainable_modules = layers[-trainable_encoder_layers:]
    parameters = tuple(parameter for module in trainable_modules for parameter in module.parameters())
    if not parameters:
        raise ValueError("The selected CrabNet Transformer layers expose no parameters to fine-tune.")
    for parameter in parameters:
        parameter.requires_grad_(True)
    return "partial", trainable_modules


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

    ``train_X`` is a standard floating-point BoTorch tensor. By default, its
    first ``len(element_ids)`` columns contain fractions in the same order as
    the fixed atomic-number vocabulary ``element_ids``; any remaining columns
    are continuous process features. This is the low-level tensor contract,
    not a second formula API. A :class:`CompositionMaterialInputTransform` may instead map
    canonical tabular composition coordinates to that packed representation.

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
        fusion: Material/process fusion strategy. Concat is currently supported.
        projection: Optional projection module.  A linear projection is used
            by default.
        strict_checkpoint: Require a complete encoder checkpoint state.
        likelihood: Optional GPyTorch likelihood.
        input_transform: ``"DEFAULT"`` normalizes process columns only.
            :class:`CompositionMaterialInputTransform` additionally supports canonical
            tabular composition coordinates while preserving autograd.
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
        model_name = self.__class__.__name__
        if train_Y.ndim > 1 and train_Y.shape[-1] != 1:
            raise ValueError(f"{model_name} currently supports single-output train_Y only.")
        if train_Yvar is not None:
            raise NotImplementedError(f"{model_name} does not yet support train_Yvar.")
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")

        validated_element_ids = _validate_composition_element_ids(element_ids)
        composition_dim = int(validated_element_ids.numel())
        if isinstance(input_transform, CompositionMaterialInputTransform):
            if input_transform.composition_dim != composition_dim:
                raise ValueError(
                    "CompositionMaterialInputTransform.n_components must match element_ids: "
                    f"{input_transform.composition_dim} != {composition_dim}."
                )
            if train_X.shape[-1] != input_transform.input_dim:
                raise ValueError(
                    "train_X width must match CompositionMaterialInputTransform.input_dim: "
                    f"{train_X.shape[-1]} != {input_transform.input_dim}."
                )
            process_dim = input_transform.process_dim
        else:
            if train_X.shape[-1] < composition_dim:
                raise ValueError(
                    "train_X must contain one fraction column for every element_id. "
                    f"Got {train_X.shape[-1]} columns for {composition_dim} elements."
                )
            _validate_composition_model_inputs(
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
        feature_extractor = MaterialGPFeatureExtractor(
            material_encoder=material_encoder,
            element_ids=validated_element_ids,
            process_dim=process_dim,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
        )
        resolved_input_transform = (
            input_transform
            if isinstance(input_transform, CompositionMaterialInputTransform)
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

        transformed_train_X = cast(tuple[Tensor, ...], self.deepkernel.train_inputs)[0]
        self.material_feature_extractor.validate_input(transformed_train_X)

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
    def material_feature_extractor(self) -> MaterialGPFeatureExtractor:
        """Return the shared material/process feature extractor."""

        return cast(MaterialGPFeatureExtractor, self.deepkernel.feature_extractor)

    @property
    def material_encoder(self) -> CrabNetEncoder:
        """Return the CrabNet material encoder."""

        return cast(CrabNetEncoder, self.material_feature_extractor.material_encoder)

    @property
    def projection(self) -> nn.Module:
        """Return the trainable latent projection."""

        return self.material_feature_extractor.projection

    @property
    def fusion(self) -> MaterialProcessFusion:
        """Return the material/process fusion module."""

        return self.material_feature_extractor.fusion

    @property
    def element_ids(self) -> Tensor:
        """Return the fixed atomic-number vocabulary buffer."""

        return self.material_feature_extractor.element_ids

    @property
    def composition_dim(self) -> int:
        """Return the number of fraction columns."""

        return self.material_feature_extractor.composition_dim

    @property
    def process_dim(self) -> int:
        """Return the number of continuous process columns."""

        return self.material_feature_extractor.process_dim


class CrabNetDKLModel(CrabNetGPModel):
    """Exact GP that jointly fine-tunes a CrabNet material encoder.

    This model retains :class:`CrabNetGPModel`'s packed fraction/process Tensor
    contract, posterior API, process-only normalization, fusion, and latent
    projection. The difference is the encoder optimization policy: a positive
    ``trainable_encoder_layers`` value fine-tunes that many final upstream
    Transformer layers, while ``"all"`` fine-tunes the complete encoder.

    Partial fine-tuning keeps the frozen encoder prefix in evaluation mode and
    switches only the selected final Transformer layers between train/eval
    modes. Full fine-tuning follows the ordinary model train/eval mode. In both
    cases :func:`bochan.fit.deep.deepkernel.fit_deepkernel_mll` jointly
    optimizes the unfrozen encoder parameters, material/process projection,
    exact GP, and likelihood.

    Args:
        train_X: ``[n, composition_dim + process_dim]`` training inputs.
        train_Y: Single-output targets with shape ``[n]`` or ``[n, 1]``.
        train_Yvar: Reserved for future fixed-noise support.
        element_ids: Atomic-number vocabulary matching the fraction columns.
        encoder: Optional :class:`CrabNetEncoder` or raw upstream encoder.
        checkpoint: Optional upstream or adapter encoder checkpoint.
        latent_dim: Width of the projection passed to the GP kernel.
        fusion: Material/process fusion strategy.
        projection: Optional latent projection module.
        strict_checkpoint: Require a complete encoder checkpoint state.
        trainable_encoder_layers: Positive number of final upstream Transformer
            layers to unfreeze, or ``"all"`` for full encoder fine-tuning.
        likelihood: Optional GPyTorch likelihood.
        input_transform: ``"DEFAULT"`` normalizes process columns only.
        outcome_transform: Outcome transform forwarded to the Gaussian
            DeepKernel wrapper.
    """

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
        trainable_encoder_layers: int | Literal["all"] = 1,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
    ) -> None:
        resolved_trainable_layers = _validate_trainable_encoder_layers(trainable_encoder_layers)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            element_ids=element_ids,
            encoder=encoder,
            checkpoint=checkpoint,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
            strict_checkpoint=strict_checkpoint,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        training_mode, trainable_modules = _configure_dkl_encoder(
            self.material_encoder,
            resolved_trainable_layers,
        )
        self._trainable_encoder_layers = resolved_trainable_layers
        self.material_feature_extractor._configure_encoder_training(
            training_mode,
            trainable_modules,
        )

    @property
    def trainable_encoder_layers(self) -> int | Literal["all"]:
        """Return the immutable partial/full encoder fine-tuning policy."""

        return self._trainable_encoder_layers


__all__ = ["CrabNetDKLModel", "CrabNetGPModel"]
