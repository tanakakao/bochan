"""Roost composition representations with exact Gaussian processes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import MaterialProcessFusion, RoostEncoder
from bochan.composition.encoders.roost import Checkpoint

from .deepkernel import InputTransformArg, OutcomeTransformArg
from .deepkernel_configurable import DeepKernelGaussianGPModel
from .material import (
    CompositionMaterialInputTransform,
    EncoderTrainingMode,
    MaterialGPFeatureExtractor,
    _resolve_composition_input_transform,
    _validate_composition_element_ids,
    _validate_composition_model_inputs,
)

_RoostEncoderTraining = Literal["partial", "full"]


def _resolve_material_encoder(
    encoder: RoostEncoder | nn.Module | None,
    checkpoint: Checkpoint | None,
    *,
    output_dim: int | None,
    strict_checkpoint: bool,
) -> RoostEncoder:
    """Build or reuse a Roost adapter and freeze its representation backbone."""

    if isinstance(encoder, RoostEncoder):
        material_encoder = encoder
        if output_dim is not None and output_dim != material_encoder.output_dim:
            raise ValueError(
                "encoder_output_dim does not match RoostEncoder.output_dim: "
                f"{output_dim} != {material_encoder.output_dim}."
            )
        if checkpoint is not None:
            material_encoder.load_checkpoint(checkpoint, strict=strict_checkpoint)
    else:
        material_encoder = RoostEncoder(
            encoder=encoder,
            checkpoint=checkpoint,
            output_dim=output_dim,
            strict_checkpoint=strict_checkpoint,
        )

    for parameter in material_encoder.parameters():
        parameter.requires_grad_(False)
    material_encoder.eval()
    return material_encoder


def _validate_encoder_training(
    encoder_training: _RoostEncoderTraining,
) -> _RoostEncoderTraining:
    """Validate the public Roost fine-tuning mode."""

    if encoder_training not in {"partial", "full"}:
        raise ValueError("encoder_training must be 'partial' or 'full'.")
    return encoder_training


def _validate_trainable_encoder_layers(trainable_encoder_layers: int) -> int:
    """Validate the number of final Roost message-passing layers to train."""

    if (
        isinstance(trainable_encoder_layers, bool)
        or not isinstance(trainable_encoder_layers, int)
        or trainable_encoder_layers <= 0
    ):
        raise ValueError("trainable_encoder_layers must be a positive integer.")
    return trainable_encoder_layers


def _roost_fine_tuning_components(
    material_encoder: RoostEncoder,
) -> tuple[nn.Module, nn.Module, tuple[nn.Module, ...], nn.Module]:
    """Return Roost element embedding, descriptor, graphs, and crystal pool."""

    raw_encoder = material_encoder.encoder
    elem_embedding = getattr(raw_encoder, "elem_embedding", None)
    material_nn = getattr(raw_encoder, "material_nn", None)
    if not isinstance(elem_embedding, nn.Module) or not isinstance(material_nn, nn.Module):
        raise ValueError(
            "Roost fine-tuning requires encoder.elem_embedding and encoder.material_nn descriptor modules."
        )

    descriptor_embedding = getattr(material_nn, "embedding", None)
    graphs = getattr(material_nn, "graphs", None)
    cry_pool = getattr(material_nn, "cry_pool", None)
    if not isinstance(descriptor_embedding, nn.Module):
        raise ValueError("Roost fine-tuning requires encoder.material_nn.embedding.")
    if not isinstance(graphs, (nn.ModuleList, nn.Sequential)) or len(graphs) == 0:
        raise ValueError(
            "Roost fine-tuning requires encoder.material_nn.graphs to be a "
            "non-empty torch.nn.ModuleList or torch.nn.Sequential."
        )
    if not isinstance(cry_pool, (nn.ModuleList, nn.Sequential)) or len(cry_pool) == 0:
        raise ValueError(
            "Roost fine-tuning requires encoder.material_nn.cry_pool to be a "
            "non-empty torch.nn.ModuleList or torch.nn.Sequential."
        )
    return elem_embedding, material_nn, tuple(graphs), cry_pool


def _unique_module_parameters(modules: tuple[nn.Module, ...]) -> tuple[nn.Parameter, ...]:
    """Return module parameters once while preserving module order."""

    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            identifier = id(parameter)
            if identifier not in seen:
                seen.add(identifier)
                parameters.append(parameter)
    return tuple(parameters)


def _configure_dkl_encoder(
    material_encoder: RoostEncoder,
    *,
    encoder_training: _RoostEncoderTraining,
    trainable_encoder_layers: int,
) -> tuple[EncoderTrainingMode, tuple[nn.Module, ...]]:
    """Apply Roost-specific partial/full unfreezing and return its mode policy."""

    for parameter in material_encoder.parameters():
        parameter.requires_grad_(False)

    elem_embedding, material_nn, graphs, cry_pool = _roost_fine_tuning_components(material_encoder)
    if encoder_training == "full":
        parameters = _unique_module_parameters((elem_embedding, material_nn))
        if not parameters:
            raise ValueError("The Roost encoder exposes no parameters to fine-tune.")
        for parameter in parameters:
            parameter.requires_grad_(True)
        return "full", ()

    if trainable_encoder_layers > len(graphs):
        raise ValueError(
            "trainable_encoder_layers exceeds the number of Roost "
            f"message-passing layers: {trainable_encoder_layers} > {len(graphs)}."
        )
    trainable_modules = (*graphs[-trainable_encoder_layers:], cry_pool)
    parameters = _unique_module_parameters(trainable_modules)
    if not parameters:
        raise ValueError(
            "The selected Roost message-passing layers and crystal attention pool expose no parameters to fine-tune."
        )
    for parameter in parameters:
        parameter.requires_grad_(True)
    return "partial", trainable_modules


class RoostGPModel(DeepKernelGaussianGPModel):
    """Exact Gaussian process over frozen Roost composition representations.

    The first ``len(element_ids)`` columns of ``train_X`` contain atomic
    fractions in the same fixed-element order as ``element_ids``. Remaining
    columns are optional continuous process variables. A
    :class:`CompositionMaterialInputTransform` may instead convert canonical
    fraction, CLR, ALR, or ILR coordinates to this packed representation.

    The public :class:`RoostEncoder` supplies the pooled Aviary
    ``DescriptorNetwork`` representation; Aviary prediction heads are not part
    of this model. The Roost encoder is frozen and kept in evaluation mode,
    while the material/process fusion, projection, exact GP, and likelihood
    remain trainable. Encoder forward passes retain autograd with respect to
    composition fractions.

    When optimizing the packed fraction representation directly with
    :func:`botorch.optim.optimize_acqf`, pass an intra-point linear equality
    constraint over the first ``len(element_ids)`` columns so that every
    candidate remains on the unit simplex. Process columns remain outside that
    equality and are optimized jointly within their box bounds. Formula-backed
    workflows may instead optimize ALR or ILR coordinates through
    :class:`CompositionMaterialInputTransform` and decode the composition slice
    with :class:`bochan.composition.CompositionTransformer`; formula strings do
    not enter the model posterior path.

    Args:
        train_X: ``[n, composition_dim + process_dim]`` training inputs.
        train_Y: Single-output targets with shape ``[n]`` or ``[n, 1]``.
        train_Yvar: Reserved for future fixed-noise support and currently
            unsupported.
        element_ids: Atomic-number vocabulary matching the fraction columns.
        encoder: Optional :class:`RoostEncoder` or raw five-tensor Roost
            backbone. Omit it to construct Aviary's default Roost descriptor.
        checkpoint: Optional upstream Aviary or adapter checkpoint.
        encoder_output_dim: Required only when an injected raw backbone does
            not expose a positive integer ``output_dim``.
        latent_dim: Width of the trainable projection passed to the GP kernel.
        fusion: Material/process fusion strategy.
        projection: Optional projection module. A linear projection is used by
            default.
        strict_checkpoint: Require a complete encoder checkpoint state.
        likelihood: Optional GPyTorch likelihood.
        input_transform: ``"DEFAULT"`` normalizes continuous process columns
            only. ``CompositionMaterialInputTransform`` additionally supports
            canonical composition coordinates while preserving autograd.
        outcome_transform: Outcome transform forwarded to the Gaussian deep
            kernel wrapper.
    """

    _supports_cache_root = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        element_ids: Tensor,
        encoder: RoostEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        encoder_output_dim: int | None = None,
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
        if train_Y.ndim == 1:
            train_Y = train_Y.unsqueeze(-1)
        elif train_Y.ndim != 2 or train_Y.shape[-1] != 1:
            raise ValueError("RoostGPModel currently supports single-output train_Y only.")
        if train_Yvar is not None:
            raise NotImplementedError("RoostGPModel does not yet support train_Yvar.")
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
            output_dim=encoder_output_dim,
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
            else _resolve_composition_input_transform(
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
    ) -> RoostGPModel:
        """Move every model component and refresh the wrapper dtype/device contract."""

        module = super()._apply(fn, recurse=recurse)
        reference = next(
            (value for value in (*self.parameters(), *self.buffers()) if value.is_floating_point()),
            None,
        )
        if reference is not None:
            self._model_dtype = reference.dtype
            self._model_device = reference.device
        return cast(RoostGPModel, module)

    @property
    def material_feature_extractor(self) -> MaterialGPFeatureExtractor:
        """Return the shared material/process feature extractor."""

        return cast(MaterialGPFeatureExtractor, self.deepkernel.feature_extractor)

    @property
    def material_encoder(self) -> RoostEncoder:
        """Return the Roost material encoder."""

        return cast(RoostEncoder, self.material_feature_extractor.material_encoder)

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


class RoostDKLModel(RoostGPModel):
    """Exact GP that jointly fine-tunes a Roost material descriptor.

    This model retains :class:`RoostGPModel`'s packed fraction/process Tensor
    contract, posterior API, process-only normalization, fusion, and latent
    projection. ``encoder_training="partial"`` trains the final
    ``trainable_encoder_layers`` Roost message-passing layers together with
    every crystal attention-pooling head in ``material_nn.cry_pool``. The
    element embedding, descriptor embedding, and earlier message-passing layers
    remain frozen. ``encoder_training="full"`` trains the element embedding and
    complete descriptor network. Aviary prediction heads are not part of the
    public :class:`RoostEncoder` representation backbone.

    Frozen modules remain in evaluation mode when :meth:`train` is called;
    only the selected partial modules enter training mode. Full fine-tuning
    follows the ordinary model train/eval mode. In both modes
    :func:`bochan.fit.deep.deepkernel.fit_deepkernel_mll` jointly optimizes the
    selected Roost parameters, material/process projection, exact GP, and
    likelihood.

    Args:
        train_X: ``[n, composition_dim + process_dim]`` training inputs.
        train_Y: Single-output targets with shape ``[n]`` or ``[n, 1]``.
        train_Yvar: Reserved for future fixed-noise support.
        element_ids: Atomic-number vocabulary matching the fraction columns.
        encoder: Optional :class:`RoostEncoder` or raw descriptor-only Roost
            backbone exposing ``elem_embedding`` and ``material_nn``.
        checkpoint: Optional upstream Aviary or adapter encoder checkpoint.
        encoder_output_dim: Required only when an injected raw backbone does
            not expose a positive integer ``output_dim``.
        latent_dim: Width of the projection passed to the GP kernel.
        fusion: Material/process fusion strategy.
        projection: Optional latent projection module.
        strict_checkpoint: Require a complete encoder checkpoint state.
        encoder_training: ``"partial"`` for late message-passing plus crystal
            attention-pooling fine-tuning, or ``"full"`` for the complete
            descriptor representation.
        trainable_encoder_layers: Positive number of final Roost
            message-passing layers selected in partial mode. Full mode trains
            all message-passing layers regardless of this value.
        likelihood: Optional GPyTorch likelihood.
        input_transform: ``"DEFAULT"`` normalizes process columns only.
        outcome_transform: Outcome transform forwarded to the Gaussian deep
            kernel wrapper.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        element_ids: Tensor,
        encoder: RoostEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        encoder_output_dim: int | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        strict_checkpoint: bool = True,
        encoder_training: Literal["partial", "full"] = "partial",
        trainable_encoder_layers: int = 1,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
    ) -> None:
        resolved_training = _validate_encoder_training(encoder_training)
        resolved_trainable_layers = _validate_trainable_encoder_layers(trainable_encoder_layers)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            element_ids=element_ids,
            encoder=encoder,
            checkpoint=checkpoint,
            encoder_output_dim=encoder_output_dim,
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
            encoder_training=resolved_training,
            trainable_encoder_layers=resolved_trainable_layers,
        )
        self._encoder_training = resolved_training
        self._trainable_encoder_layers = resolved_trainable_layers
        self.material_feature_extractor._configure_encoder_training(
            training_mode,
            trainable_modules,
        )

    @property
    def encoder_training(self) -> Literal["partial", "full"]:
        """Return the immutable high-level encoder fine-tuning policy."""

        return self._encoder_training

    @property
    def trainable_encoder_layers(self) -> int:
        """Return the configured partial message-passing depth."""

        return self._trainable_encoder_layers


__all__ = ["RoostDKLModel", "RoostGPModel"]
