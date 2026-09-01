"""CHGNet crystal representations with exact Gaussian processes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

from botorch.models.transforms.input import Normalize
from botorch.utils.transforms import normalize_indices
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import CHGNetEncoder, MaterialProcessFusion
from bochan.composition.encoders.chgnet import Checkpoint

from .deepkernel import InputTransformArg, OutcomeTransformArg
from .deepkernel_configurable import (
    DeepKernelGaussianGPModel,
    DeepKernelGaussianMixedGPModel,
)
from .material import EncoderTrainingMode
from .structure import (
    _resolve_structure_input_transform,
    _StructureGPFeatureExtractor,
    _validate_structure_bank,
    _validate_structure_model_inputs,
)

_TrainableEncoderLayers = int | Literal["all"]


def _validate_model_inputs(X: Tensor, *, num_structures: int, input_dim: int) -> None:
    """Validate structure-index and continuous-process columns."""

    _validate_structure_model_inputs(
        X,
        num_structures=num_structures,
        input_dim=input_dim,
        encoder_name="CHGNet",
    )


def _resolve_material_encoder(
    encoder: CHGNetEncoder | nn.Module | None,
    checkpoint: Checkpoint | None,
    *,
    model_name: str,
    output_dim: int | None,
    strict_checkpoint: bool,
) -> CHGNetEncoder:
    """Build or reuse a CHGNet material encoder and freeze it initially."""

    if isinstance(encoder, CHGNetEncoder):
        if checkpoint is not None:
            raise ValueError(
                "checkpoint must be omitted when encoder is already a CHGNetEncoder; "
                "load the checkpoint when constructing that encoder."
            )
        if output_dim is not None and output_dim != encoder.output_dim:
            raise ValueError(
                "encoder_output_dim does not match CHGNetEncoder.output_dim: "
                f"{output_dim} != {encoder.output_dim}."
            )
        material_encoder = encoder
    else:
        material_encoder = CHGNetEncoder(
            encoder=encoder,
            model_name=model_name,
            checkpoint=checkpoint,
            output_dim=output_dim,
            strict_checkpoint=strict_checkpoint,
        )

    for parameter in material_encoder.parameters():
        parameter.requires_grad_(False)
    material_encoder.eval()
    return material_encoder


def _validate_trainable_encoder_layers(
    trainable_encoder_layers: _TrainableEncoderLayers,
) -> _TrainableEncoderLayers:
    if trainable_encoder_layers == "all":
        return "all"
    if (
        isinstance(trainable_encoder_layers, bool)
        or not isinstance(trainable_encoder_layers, int)
        or trainable_encoder_layers <= 0
    ):
        raise ValueError("trainable_encoder_layers must be a positive integer or 'all'.")
    return trainable_encoder_layers


def _unique_parameters(modules: Sequence[nn.Module]) -> tuple[nn.Parameter, ...]:
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    return tuple(parameters)


def _atom_conv_layers(material_encoder: CHGNetEncoder) -> tuple[nn.Module, ...]:
    """Return ordered CHGNet atom-convolution blocks for partial fine-tuning."""

    layers = getattr(material_encoder.encoder, "atom_conv_layers", None)
    if isinstance(layers, (nn.ModuleList, list, tuple)):
        candidates = tuple(layers)
    else:
        return ()
    return tuple(layer for layer in candidates if isinstance(layer, nn.Module))


def _configure_dkl_encoder(
    material_encoder: CHGNetEncoder,
    trainable_encoder_layers: _TrainableEncoderLayers,
) -> tuple[EncoderTrainingMode, tuple[nn.Module, ...]]:
    """Unfreeze the requested CHGNet representation parameters."""

    for parameter in material_encoder.parameters():
        parameter.requires_grad_(False)

    if trainable_encoder_layers == "all":
        modules = material_encoder.backbone_modules()
        parameters = _unique_parameters(modules)
        if not parameters:
            raise ValueError("The CHGNet encoder exposes no backbone parameters to fine-tune.")
        for parameter in parameters:
            parameter.requires_grad_(True)
        return "full", ()

    layers = _atom_conv_layers(material_encoder)
    if not layers:
        raise ValueError(
            "Partial CHGNet fine-tuning requires encoder.atom_conv_layers. "
            "Use trainable_encoder_layers='all' for an injected encoder without those blocks."
        )
    if trainable_encoder_layers > len(layers):
        raise ValueError(
            "trainable_encoder_layers exceeds the number of CHGNet atom-convolution blocks: "
            f"{trainable_encoder_layers} > {len(layers)}."
        )
    trainable_modules = layers[-trainable_encoder_layers:]
    parameters = _unique_parameters(trainable_modules)
    if not parameters:
        raise ValueError(
            "The selected CHGNet atom-convolution blocks expose no parameters to fine-tune."
        )
    for parameter in parameters:
        parameter.requires_grad_(True)
    return "partial", trainable_modules


class _CHGNetGPFeatureExtractor(_StructureGPFeatureExtractor):
    """CHGNet specialization of the shared structure/process feature extractor."""

    def __init__(
        self,
        *,
        material_encoder: CHGNetEncoder,
        structures: Sequence[Any],
        process_dim: int,
        latent_dim: int,
        fusion: Literal["concat"] | MaterialProcessFusion,
        projection: nn.Module | None,
    ) -> None:
        validated_structures = _validate_structure_bank(
            structures,
            argument_name="structures",
        )
        super().__init__(
            material_encoder=material_encoder,
            structure_inputs=validated_structures,
            process_dim=process_dim,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
            structure_argument_name="structures",
            encoder_name="CHGNet",
        )
        self.structures = self.structure_inputs


def _resolve_input_transform(
    train_X: Tensor,
    *,
    input_transform: InputTransformArg,
) -> InputTransformArg:
    """Resolve DEFAULT to process-only normalization, preserving structure ids."""

    return _resolve_structure_input_transform(
        train_X,
        input_transform=input_transform,
    )


def _resolve_mixed_input_transform(
    train_X: Tensor,
    *,
    cat_dims: Sequence[int],
    input_transform: InputTransformArg,
) -> InputTransformArg:
    """Normalize numeric process columns while preserving selectors/categories."""

    if not isinstance(input_transform, str) or input_transform.upper() != "DEFAULT":
        return input_transform
    categorical = set(cat_dims)
    process_dims = [
        index
        for index in range(1, train_X.shape[-1])
        if index not in categorical
    ]
    if not process_dims:
        return None
    return Normalize(d=train_X.shape[-1], indices=process_dims)


class CHGNetGPModel(DeepKernelGaussianGPModel):
    """Exact GP over frozen CHGNet crystal representations.

    The first input column is an integer-valued index into ``structures``.
    Remaining columns are optional continuous process variables. Structure
    selection is discrete and should be optimized by enumeration or fixed
    features; continuous process variables retain acquisition gradients.
    """

    _supports_cache_root = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        encoder: CHGNetEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        model_name: str = "0.3.0",
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
            raise ValueError("train_X must have shape [n, 1 + process_dim].")
        if train_X.shape[-1] < 1:
            raise ValueError("train_X must contain a structure-index column.")
        model_class_name = self.__class__.__name__
        if train_Y.ndim > 1 and train_Y.shape[-1] != 1:
            raise ValueError(
                f"{model_class_name} currently supports single-output train_Y only."
            )
        if train_Yvar is not None:
            raise NotImplementedError(
                f"{model_class_name} does not yet support train_Yvar."
            )
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string.")

        validated_structures = _validate_structure_bank(
            structures,
            argument_name="structures",
        )
        _validate_model_inputs(
            train_X,
            num_structures=len(validated_structures),
            input_dim=train_X.shape[-1],
        )
        process_dim = train_X.shape[-1] - 1

        material_encoder = _resolve_material_encoder(
            encoder,
            checkpoint,
            model_name=model_name,
            output_dim=encoder_output_dim,
            strict_checkpoint=strict_checkpoint,
        )
        feature_extractor = _CHGNetGPFeatureExtractor(
            material_encoder=material_encoder,
            structures=validated_structures,
            process_dim=process_dim,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
        )
        resolved_input_transform = _resolve_input_transform(
            train_X,
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
        self.chgnet_feature_extractor.validate_input(transformed_train_X)

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> CHGNetGPModel:
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
        return cast(CHGNetGPModel, module)

    @property
    def chgnet_feature_extractor(self) -> _CHGNetGPFeatureExtractor:
        return cast(_CHGNetGPFeatureExtractor, self.deepkernel.feature_extractor)

    @property
    def material_encoder(self) -> CHGNetEncoder:
        return cast(CHGNetEncoder, self.chgnet_feature_extractor.material_encoder)

    @property
    def projection(self) -> nn.Module:
        return self.chgnet_feature_extractor.projection

    @property
    def fusion(self) -> MaterialProcessFusion:
        return self.chgnet_feature_extractor.fusion

    @property
    def structures(self) -> tuple[Any, ...]:
        return self.chgnet_feature_extractor.structures

    @property
    def num_structures(self) -> int:
        return self.chgnet_feature_extractor.num_structures

    @property
    def process_dim(self) -> int:
        return self.chgnet_feature_extractor.process_dim

    @property
    def structure_feature_cache_enabled(self) -> bool:
        return self.chgnet_feature_extractor.material_feature_cache_enabled

    def clear_structure_feature_cache(self) -> None:
        self.chgnet_feature_extractor.clear_material_feature_cache()


class CHGNetDKLModel(CHGNetGPModel):
    """Exact GP that jointly fine-tunes a CHGNet structure encoder.

    A positive ``trainable_encoder_layers`` value unfreezes that many final
    ``atom_conv_layers``. ``"all"`` fine-tunes all modules contributing to the
    CHGNet ``crystal_fea`` representation while leaving unrelated property-head
    modules outside the DKL training policy.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        encoder: CHGNetEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        model_name: str = "0.3.0",
        encoder_output_dim: int | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        strict_checkpoint: bool = True,
        trainable_encoder_layers: int | Literal["all"] = 1,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
    ) -> None:
        resolved_trainable_layers = _validate_trainable_encoder_layers(
            trainable_encoder_layers
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            structures=structures,
            encoder=encoder,
            checkpoint=checkpoint,
            model_name=model_name,
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
            resolved_trainable_layers,
        )
        self._trainable_encoder_layers = resolved_trainable_layers
        self.chgnet_feature_extractor._configure_encoder_training(
            training_mode,
            trainable_modules,
        )

    @property
    def trainable_encoder_layers(self) -> int | Literal["all"]:
        return self._trainable_encoder_layers


class CHGNetMixedGPModel(DeepKernelGaussianMixedGPModel):
    """Exact mixed GP over frozen CHGNet crystal representations.

    Column 0 is the discrete structure selector. ``cat_dims`` contains only
    integer-coded categorical process columns. The remaining process columns
    are numeric and are fused with the selected CHGNet crystal representation.
    """

    _supports_cache_root = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        encoder: CHGNetEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        model_name: str = "0.3.0",
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
        if train_X.shape[-1] < 2:
            raise ValueError(
                "CHGNetMixedGPModel requires a structure-index column and at least "
                "one categorical process column."
            )
        model_class_name = self.__class__.__name__
        if train_Y.ndim > 1 and train_Y.shape[-1] != 1:
            raise ValueError(
                f"{model_class_name} currently supports single-output train_Y only."
            )
        if train_Yvar is not None:
            raise NotImplementedError(
                f"{model_class_name} does not yet support train_Yvar."
            )
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string.")

        d = train_X.shape[-1]
        normalized_cat_dims = normalize_indices(indices=list(cat_dims), d=d)
        if not normalized_cat_dims:
            raise ValueError(
                "CHGNetMixedGPModel requires at least one categorical process dimension."
            )
        if 0 in normalized_cat_dims:
            raise ValueError(
                "The structure-index column (feature 0) is handled by CHGNet and "
                "cannot be included in cat_dims."
            )

        validated_structures = _validate_structure_bank(
            structures,
            argument_name="structures",
        )
        _validate_model_inputs(
            train_X,
            num_structures=len(validated_structures),
            input_dim=d,
        )

        continuous_dims = sorted(set(range(d)) - set(normalized_cat_dims))
        if not continuous_dims or continuous_dims[0] != 0:
            raise RuntimeError(
                "The CHGNet mixed continuous branch must retain structure-index feature 0."
            )
        process_dims = [index for index in continuous_dims if index != 0]

        material_encoder = _resolve_material_encoder(
            encoder,
            checkpoint,
            model_name=model_name,
            output_dim=encoder_output_dim,
            strict_checkpoint=strict_checkpoint,
        )
        feature_extractor = _CHGNetGPFeatureExtractor(
            material_encoder=material_encoder,
            structures=validated_structures,
            process_dim=len(process_dims),
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
        ).to(train_X)
        resolved_input_transform = _resolve_mixed_input_transform(
            train_X,
            cat_dims=normalized_cat_dims,
            input_transform=input_transform,
        )

        self._continuous_process_dims = tuple(process_dims)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=normalized_cat_dims,
            train_Yvar=None,
            likelihood=likelihood,
            input_transform=resolved_input_transform,
            outcome_transform=outcome_transform,
            feature_extractor=feature_extractor,
            latent_dim=latent_dim,
        )

        transformed_train_X = cast(tuple[Tensor, ...], self.deepkernel.train_inputs)[0]
        continuous_train_X = transformed_train_X[..., self.deepkernel.ord_dims]
        self.chgnet_feature_extractor.validate_input(continuous_train_X)

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> CHGNetMixedGPModel:
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
        return cast(CHGNetMixedGPModel, module)

    @property
    def chgnet_feature_extractor(self) -> _CHGNetGPFeatureExtractor:
        return cast(_CHGNetGPFeatureExtractor, self.deepkernel.feature_extractor)

    @property
    def material_encoder(self) -> CHGNetEncoder:
        return cast(CHGNetEncoder, self.chgnet_feature_extractor.material_encoder)

    @property
    def projection(self) -> nn.Module:
        return self.chgnet_feature_extractor.projection

    @property
    def fusion(self) -> MaterialProcessFusion:
        return self.chgnet_feature_extractor.fusion

    @property
    def structures(self) -> tuple[Any, ...]:
        return self.chgnet_feature_extractor.structures

    @property
    def num_structures(self) -> int:
        return self.chgnet_feature_extractor.num_structures

    @property
    def process_dim(self) -> int:
        return self.chgnet_feature_extractor.process_dim

    @property
    def continuous_process_dims(self) -> tuple[int, ...]:
        return self._continuous_process_dims

    @property
    def categorical_process_dim(self) -> int:
        return len(self.cat_dims)

    @property
    def structure_feature_cache_enabled(self) -> bool:
        return self.chgnet_feature_extractor.material_feature_cache_enabled

    def clear_structure_feature_cache(self) -> None:
        self.chgnet_feature_extractor.clear_material_feature_cache()


class CHGNetMixedDKLModel(CHGNetMixedGPModel):
    """Mixed exact GP that jointly fine-tunes the CHGNet structure encoder."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        encoder: CHGNetEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        model_name: str = "0.3.0",
        encoder_output_dim: int | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        strict_checkpoint: bool = True,
        trainable_encoder_layers: int | Literal["all"] = 1,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
    ) -> None:
        resolved_trainable_layers = _validate_trainable_encoder_layers(
            trainable_encoder_layers
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            train_Yvar=train_Yvar,
            structures=structures,
            encoder=encoder,
            checkpoint=checkpoint,
            model_name=model_name,
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
            resolved_trainable_layers,
        )
        self._trainable_encoder_layers = resolved_trainable_layers
        self.chgnet_feature_extractor._configure_encoder_training(
            training_mode,
            trainable_modules,
        )

    @property
    def trainable_encoder_layers(self) -> int | Literal["all"]:
        return self._trainable_encoder_layers


__all__ = [
    "CHGNetDKLModel",
    "CHGNetGPModel",
    "CHGNetMixedDKLModel",
    "CHGNetMixedGPModel",
]
