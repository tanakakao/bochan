"""MACE crystal representations with exact Gaussian processes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import MACEEncoder, MaterialProcessFusion

from .deepkernel import InputTransformArg, OutcomeTransformArg
from .deepkernel_configurable import DeepKernelGaussianGPModel
from .material import EncoderTrainingMode
from .structure import (
    _resolve_structure_input_transform,
    _StructureGPFeatureExtractor,
    _validate_structure_bank,
    _validate_structure_model_inputs,
)

_TrainableEncoderLayers = int | Literal["all"]
_Pooling = Literal["mean", "sum"]
_DEFAULT_MODEL_NAME = "medium-mpa-0"


def _validate_model_inputs(X: Tensor, *, num_structures: int, input_dim: int) -> None:
    """Validate structure-index and continuous-process columns."""

    _validate_structure_model_inputs(
        X,
        num_structures=num_structures,
        input_dim=input_dim,
        encoder_name="MACE",
    )


def _resolve_material_encoder(
    encoder: MACEEncoder | nn.Module | None,
    *,
    model_name: str,
    num_layers: int,
    pooling: _Pooling,
    head: str | None,
) -> MACEEncoder:
    """Build or reuse a MACE material encoder and freeze it initially."""

    if isinstance(encoder, MACEEncoder):
        material_encoder = encoder
    else:
        material_encoder = MACEEncoder(
            encoder=encoder,
            model_name=model_name,
            num_layers=num_layers,
            pooling=pooling,
            head=head,
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


def _interaction_product_pairs(
    material_encoder: MACEEncoder,
) -> tuple[tuple[nn.Module, nn.Module], ...]:
    """Return ordered MACE interaction/product block pairs."""

    interactions = getattr(material_encoder.encoder, "interactions", None)
    products = getattr(material_encoder.encoder, "products", None)
    if not isinstance(interactions, (nn.ModuleList, list, tuple)):
        return ()
    if not isinstance(products, (nn.ModuleList, list, tuple)):
        return ()
    interaction_modules = tuple(module for module in interactions if isinstance(module, nn.Module))
    product_modules = tuple(module for module in products if isinstance(module, nn.Module))
    if len(interaction_modules) != len(interactions) or len(product_modules) != len(products):
        return ()
    if not interaction_modules or len(interaction_modules) != len(product_modules):
        return ()
    return tuple(zip(interaction_modules, product_modules, strict=True))


def _configure_dkl_encoder(
    material_encoder: MACEEncoder,
    trainable_encoder_layers: _TrainableEncoderLayers,
) -> tuple[EncoderTrainingMode, tuple[nn.Module, ...]]:
    """Unfreeze the requested MACE representation backbone parameters."""

    for parameter in material_encoder.parameters():
        parameter.requires_grad_(False)

    if trainable_encoder_layers == "all":
        modules = material_encoder.backbone_modules()
        parameters = _unique_parameters(modules)
        if not parameters:
            raise ValueError("The MACE encoder exposes no backbone parameters to fine-tune.")
        for parameter in parameters:
            parameter.requires_grad_(True)
        return "full", ()

    pairs = _interaction_product_pairs(material_encoder)
    if not pairs:
        raise ValueError(
            "Partial MACE fine-tuning requires matching encoder.interactions and encoder.products. "
            "Use trainable_encoder_layers='all' for an injected encoder without those blocks."
        )
    if trainable_encoder_layers > len(pairs):
        raise ValueError(
            "trainable_encoder_layers exceeds the number of MACE interaction/product pairs: "
            f"{trainable_encoder_layers} > {len(pairs)}."
        )
    selected_pairs = pairs[-trainable_encoder_layers:]
    trainable_modules = tuple(module for pair in selected_pairs for module in pair)
    parameters = _unique_parameters(trainable_modules)
    if not parameters:
        raise ValueError("The selected MACE interaction/product blocks expose no parameters to fine-tune.")
    for parameter in parameters:
        parameter.requires_grad_(True)
    return "partial", trainable_modules


class _MACEGPFeatureExtractor(_StructureGPFeatureExtractor):
    """MACE specialization of the shared structure/process feature extractor."""

    def __init__(
        self,
        *,
        material_encoder: MACEEncoder,
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
            encoder_name="MACE",
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


class MACEGPModel(DeepKernelGaussianGPModel):
    """Exact GP over frozen invariant MACE crystal representations.

    The first input column is an integer-valued index into ``structures``.
    Remaining columns are optional continuous process variables. Frozen MACE
    representations are cached for the complete structure bank and reused by
    posterior and acquisition calls.
    """

    _supports_cache_root = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        encoder: MACEEncoder | nn.Module | None = None,
        model_name: str = _DEFAULT_MODEL_NAME,
        num_layers: int = -1,
        pooling: _Pooling = "mean",
        head: str | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
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
            raise ValueError(f"{model_class_name} currently supports single-output train_Y only.")
        if train_Yvar is not None:
            raise NotImplementedError(f"{model_class_name} does not yet support train_Yvar.")
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string.")
        if pooling not in {"mean", "sum"}:
            raise ValueError("pooling must be 'mean' or 'sum'.")

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
            model_name=model_name,
            num_layers=num_layers,
            pooling=pooling,
            head=head,
        )
        feature_extractor = _MACEGPFeatureExtractor(
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
        self.mace_feature_extractor.validate_input(transformed_train_X)

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> MACEGPModel:
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
        return cast(MACEGPModel, module)

    @property
    def mace_feature_extractor(self) -> _MACEGPFeatureExtractor:
        return cast(_MACEGPFeatureExtractor, self.deepkernel.feature_extractor)

    @property
    def material_encoder(self) -> MACEEncoder:
        return cast(MACEEncoder, self.mace_feature_extractor.material_encoder)

    @property
    def projection(self) -> nn.Module:
        return self.mace_feature_extractor.projection

    @property
    def fusion(self) -> MaterialProcessFusion:
        return self.mace_feature_extractor.fusion

    @property
    def structures(self) -> tuple[Any, ...]:
        return self.mace_feature_extractor.structures

    @property
    def num_structures(self) -> int:
        return self.mace_feature_extractor.num_structures

    @property
    def process_dim(self) -> int:
        return self.mace_feature_extractor.process_dim

    @property
    def structure_feature_cache_enabled(self) -> bool:
        return self.mace_feature_extractor.material_feature_cache_enabled

    def clear_structure_feature_cache(self) -> None:
        self.mace_feature_extractor.clear_material_feature_cache()


class MACEDKLModel(MACEGPModel):
    """Exact GP that jointly fine-tunes the MACE representation backbone.

    A positive ``trainable_encoder_layers`` value unfreezes that many final
    ``interaction[i]`` + ``product[i]`` pairs. ``"all"`` fine-tunes all modules
    contributing to bochan's invariant crystal representation. The original
    MACE energy ``readouts`` remain frozen in both modes.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        encoder: MACEEncoder | nn.Module | None = None,
        model_name: str = _DEFAULT_MODEL_NAME,
        num_layers: int = -1,
        pooling: _Pooling = "mean",
        head: str | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
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
            structures=structures,
            encoder=encoder,
            model_name=model_name,
            num_layers=num_layers,
            pooling=pooling,
            head=head,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        training_mode, trainable_modules = _configure_dkl_encoder(
            self.material_encoder,
            resolved_trainable_layers,
        )
        self._trainable_encoder_layers = resolved_trainable_layers
        self.mace_feature_extractor._configure_encoder_training(
            training_mode,
            trainable_modules,
        )

    @property
    def trainable_encoder_layers(self) -> int | Literal["all"]:
        return self._trainable_encoder_layers


__all__ = ["MACEDKLModel", "MACEGPModel"]
