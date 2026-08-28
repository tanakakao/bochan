"""ALIGNN crystal representations with exact Gaussian processes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import torch
from botorch.models.transforms.input import Normalize
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import ALIGNNEncoder, MaterialProcessFusion
from bochan.composition.encoders.alignn import Checkpoint

from .deepkernel import InputTransformArg, OutcomeTransformArg
from .deepkernel_configurable import DeepKernelGaussianGPModel
from .material import EncoderTrainingMode, _BaseMaterialGPFeatureExtractor

_TrainableEncoderLayers = int | Literal["all"]


def _validate_structure_bank(structure_graphs: Sequence[Any]) -> tuple[Any, ...]:
    if not isinstance(structure_graphs, Sequence) or isinstance(structure_graphs, (str, bytes)):
        raise TypeError("structure_graphs must be a sequence.")
    if not structure_graphs:
        raise ValueError("structure_graphs must contain at least one structure graph.")
    return tuple(structure_graphs)


def _validate_model_inputs(X: Tensor, *, num_structures: int, input_dim: int) -> None:
    """Validate structure-index and continuous-process columns."""

    if not torch.is_tensor(X):
        raise TypeError("X must be a Tensor.")
    if X.ndim == 0 or X.shape[-1] != input_dim:
        raise ValueError(f"X width must equal input_dim: {X.shape[-1] if X.ndim else 0} != {input_dim}.")
    if not X.is_floating_point():
        raise TypeError("X must have a floating-point dtype.")
    if not torch.isfinite(X).all():
        raise ValueError("X must contain only finite values.")

    indices = X[..., 0]
    rounded = indices.round()
    if not torch.allclose(indices, rounded, rtol=0.0, atol=1e-6):
        raise ValueError("The first ALIGNN input column must contain integer-valued structure indices.")
    if (rounded < 0).any() or (rounded >= num_structures).any():
        raise ValueError(f"Structure indices must be in [0, {num_structures - 1}].")


def _resolve_material_encoder(
    encoder: ALIGNNEncoder | nn.Module | None,
    checkpoint: Checkpoint | None,
    *,
    output_dim: int | None,
    config: object | None,
    strict_checkpoint: bool,
) -> ALIGNNEncoder:
    """Build or reuse an ALIGNN adapter and apply an optional checkpoint."""

    if isinstance(encoder, ALIGNNEncoder):
        material_encoder = encoder
        if output_dim is not None and output_dim != material_encoder.output_dim:
            raise ValueError(
                f"output_dim does not match ALIGNNEncoder.output_dim: {output_dim} != {material_encoder.output_dim}."
            )
        if config is not None:
            raise ValueError("config must be omitted when encoder is already an ALIGNNEncoder.")
        if checkpoint is not None:
            material_encoder.load_checkpoint(checkpoint, strict=strict_checkpoint)
    else:
        material_encoder = ALIGNNEncoder(
            encoder=encoder,
            checkpoint=checkpoint,
            output_dim=output_dim,
            config=config,
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


def _configure_dkl_encoder(
    material_encoder: ALIGNNEncoder,
    trainable_encoder_layers: _TrainableEncoderLayers,
) -> tuple[EncoderTrainingMode, tuple[nn.Module, ...]]:
    """Unfreeze the requested ALIGNN backbone parameters."""

    for parameter in material_encoder.parameters():
        parameter.requires_grad_(False)

    if trainable_encoder_layers == "all":
        modules = material_encoder.backbone_modules()
        parameters = _unique_parameters(modules)
        if not parameters:
            raise ValueError("The ALIGNN encoder exposes no backbone parameters to fine-tune.")
        for parameter in parameters:
            parameter.requires_grad_(True)
        return "full", ()

    layers = material_encoder.graph_conv_layers()
    if not layers:
        raise ValueError(
            "Partial ALIGNN fine-tuning requires encoder.alignn_layers and/or encoder.gcn_layers. "
            "Use trainable_encoder_layers='all' for an injected encoder without those blocks."
        )
    if trainable_encoder_layers > len(layers):
        raise ValueError(
            "trainable_encoder_layers exceeds the number of ALIGNN/GCN graph-convolution blocks: "
            f"{trainable_encoder_layers} > {len(layers)}."
        )
    trainable_modules = layers[-trainable_encoder_layers:]
    parameters = _unique_parameters(trainable_modules)
    if not parameters:
        raise ValueError("The selected ALIGNN graph-convolution blocks expose no parameters to fine-tune.")
    for parameter in parameters:
        parameter.requires_grad_(True)
    return "partial", trainable_modules


class _ALIGNNGPFeatureExtractor(_BaseMaterialGPFeatureExtractor):
    """Map structure-bank indices and process features to a GP latent space."""

    def __init__(
        self,
        *,
        material_encoder: ALIGNNEncoder,
        structure_graphs: Sequence[Any],
        process_dim: int,
        latent_dim: int,
        fusion: Literal["concat"] | MaterialProcessFusion,
        projection: nn.Module | None,
    ) -> None:
        validated_graphs = _validate_structure_bank(structure_graphs)
        super().__init__(
            material_encoder=material_encoder,
            input_dim=1 + process_dim,
            process_dim=process_dim,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
        )
        self.structure_graphs = validated_graphs
        self.num_structures = len(self.structure_graphs)
        self._material_feature_cache_versions: tuple[int, ...] | None = None
        self.register_buffer("_material_feature_cache", None, persistent=False)

    def __getstate__(self) -> dict[str, Any]:
        """Exclude derived structure-feature cache state from pickle artifacts."""

        state = dict(super().__getstate__())
        buffers = dict(state.get("_buffers", {}))
        buffers["_material_feature_cache"] = None
        state["_buffers"] = buffers
        state["_material_feature_cache_versions"] = None
        return state

    @property
    def material_feature_cache_enabled(self) -> bool:
        """Return whether structure-only encoder outputs can be safely reused."""

        return self._encoder_training_mode == "frozen" and not any(
            parameter.requires_grad for parameter in self.material_encoder.parameters()
        )

    @property
    def material_feature_cache(self) -> Tensor | None:
        """Return the non-persistent frozen structure-feature bank, if materialized."""

        return self._material_feature_cache

    def _encoder_parameter_versions(self) -> tuple[int, ...]:
        """Return mutation counters used to invalidate stale frozen embeddings."""

        return tuple(
            int(getattr(parameter, "_version", 0))
            for parameter in self.material_encoder.parameters()
        )

    def clear_material_feature_cache(self) -> None:
        """Discard cached structure representations."""

        self._material_feature_cache = None
        self._material_feature_cache_versions = None

    def _cached_material_features(self, X: Tensor) -> Tensor | None:
        if not self.material_feature_cache_enabled:
            return None
        cache = self._material_feature_cache
        encoder_versions = self._encoder_parameter_versions()
        if (
            cache is None
            or cache.device != X.device
            or cache.dtype != X.dtype
            or self._material_feature_cache_versions != encoder_versions
        ):
            with torch.no_grad():
                cache = self.material_encoder(list(self.structure_graphs)).detach()
            expected = (self.num_structures, int(self.material_encoder.output_dim))
            if tuple(cache.shape) != expected:
                raise ValueError(
                    "ALIGNN encoder must return one cached vector per structure: "
                    f"{tuple(cache.shape)} != {expected}."
                )
            if cache.device != X.device or cache.dtype != X.dtype:
                raise ValueError("Cached ALIGNN structure features must match X's device and dtype.")
            if not torch.isfinite(cache).all():
                raise FloatingPointError("Cached ALIGNN structure features contain non-finite values.")
            self._material_feature_cache = cache
            self._material_feature_cache_versions = self._encoder_parameter_versions()
        return cache

    def _on_encoder_training_policy_change(self) -> None:
        """Discard frozen features before an encoder policy change."""

        self.clear_material_feature_cache()

    def validate_input(self, X: Tensor) -> None:
        _validate_model_inputs(
            X,
            num_structures=self.num_structures,
            input_dim=self.input_dim,
        )

    def _material_features(self, X: Tensor) -> Tensor:
        """Return structure features with the leading input shape restored."""

        leading_shape = X.shape[:-1]
        flat_X = X.reshape(-1, self.input_dim)
        structure_index_tensor = flat_X[:, 0].round().to(dtype=torch.long)
        cached_bank = self._cached_material_features(X)
        if cached_bank is None:
            structure_indices = structure_index_tensor.detach().cpu().tolist()
            graphs = [self.structure_graphs[index] for index in structure_indices]
            material_features = self.material_encoder(graphs)
        else:
            material_features = cached_bank.index_select(
                0,
                structure_index_tensor.to(device=cached_bank.device),
            )
        if material_features.device != X.device or material_features.dtype != X.dtype:
            raise ValueError("ALIGNN structure features must match X's device and dtype.")
        return material_features.reshape(
            *leading_shape,
            int(self.material_encoder.output_dim),
        )

    def _process_features(self, X: Tensor) -> Tensor | None:
        """Return process columns aligned with the structure representation."""

        if not self.process_dim:
            return None
        return X[..., 1:]

    def forward(self, X: Tensor) -> Tensor:
        """Flatten candidate rows for custom modules, then restore leading dimensions."""

        self.validate_input(X)
        leading_shape = X.shape[:-1]
        flat_X = X.reshape(-1, self.input_dim)
        flat_features = super().forward(flat_X)
        return flat_features.reshape(*leading_shape, self.output_dim)


def _resolve_input_transform(
    train_X: Tensor,
    *,
    input_transform: InputTransformArg,
) -> InputTransformArg:
    """Resolve DEFAULT to process-only normalization, preserving structure ids."""

    if not isinstance(input_transform, str) or input_transform.upper() != "DEFAULT":
        return input_transform
    process_dims = list(range(1, train_X.shape[-1]))
    if not process_dims:
        return None
    return Normalize(d=train_X.shape[-1], indices=process_dims)


class ALIGNNGPModel(DeepKernelGaussianGPModel):
    """Exact GP over frozen ALIGNN crystal-structure representations.

    The first input column is an integer-valued index into ``structure_graphs``.
    Remaining columns are optional continuous process variables. The structure
    index is deliberately not differentiated; optimize it by enumeration or
    ``optimize_acqf_mixed`` fixed-feature configurations while continuous
    process variables retain acquisition gradients.
    """

    _supports_cache_root = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structure_graphs: Sequence[Any],
        encoder: ALIGNNEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        encoder_output_dim: int | None = None,
        encoder_config: object | None = None,
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
        model_name = self.__class__.__name__
        if train_Y.ndim > 1 and train_Y.shape[-1] != 1:
            raise ValueError(f"{model_name} currently supports single-output train_Y only.")
        if train_Yvar is not None:
            raise NotImplementedError(f"{model_name} does not yet support train_Yvar.")
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")

        validated_graphs = _validate_structure_bank(structure_graphs)
        _validate_model_inputs(
            train_X,
            num_structures=len(validated_graphs),
            input_dim=train_X.shape[-1],
        )
        process_dim = train_X.shape[-1] - 1

        material_encoder = _resolve_material_encoder(
            encoder,
            checkpoint,
            output_dim=encoder_output_dim,
            config=encoder_config,
            strict_checkpoint=strict_checkpoint,
        )
        feature_extractor = _ALIGNNGPFeatureExtractor(
            material_encoder=material_encoder,
            structure_graphs=validated_graphs,
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
        self.alignn_feature_extractor.validate_input(transformed_train_X)

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> ALIGNNGPModel:
        module = super()._apply(fn, recurse=recurse)
        reference = next(
            (value for value in (*self.parameters(), *self.buffers()) if value.is_floating_point()),
            None,
        )
        if reference is not None:
            self._model_dtype = reference.dtype
            self._model_device = reference.device
        return cast(ALIGNNGPModel, module)

    @property
    def alignn_feature_extractor(self) -> _ALIGNNGPFeatureExtractor:
        return cast(_ALIGNNGPFeatureExtractor, self.deepkernel.feature_extractor)

    @property
    def material_encoder(self) -> ALIGNNEncoder:
        return self.alignn_feature_extractor.material_encoder

    @property
    def projection(self) -> nn.Module:
        return self.alignn_feature_extractor.projection

    @property
    def fusion(self) -> MaterialProcessFusion:
        return self.alignn_feature_extractor.fusion

    @property
    def structure_graphs(self) -> tuple[Any, ...]:
        return self.alignn_feature_extractor.structure_graphs

    @property
    def num_structures(self) -> int:
        return self.alignn_feature_extractor.num_structures

    @property
    def process_dim(self) -> int:
        return self.alignn_feature_extractor.process_dim

    @property
    def structure_feature_cache_enabled(self) -> bool:
        return self.alignn_feature_extractor.material_feature_cache_enabled

    def clear_structure_feature_cache(self) -> None:
        self.alignn_feature_extractor.clear_material_feature_cache()


class ALIGNNDKLModel(ALIGNNGPModel):
    """Exact GP that jointly fine-tunes an ALIGNN structure encoder.

    A positive ``trainable_encoder_layers`` value unfreezes that many final
    graph-convolution blocks from the ordered ALIGNN + GCN stacks. ``"all"``
    fine-tunes the complete representation backbone while keeping an upstream
    property head outside the DKL path.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structure_graphs: Sequence[Any],
        encoder: ALIGNNEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        encoder_output_dim: int | None = None,
        encoder_config: object | None = None,
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
            structure_graphs=structure_graphs,
            encoder=encoder,
            checkpoint=checkpoint,
            encoder_output_dim=encoder_output_dim,
            encoder_config=encoder_config,
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
        self.alignn_feature_extractor._configure_encoder_training(
            training_mode,
            trainable_modules,
        )

    @property
    def trainable_encoder_layers(self) -> int | Literal["all"]:
        return self._trainable_encoder_layers


__all__ = ["ALIGNNDKLModel", "ALIGNNGPModel"]
