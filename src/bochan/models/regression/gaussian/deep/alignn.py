"""ALIGNN crystal representations with exact Gaussian-process surrogates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import torch
from botorch.models.transforms.input import Normalize
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import (
    ALIGNNEncoder,
    MaterialProcessFusion,
    build_material_process_fusion,
)
from bochan.composition.encoders.alignn import Checkpoint

from .deepkernel import InputTransformArg, OutcomeTransformArg
from .deepkernel_configurable import DeepKernelGaussianGPModel

_EncoderTrainingMode = Literal["frozen", "partial", "full"]
_TrainableEncoderLayers = int | Literal["all"]


def _validate_structure_catalog(structure_catalog: Sequence[Any]) -> tuple[Any, ...]:
    if isinstance(structure_catalog, (str, bytes)):
        raise TypeError("structure_catalog must be a sequence of structure inputs, not text.")
    catalog = tuple(structure_catalog)
    if not catalog:
        raise ValueError("structure_catalog must contain at least one crystal structure.")
    return catalog


def _validate_model_inputs(X: Tensor, *, n_structures: int, input_dim: int) -> None:
    """Validate structure-index and continuous-process columns."""

    if not torch.is_tensor(X):
        raise TypeError("X must be a Tensor.")
    if X.ndim == 0 or X.shape[-1] != input_dim:
        raise ValueError(
            f"X width must equal 1 + process_dim: {X.shape[-1] if X.ndim else 0} != {input_dim}."
        )
    if not X.is_floating_point():
        raise TypeError("X must have a floating-point dtype.")
    if not torch.isfinite(X).all():
        raise ValueError("X must contain only finite values.")

    structure_ids = X[..., 0]
    rounded = structure_ids.round()
    if not torch.allclose(structure_ids, rounded, rtol=0.0, atol=1e-7):
        raise ValueError("The first ALIGNN input column must contain integer-valued structure ids.")
    if (rounded < 0).any() or (rounded >= n_structures).any():
        raise ValueError(
            "ALIGNN structure ids must index structure_catalog in the interval "
            f"[0, {n_structures - 1}]."
        )


def _resolve_material_encoder(
    encoder: ALIGNNEncoder | nn.Module | None,
    checkpoint: Checkpoint | None,
    *,
    config: dict[str, object] | None,
    output_dim: int | None,
    strict_checkpoint: bool,
) -> ALIGNNEncoder:
    """Build or reuse an ALIGNN adapter and apply an optional checkpoint."""

    if isinstance(encoder, ALIGNNEncoder):
        material_encoder = encoder
        if config is not None or output_dim is not None:
            raise ValueError("config/output_dim must be configured on an injected ALIGNNEncoder itself.")
        if checkpoint is not None:
            material_encoder.load_checkpoint(checkpoint, strict=strict_checkpoint)
    else:
        material_encoder = ALIGNNEncoder(
            encoder=encoder,
            checkpoint=checkpoint,
            config=config,
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


def _alignn_message_passing_layers(material_encoder: ALIGNNEncoder) -> tuple[nn.Module, ...]:
    """Return upstream ALIGNN message-passing layers in forward order."""

    encoder = material_encoder.encoder
    modules: list[nn.Module] = []
    for attribute in ("alignn_layers", "gcn_layers"):
        layers = getattr(encoder, attribute, None)
        if layers is None:
            continue
        if not isinstance(layers, (nn.ModuleList, nn.Sequential)):
            raise ValueError(f"encoder.{attribute} must be a torch.nn.ModuleList or torch.nn.Sequential.")
        modules.extend(layers)
    if not modules:
        raise ValueError(
            "Partial ALIGNN unfreezing requires encoder.alignn_layers and/or encoder.gcn_layers. "
            "Use trainable_encoder_layers='all' for a custom injected encoder."
        )
    return tuple(modules)


def _configure_dkl_encoder(
    material_encoder: ALIGNNEncoder,
    trainable_encoder_layers: _TrainableEncoderLayers,
) -> tuple[_EncoderTrainingMode, tuple[nn.Module, ...]]:
    """Unfreeze the requested final ALIGNN message-passing layers."""

    for parameter in material_encoder.parameters():
        parameter.requires_grad_(False)

    if trainable_encoder_layers == "all":
        parameters = tuple(material_encoder.parameters())
        if not parameters:
            raise ValueError("The ALIGNN encoder exposes no parameters to fine-tune.")
        for parameter in parameters:
            parameter.requires_grad_(True)
        return "full", ()

    layers = _alignn_message_passing_layers(material_encoder)
    if trainable_encoder_layers > len(layers):
        raise ValueError(
            "trainable_encoder_layers exceeds the number of ALIGNN message-passing layers: "
            f"{trainable_encoder_layers} > {len(layers)}."
        )
    trainable_modules = layers[-trainable_encoder_layers:]
    parameters = tuple(parameter for module in trainable_modules for parameter in module.parameters())
    if not parameters:
        raise ValueError("The selected ALIGNN layers expose no parameters to fine-tune.")
    for parameter in parameters:
        parameter.requires_grad_(True)
    return "partial", trainable_modules


class _ALIGNNGPFeatureExtractor(nn.Module):
    """Map discrete structure ids and process variables into a GP latent space."""

    def __init__(
        self,
        *,
        material_encoder: ALIGNNEncoder,
        structure_catalog: Sequence[Any],
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
        self.structure_catalog = _validate_structure_catalog(structure_catalog)
        self.n_structures = len(self.structure_catalog)
        self.structure_dim = 1
        self.process_dim = int(process_dim)
        self.input_dim = 1 + self.process_dim
        self._encoder_training_mode: _EncoderTrainingMode = "frozen"
        self._trainable_encoder_modules: tuple[nn.Module, ...] = ()
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

    def train(self, mode: bool = True) -> _ALIGNNGPFeatureExtractor:
        super().train(mode)
        if self._encoder_training_mode == "frozen":
            self.material_encoder.eval()
        elif self._encoder_training_mode == "partial":
            self.material_encoder.eval()
            for module in self._trainable_encoder_modules:
                module.train(mode)
        return self

    def _configure_encoder_training(
        self,
        mode: _EncoderTrainingMode,
        trainable_modules: tuple[nn.Module, ...] = (),
    ) -> None:
        if mode == "partial" and not trainable_modules:
            raise ValueError("Partial encoder training requires at least one trainable module.")
        if mode != "partial" and trainable_modules:
            raise ValueError("Trainable encoder modules are valid only for partial training.")
        self._encoder_training_mode = mode
        self._trainable_encoder_modules = trainable_modules
        self.train(self.training)

    def validate_input(self, X: Tensor) -> None:
        _validate_model_inputs(X, n_structures=self.n_structures, input_dim=self.input_dim)

    def _encode_structure_ids(self, structure_ids: Tensor) -> Tensor:
        leading_shape = structure_ids.shape
        flat_ids = structure_ids.reshape(-1).round().to(dtype=torch.long)
        unique_ids, inverse = torch.unique(flat_ids, sorted=True, return_inverse=True)
        encoded = [
            self.material_encoder(self.structure_catalog[int(index)])
            for index in unique_ids.detach().cpu().tolist()
        ]
        unique_features = torch.stack(encoded, dim=0)
        flat_features = unique_features.index_select(0, inverse.to(device=unique_features.device))
        return flat_features.reshape(*leading_shape, self.material_encoder.output_dim)

    def forward(self, X: Tensor) -> Tensor:
        """Return projected ALIGNN/process features for the exact GP kernel."""

        self.validate_input(X)
        material_features = self._encode_structure_ids(X[..., 0])
        if material_features.device != X.device or material_features.dtype != X.dtype:
            raise ValueError("ALIGNN material features must match X's device and dtype.")
        process_features = X[..., 1:] if self.process_dim else None
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
            raise FloatingPointError("ALIGNN material/process projection produced non-finite values.")
        return projected_features


def _resolve_input_transform(train_X: Tensor, input_transform: InputTransformArg) -> InputTransformArg:
    """Resolve DEFAULT to process-only normalization, preserving structure ids."""

    if not isinstance(input_transform, str) or input_transform.upper() != "DEFAULT":
        return input_transform
    process_dims = list(range(1, train_X.shape[-1]))
    if not process_dims:
        return None
    return Normalize(d=train_X.shape[-1], indices=process_dims)


class ALIGNNGPModel(DeepKernelGaussianGPModel):
    """Exact GP over frozen ALIGNN crystal-structure representations.

    The low-level BoTorch tensor contract is ``[structure_id, process...]``.
    ``structure_id`` is an integer-valued index into ``structure_catalog`` and
    is intentionally discrete. Candidate optimization must therefore enumerate
    or fix structure ids while optimizing any process columns continuously.

    ``structure_catalog`` contains the actual crystal graph inputs accepted by
    :class:`ALIGNNEncoder`. The tabular/API layer can later map CIF/POSCAR/JARVIS
    structures onto this catalog without changing the GP model contract.
    """

    _supports_cache_root = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structure_catalog: Sequence[Any],
        encoder: ALIGNNEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        alignn_config: dict[str, object] | None = None,
        encoder_output_dim: int | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        strict_checkpoint: bool = True,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
    ) -> None:
        if train_X.ndim != 2 or train_X.shape[-1] < 1:
            raise ValueError("train_X must have shape [n, 1 + process_dim].")
        model_name = self.__class__.__name__
        if train_Y.ndim > 1 and train_Y.shape[-1] != 1:
            raise ValueError(f"{model_name} currently supports single-output train_Y only.")
        if train_Yvar is not None:
            raise NotImplementedError(f"{model_name} does not yet support train_Yvar.")
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")

        catalog = _validate_structure_catalog(structure_catalog)
        _validate_model_inputs(train_X, n_structures=len(catalog), input_dim=train_X.shape[-1])
        process_dim = train_X.shape[-1] - 1
        material_encoder = _resolve_material_encoder(
            encoder,
            checkpoint,
            config=alignn_config,
            output_dim=encoder_output_dim,
            strict_checkpoint=strict_checkpoint,
        )
        feature_extractor = _ALIGNNGPFeatureExtractor(
            material_encoder=material_encoder,
            structure_catalog=catalog,
            process_dim=process_dim,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
        )
        resolved_input_transform = _resolve_input_transform(train_X, input_transform)

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
    def structure_catalog(self) -> tuple[Any, ...]:
        return self.alignn_feature_extractor.structure_catalog

    @property
    def n_structures(self) -> int:
        return self.alignn_feature_extractor.n_structures

    @property
    def process_dim(self) -> int:
        return self.alignn_feature_extractor.process_dim


class ALIGNNDKLModel(ALIGNNGPModel):
    """Exact GP that jointly fine-tunes final ALIGNN message-passing layers."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structure_catalog: Sequence[Any],
        encoder: ALIGNNEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        alignn_config: dict[str, object] | None = None,
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
        resolved_trainable_layers = _validate_trainable_encoder_layers(trainable_encoder_layers)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            structure_catalog=structure_catalog,
            encoder=encoder,
            checkpoint=checkpoint,
            alignn_config=alignn_config,
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
        self.alignn_feature_extractor._configure_encoder_training(training_mode, trainable_modules)

    @property
    def trainable_encoder_layers(self) -> int | Literal["all"]:
        return self._trainable_encoder_layers


__all__ = ["ALIGNNDKLModel", "ALIGNNGPModel"]
