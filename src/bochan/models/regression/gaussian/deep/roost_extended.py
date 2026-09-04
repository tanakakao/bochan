"""Roost mixed-input and correlated multi-output Gaussian surrogates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, cast

from botorch.utils.transforms import normalize_indices
from gpytorch.kernels import MultitaskKernel
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import MaterialProcessFusion, RoostEncoder
from bochan.composition.encoders.roost import Checkpoint

from .deepkernel import InputTransformArg, OutcomeTransformArg
from .deepkernel_configurable import (
    DeepKernelGaussianGPModel,
    DeepKernelGaussianMixedGPModel,
)
from .material import (
    CompositionMaterialInputTransform,
    MaterialGPFeatureExtractor,
    _resolve_composition_input_transform,
    _validate_composition_element_ids,
    _validate_composition_model_inputs,
)
from .roost import (
    _configure_dkl_encoder,
    _resolve_material_encoder,
    _validate_encoder_training,
    _validate_trainable_encoder_layers,
)


class _RoostMixedContinuousFeatureExtractor(nn.Module):
    """Encode composition plus numeric process features for a mixed Roost GP."""

    def __init__(
        self,
        *,
        continuous_input_dim: int,
        composition_indices: Sequence[int],
        element_ids: Tensor,
        method: str,
        reference_index: int | None,
        process_bounds: Tensor | None,
        component_weights: Tensor | None,
        normalize_process: bool,
        material_encoder: RoostEncoder,
        latent_dim: int,
        fusion: Literal["concat"] | MaterialProcessFusion,
        projection: nn.Module | None,
    ) -> None:
        super().__init__()
        validated_element_ids = _validate_composition_element_ids(element_ids)
        self.packer = CompositionMaterialInputTransform(
            input_dim=continuous_input_dim,
            composition_indices=composition_indices,
            n_components=int(validated_element_ids.numel()),
            method=method,
            reference_index=reference_index,
            process_bounds=process_bounds,
            component_weights=component_weights,
            normalize_process=normalize_process,
        )
        self.roost = MaterialGPFeatureExtractor(
            material_encoder=material_encoder,
            element_ids=validated_element_ids,
            process_dim=self.packer.process_dim,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
        )
        self.output_dim = int(latent_dim)

    def forward(self, X: Tensor) -> Tensor:
        return self.roost(self.packer(X))


def _validate_wide_targets(train_X: Tensor, train_Y: Tensor, *, model_name: str) -> None:
    if train_X.ndim != 2:
        raise ValueError("train_X must have shape [n, d].")
    if train_Y.ndim != 2 or train_Y.shape[-1] < 2:
        raise ValueError(
            f"{model_name} requires wide train_Y with shape [n, m] and at least two outputs."
        )


class _CorrelatedRoostMixin:
    deepkernel: object
    num_outputs: int

    def _validate_correlated_kernel(self) -> None:
        covar_module = getattr(self.deepkernel, "covar_module", None)
        if not isinstance(covar_module, MultitaskKernel):
            raise RuntimeError(
                f"{self.__class__.__name__} requires a correlated MultitaskKernel."
            )

    @property
    def num_tasks(self) -> int:
        return int(self.num_outputs)

    @property
    def task_covar_module(self) -> nn.Module:
        self._validate_correlated_kernel()
        return cast(MultitaskKernel, self.deepkernel.covar_module).task_covar_module

    @property
    def task_kernel(self) -> nn.Module:
        return self.task_covar_module


class _RoostModelProperties:
    deepkernel: object

    @property
    def material_feature_extractor(self) -> MaterialGPFeatureExtractor:
        feature_extractor = getattr(self.deepkernel, "feature_extractor")
        if isinstance(feature_extractor, _RoostMixedContinuousFeatureExtractor):
            return feature_extractor.roost
        return cast(MaterialGPFeatureExtractor, feature_extractor)

    @property
    def material_encoder(self) -> RoostEncoder:
        return cast(RoostEncoder, self.material_feature_extractor.material_encoder)

    @property
    def projection(self) -> nn.Module:
        return self.material_feature_extractor.projection

    @property
    def fusion(self) -> MaterialProcessFusion:
        return self.material_feature_extractor.fusion

    @property
    def element_ids(self) -> Tensor:
        return self.material_feature_extractor.element_ids

    @property
    def composition_dim(self) -> int:
        return self.material_feature_extractor.composition_dim

    @property
    def process_dim(self) -> int:
        return self.material_feature_extractor.process_dim


def _build_mixed_roost(
    *,
    train_X: Tensor,
    cat_dims: Sequence[int],
    element_ids: Tensor,
    composition_indices: Sequence[int],
    method: str,
    reference_index: int | None,
    process_bounds: Tensor | None,
    component_weights: Tensor | None,
    normalize_process: bool,
    encoder: RoostEncoder | nn.Module | None,
    checkpoint: Checkpoint | None,
    encoder_output_dim: int | None,
    latent_dim: int,
    fusion: Literal["concat"] | MaterialProcessFusion,
    projection: nn.Module | None,
    strict_checkpoint: bool,
) -> tuple[list[int], _RoostMixedContinuousFeatureExtractor]:
    d = train_X.shape[-1]
    normalized_cat_dims = normalize_indices(indices=list(cat_dims), d=d)
    if not normalized_cat_dims:
        raise ValueError("Roost mixed models require at least one categorical process dimension.")
    continuous_dims = sorted(set(range(d)) - set(normalized_cat_dims))
    raw_composition_indices = [int(index) for index in composition_indices]
    if not raw_composition_indices:
        raise ValueError("composition_indices must not be empty.")
    if any(index in normalized_cat_dims for index in raw_composition_indices):
        raise ValueError("composition_indices must refer only to continuous columns.")
    if min(raw_composition_indices) < 0 or max(raw_composition_indices) >= d:
        raise ValueError("composition_indices must be valid train_X columns.")
    continuous_position = {raw: idx for idx, raw in enumerate(continuous_dims)}
    continuous_composition_indices = [
        continuous_position[index] for index in raw_composition_indices
    ]
    validated_element_ids = _validate_composition_element_ids(element_ids)
    material_encoder = _resolve_material_encoder(
        encoder,
        checkpoint,
        output_dim=encoder_output_dim,
        strict_checkpoint=strict_checkpoint,
    )
    feature_extractor = _RoostMixedContinuousFeatureExtractor(
        continuous_input_dim=len(continuous_dims),
        composition_indices=continuous_composition_indices,
        element_ids=validated_element_ids,
        method=method,
        reference_index=reference_index,
        process_bounds=process_bounds,
        component_weights=component_weights,
        normalize_process=normalize_process,
        material_encoder=material_encoder,
        latent_dim=latent_dim,
        fusion=fusion,
        projection=projection,
    ).to(train_X)
    return normalized_cat_dims, feature_extractor


class RoostMixedGPModel(_RoostModelProperties, DeepKernelGaussianMixedGPModel):
    """Exact GP over Roost representations with categorical process variables."""

    _supports_cache_root = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Tensor | None = None,
        *,
        element_ids: Tensor,
        composition_indices: Sequence[int],
        method: str = "ilr",
        reference_index: int | None = None,
        process_bounds: Tensor | None = None,
        component_weights: Tensor | None = None,
        normalize_process: bool = True,
        encoder: RoostEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        encoder_output_dim: int | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        strict_checkpoint: bool = True,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransformArg = "DEFAULT",
    ) -> None:
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape [n, d].")
        if train_Y.ndim > 1 and train_Y.shape[-1] != 1:
            raise ValueError("RoostMixedGPModel currently supports single-output train_Y only.")
        normalized_cat_dims, feature_extractor = _build_mixed_roost(
            train_X=train_X,
            cat_dims=cat_dims,
            element_ids=element_ids,
            composition_indices=composition_indices,
            method=method,
            reference_index=reference_index,
            process_bounds=process_bounds,
            component_weights=component_weights,
            normalize_process=normalize_process,
            encoder=encoder,
            checkpoint=checkpoint,
            encoder_output_dim=encoder_output_dim,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
            strict_checkpoint=strict_checkpoint,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=normalized_cat_dims,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            input_transform=None,
            outcome_transform=outcome_transform,
            feature_extractor=feature_extractor,
            latent_dim=latent_dim,
        )

    @property
    def categorical_process_dim(self) -> int:
        return len(self.cat_dims)


class RoostMixedDKLModel(RoostMixedGPModel):
    """Mixed Roost GP with partial or full encoder fine-tuning."""

    def __init__(self, *args, encoder_training: Literal["partial", "full"] = "partial",
                 trainable_encoder_layers: int = 1, **kwargs) -> None:
        resolved_training = _validate_encoder_training(encoder_training)
        resolved_layers = _validate_trainable_encoder_layers(trainable_encoder_layers)
        super().__init__(*args, **kwargs)
        training_mode, trainable_modules = _configure_dkl_encoder(
            self.material_encoder,
            encoder_training=resolved_training,
            trainable_encoder_layers=resolved_layers,
        )
        self._encoder_training = resolved_training
        self._trainable_encoder_layers = resolved_layers
        self.material_feature_extractor._configure_encoder_training(training_mode, trainable_modules)

    @property
    def encoder_training(self) -> Literal["partial", "full"]:
        return self._encoder_training

    @property
    def trainable_encoder_layers(self) -> int:
        return self._trainable_encoder_layers


class RoostMultiTaskGPModel(_CorrelatedRoostMixin, _RoostModelProperties, DeepKernelGaussianGPModel):
    """Correlated wide-output GP over one shared frozen Roost representation."""

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
        _validate_wide_targets(train_X, train_Y, model_name=self.__class__.__name__)
        validated_element_ids = _validate_composition_element_ids(element_ids)
        composition_dim = int(validated_element_ids.numel())
        if isinstance(input_transform, CompositionMaterialInputTransform):
            process_dim = input_transform.process_dim
        else:
            if train_X.shape[-1] < composition_dim:
                raise ValueError("train_X must contain one fraction column for every element_id.")
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
        DeepKernelGaussianGPModel.__init__(
            self,
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            input_transform=resolved_input_transform,
            outcome_transform=outcome_transform,
            feature_extractor=feature_extractor,
            latent_dim=latent_dim,
        )
        self._validate_correlated_kernel()


class RoostMultiTaskDKLModel(RoostMultiTaskGPModel):
    """Correlated Roost wide-output GP with encoder fine-tuning."""

    def __init__(self, *args, encoder_training: Literal["partial", "full"] = "partial",
                 trainable_encoder_layers: int = 1, **kwargs) -> None:
        resolved_training = _validate_encoder_training(encoder_training)
        resolved_layers = _validate_trainable_encoder_layers(trainable_encoder_layers)
        super().__init__(*args, **kwargs)
        training_mode, trainable_modules = _configure_dkl_encoder(
            self.material_encoder,
            encoder_training=resolved_training,
            trainable_encoder_layers=resolved_layers,
        )
        self._encoder_training = resolved_training
        self._trainable_encoder_layers = resolved_layers
        self.material_feature_extractor._configure_encoder_training(training_mode, trainable_modules)


class RoostMixedMultiTaskGPModel(
    _CorrelatedRoostMixin,
    _RoostModelProperties,
    DeepKernelGaussianMixedGPModel,
):
    """Correlated wide-output Roost GP with mixed process inputs."""

    _supports_cache_root = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Tensor | None = None,
        *,
        element_ids: Tensor,
        composition_indices: Sequence[int],
        method: str = "ilr",
        reference_index: int | None = None,
        process_bounds: Tensor | None = None,
        component_weights: Tensor | None = None,
        normalize_process: bool = True,
        encoder: RoostEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        encoder_output_dim: int | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        strict_checkpoint: bool = True,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransformArg = "DEFAULT",
    ) -> None:
        _validate_wide_targets(train_X, train_Y, model_name=self.__class__.__name__)
        normalized_cat_dims, feature_extractor = _build_mixed_roost(
            train_X=train_X,
            cat_dims=cat_dims,
            element_ids=element_ids,
            composition_indices=composition_indices,
            method=method,
            reference_index=reference_index,
            process_bounds=process_bounds,
            component_weights=component_weights,
            normalize_process=normalize_process,
            encoder=encoder,
            checkpoint=checkpoint,
            encoder_output_dim=encoder_output_dim,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
            strict_checkpoint=strict_checkpoint,
        )
        DeepKernelGaussianMixedGPModel.__init__(
            self,
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=normalized_cat_dims,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            input_transform=None,
            outcome_transform=outcome_transform,
            feature_extractor=feature_extractor,
            latent_dim=latent_dim,
        )
        self._validate_correlated_kernel()


class RoostMixedMultiTaskDKLModel(RoostMixedMultiTaskGPModel):
    """Mixed correlated Roost wide-output GP with encoder fine-tuning."""

    def __init__(self, *args, encoder_training: Literal["partial", "full"] = "partial",
                 trainable_encoder_layers: int = 1, **kwargs) -> None:
        resolved_training = _validate_encoder_training(encoder_training)
        resolved_layers = _validate_trainable_encoder_layers(trainable_encoder_layers)
        super().__init__(*args, **kwargs)
        training_mode, trainable_modules = _configure_dkl_encoder(
            self.material_encoder,
            encoder_training=resolved_training,
            trainable_encoder_layers=resolved_layers,
        )
        self._encoder_training = resolved_training
        self._trainable_encoder_layers = resolved_layers
        self.material_feature_extractor._configure_encoder_training(training_mode, trainable_modules)


__all__ = [
    "RoostMixedDKLModel",
    "RoostMixedGPModel",
    "RoostMixedMultiTaskDKLModel",
    "RoostMixedMultiTaskGPModel",
    "RoostMultiTaskDKLModel",
    "RoostMultiTaskGPModel",
]
