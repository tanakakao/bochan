"""Correlated multi-output Gaussian processes over shared CHGNet representations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

from botorch.acquisition.objective import PosteriorTransform
from botorch.posteriors.transformed import TransformedPosterior
from botorch.utils.transforms import normalize_indices
from gpytorch.kernels import MultitaskKernel
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import CHGNetEncoder, MaterialProcessFusion
from bochan.composition.encoders.chgnet import Checkpoint

from .chgnet import (
    _CHGNetGPFeatureExtractor,
    _configure_dkl_encoder,
    _resolve_input_transform,
    _resolve_material_encoder,
    _resolve_mixed_input_transform,
    _validate_model_inputs,
    _validate_structure_bank,
    _validate_trainable_encoder_layers,
)
from .deepkernel import InputTransformArg, OutcomeTransformArg
from .deepkernel_configurable import (
    DeepKernelGaussianGPModel,
    DeepKernelGaussianMixedGPModel,
)


class _CorrelatedCHGNetMultiTaskMixin:
    """Expose shared diagnostics and posterior semantics for correlated CHGNet tasks."""

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
        """Return the number of correlated material-property tasks."""

        return int(self.num_outputs)

    @property
    def task_covar_module(self) -> nn.Module:
        """Expose the learned task covariance module for diagnostics."""

        self._validate_correlated_kernel()
        return cast(MultitaskKernel, self.deepkernel.covar_module).task_covar_module

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
    ):
        """Return the correlated posterior, optionally restricted to output tasks."""

        if output_indices is None:
            return super().posterior(
                X,
                output_indices=None,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
            )

        indices = normalize_indices(indices=list(output_indices), d=self.num_tasks)
        if not indices:
            raise ValueError("output_indices must contain at least one task index.")

        posterior = super().posterior(
            X,
            output_indices=None,
            observation_noise=observation_noise,
            posterior_transform=None,
        )
        subset = TransformedPosterior(
            posterior=posterior,
            sample_transform=lambda samples: samples[..., indices],
            mean_transform=lambda mean, variance: mean[..., indices],
            variance_transform=lambda mean, variance: variance[..., indices],
        )
        if posterior_transform is not None:
            subset = posterior_transform(subset)
        return subset

    @property
    def chgnet_feature_extractor(self) -> _CHGNetGPFeatureExtractor:
        """Return the shared CHGNet structure/process feature extractor."""

        return cast(_CHGNetGPFeatureExtractor, self.deepkernel.feature_extractor)

    @property
    def material_encoder(self) -> CHGNetEncoder:
        """Return the single CHGNet encoder shared by every output task."""

        return cast(CHGNetEncoder, self.chgnet_feature_extractor.material_encoder)

    @property
    def projection(self) -> nn.Module:
        """Return the shared trainable latent projection."""

        return self.chgnet_feature_extractor.projection

    @property
    def fusion(self) -> MaterialProcessFusion:
        """Return the shared structure/process fusion module."""

        return self.chgnet_feature_extractor.fusion

    @property
    def structures(self) -> tuple[Any, ...]:
        """Return the canonical raw structure bank."""

        return self.chgnet_feature_extractor.structures

    @property
    def num_structures(self) -> int:
        """Return the number of structures in the raw structure bank."""

        return self.chgnet_feature_extractor.num_structures

    @property
    def process_dim(self) -> int:
        """Return the number of continuous process dimensions."""

        return self.chgnet_feature_extractor.process_dim

    @property
    def structure_feature_cache_enabled(self) -> bool:
        """Return whether frozen structure representations are reusable."""

        return self.chgnet_feature_extractor.material_feature_cache_enabled

    def clear_structure_feature_cache(self) -> None:
        """Discard the derived frozen structure representation cache."""

        self.chgnet_feature_extractor.clear_material_feature_cache()


def _validate_multitask_targets(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None,
    *,
    model_name: str,
    latent_dim: int,
) -> None:
    """Validate the wide-target contract used by correlated CHGNet models."""

    if train_X.ndim != 2:
        raise ValueError("train_X must have shape [n, d].")
    if train_X.shape[-1] < 1:
        raise ValueError("train_X must contain a structure-index column.")
    if train_Y.ndim != 2 or train_Y.shape[-1] < 2:
        raise ValueError(
            f"{model_name} requires wide train_Y with shape [n, m] and at least "
            "two target columns."
        )
    if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
        raise ValueError("latent_dim must be a positive integer.")


class CHGNetMultiTaskGPModel(
    _CorrelatedCHGNetMultiTaskMixin,
    DeepKernelGaussianGPModel,
):
    """Correlated multi-output GP with one shared frozen CHGNet encoder."""

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
        _validate_multitask_targets(
            train_X,
            train_Y,
            train_Yvar,
            model_name=self.__class__.__name__,
            latent_dim=latent_dim,
        )
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
        transformed_train_X = cast(tuple[Tensor, ...], self.deepkernel.train_inputs)[0]
        self.chgnet_feature_extractor.validate_input(transformed_train_X)

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> CHGNetMultiTaskGPModel:
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
        return cast(CHGNetMultiTaskGPModel, module)


class CHGNetMultiTaskDKLModel(CHGNetMultiTaskGPModel):
    """Correlated CHGNet multitask DKL with shared encoder fine-tuning."""

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
        """Return the shared encoder fine-tuning policy."""

        return self._trainable_encoder_layers


class CHGNetMixedMultiTaskGPModel(
    _CorrelatedCHGNetMultiTaskMixin,
    DeepKernelGaussianMixedGPModel,
):
    """Correlated CHGNet multitask GP with categorical process variables."""

    _supports_cache_root = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        cat_dims: Sequence[int],
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
        _validate_multitask_targets(
            train_X,
            train_Y,
            train_Yvar,
            model_name=self.__class__.__name__,
            latent_dim=latent_dim,
        )
        d = train_X.shape[-1]
        normalized_cat_dims = normalize_indices(indices=list(cat_dims), d=d)
        if not normalized_cat_dims:
            raise ValueError(
                "CHGNetMixedMultiTaskGPModel requires at least one categorical "
                "process dimension."
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
        DeepKernelGaussianMixedGPModel.__init__(
            self,
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=normalized_cat_dims,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            input_transform=resolved_input_transform,
            outcome_transform=outcome_transform,
            feature_extractor=feature_extractor,
            latent_dim=latent_dim,
        )
        self._validate_correlated_kernel()
        transformed_train_X = cast(tuple[Tensor, ...], self.deepkernel.train_inputs)[0]
        continuous_train_X = transformed_train_X[..., self.deepkernel.ord_dims]
        self.chgnet_feature_extractor.validate_input(continuous_train_X)

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> CHGNetMixedMultiTaskGPModel:
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
        return cast(CHGNetMixedMultiTaskGPModel, module)

    @property
    def continuous_process_dims(self) -> tuple[int, ...]:
        """Return raw input indices of numeric process columns."""

        return self._continuous_process_dims

    @property
    def categorical_process_dim(self) -> int:
        """Return the number of categorical process dimensions."""

        return len(self.cat_dims)


class CHGNetMixedMultiTaskDKLModel(CHGNetMixedMultiTaskGPModel):
    """Correlated mixed CHGNet multitask DKL with shared encoder fine-tuning."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        cat_dims: Sequence[int],
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
            cat_dims=cat_dims,
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
        """Return the shared encoder fine-tuning policy."""

        return self._trainable_encoder_layers


__all__ = [
    "CHGNetMixedMultiTaskDKLModel",
    "CHGNetMixedMultiTaskGPModel",
    "CHGNetMultiTaskDKLModel",
    "CHGNetMultiTaskGPModel",
]
