"""CrabNet deep-kernel GP with learned categorical process embeddings."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal, cast

import torch
from botorch.utils.transforms import normalize_indices
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import CrabNetEncoder
from bochan.composition.encoders.crabnet import Checkpoint

from .crabnet import (
    CrabNetInputTransform,
    _CrabNetGPFeatureExtractor,
    _configure_dkl_encoder,
    _resolve_material_encoder,
    _validate_element_ids,
    _validate_trainable_encoder_layers,
)
from .deepkernel import OutcomeTransformArg
from .deepkernel_configurable import DeepKernelGaussianGPModel


def _resolve_category_cardinalities(
    train_X: Tensor,
    cat_dims: Sequence[int],
    category_cardinalities: Sequence[int] | None,
) -> tuple[int, ...]:
    """Resolve stable category cardinalities for embedding tables."""

    values = train_X[:, list(cat_dims)]
    if not torch.isfinite(values).all():
        raise ValueError("Categorical process columns must contain only finite values.")
    rounded = values.round()
    if not torch.allclose(values, rounded, rtol=0.0, atol=1e-6):
        raise ValueError("Categorical process columns must use integer-coded values.")
    if (rounded < 0).any():
        raise ValueError("Categorical process codes must be non-negative.")

    if category_cardinalities is None:
        return tuple(int(rounded[:, index].max().item()) + 1 for index in range(rounded.shape[-1]))

    cardinalities = tuple(int(value) for value in category_cardinalities)
    if len(cardinalities) != len(cat_dims):
        raise ValueError(
            "category_cardinalities must contain one value per categorical dimension: "
            f"{len(cardinalities)} != {len(cat_dims)}."
        )
    if any(value <= 0 for value in cardinalities):
        raise ValueError("category_cardinalities must contain positive integers.")
    for index, cardinality in enumerate(cardinalities):
        maximum = int(rounded[:, index].max().item())
        if maximum >= cardinality:
            raise ValueError(
                "Categorical process code exceeds its configured cardinality: "
                f"{maximum} >= {cardinality} for categorical column {index}."
            )
    return cardinalities


def _resolve_category_embedding_dims(
    cardinalities: Sequence[int],
    category_embedding_dims: int | Sequence[int] | None,
) -> tuple[int, ...]:
    """Resolve one trainable embedding width per categorical process column."""

    if category_embedding_dims is None:
        return tuple(max(2, min(16, int(math.ceil(math.sqrt(value))))) for value in cardinalities)
    if isinstance(category_embedding_dims, bool):
        raise TypeError("category_embedding_dims must be a positive integer or sequence of integers.")
    if isinstance(category_embedding_dims, int):
        if category_embedding_dims <= 0:
            raise ValueError("category_embedding_dims must be positive.")
        return tuple(category_embedding_dims for _ in cardinalities)

    dims = tuple(int(value) for value in category_embedding_dims)
    if len(dims) != len(cardinalities):
        raise ValueError(
            "category_embedding_dims must contain one value per categorical dimension: "
            f"{len(dims)} != {len(cardinalities)}."
        )
    if any(value <= 0 for value in dims):
        raise ValueError("category_embedding_dims must contain positive integers.")
    return dims


class _CrabNetMixedDKLFeatureExtractor(nn.Module):
    """Fuse CrabNet, numeric process values, and learned category embeddings."""

    categorical_indices: Tensor
    continuous_indices: Tensor

    def __init__(
        self,
        *,
        input_dim: int,
        cat_dims: Sequence[int],
        composition_indices: Sequence[int],
        element_ids: Tensor,
        method: str,
        reference_index: int | None,
        process_bounds: Tensor | None,
        component_weights: Tensor | None,
        normalize_process: bool,
        material_encoder: CrabNetEncoder,
        category_cardinalities: Sequence[int],
        category_embedding_dims: Sequence[int],
        latent_dim: int,
        projection_hidden_dim: int | None,
        projection: nn.Module | None,
    ) -> None:
        super().__init__()
        normalized_cat_dims = normalize_indices(indices=list(cat_dims), d=input_dim)
        continuous_dims = sorted(set(range(input_dim)) - set(normalized_cat_dims))
        raw_composition_indices = [int(index) for index in composition_indices]
        continuous_position = {raw_index: index for index, raw_index in enumerate(continuous_dims)}
        continuous_composition_indices = [
            continuous_position[index] for index in raw_composition_indices
        ]

        validated_element_ids = _validate_element_ids(element_ids)
        self.packer = CrabNetInputTransform(
            input_dim=len(continuous_dims),
            composition_indices=continuous_composition_indices,
            n_components=int(validated_element_ids.numel()),
            method=method,
            reference_index=reference_index,
            process_bounds=process_bounds,
            component_weights=component_weights,
            normalize_process=normalize_process,
        )
        base_feature_dim = int(material_encoder.output_dim) + int(self.packer.process_dim)
        self.crabnet = _CrabNetGPFeatureExtractor(
            material_encoder=material_encoder,
            element_ids=validated_element_ids,
            process_dim=self.packer.process_dim,
            latent_dim=base_feature_dim,
            fusion="concat",
            projection=nn.Identity(),
        )

        self.category_cardinalities = tuple(int(value) for value in category_cardinalities)
        self.category_embedding_dims = tuple(int(value) for value in category_embedding_dims)
        self.category_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, embedding_dim)
            for cardinality, embedding_dim in zip(
                self.category_cardinalities,
                self.category_embedding_dims,
                strict=True,
            )
        )
        self.register_buffer(
            "categorical_indices",
            torch.tensor(normalized_cat_dims, dtype=torch.long),
        )
        self.register_buffer(
            "continuous_indices",
            torch.tensor(continuous_dims, dtype=torch.long),
        )
        self.input_dim = int(input_dim)
        self.output_dim = int(latent_dim)
        fused_dim = base_feature_dim + sum(self.category_embedding_dims)

        if projection is None:
            if projection_hidden_dim is None:
                projection_hidden_dim = max(self.output_dim * 2, min(128, fused_dim * 2))
            if (
                isinstance(projection_hidden_dim, bool)
                or not isinstance(projection_hidden_dim, int)
                or projection_hidden_dim <= 0
            ):
                raise ValueError("projection_hidden_dim must be a positive integer.")
            projection = nn.Sequential(
                nn.Linear(fused_dim, projection_hidden_dim),
                nn.SiLU(),
                nn.Linear(projection_hidden_dim, self.output_dim),
            )
        elif not isinstance(projection, nn.Module):
            raise TypeError("projection must be a torch.nn.Module.")

        declared_output_dim = getattr(projection, "output_dim", None)
        if declared_output_dim is not None and int(declared_output_dim) != self.output_dim:
            raise ValueError(
                "projection.output_dim does not match latent_dim: "
                f"{int(declared_output_dim)} != {self.output_dim}."
            )
        if isinstance(projection, nn.Linear):
            if projection.in_features != fused_dim:
                raise ValueError(
                    "projection.in_features does not match the fused feature width: "
                    f"{projection.in_features} != {fused_dim}."
                )
            if projection.out_features != self.output_dim:
                raise ValueError(
                    "projection.out_features does not match latent_dim: "
                    f"{projection.out_features} != {self.output_dim}."
                )
        self.projection = projection

    def _categorical_features(self, X: Tensor) -> Tensor:
        category_values = X.index_select(-1, self.categorical_indices.to(device=X.device))
        if not torch.isfinite(category_values).all():
            raise ValueError("Categorical process columns must contain only finite values.")
        rounded = category_values.round()
        if not torch.allclose(category_values, rounded, rtol=0.0, atol=1e-6):
            raise ValueError("Categorical process columns must use integer-coded values.")
        if (rounded < 0).any():
            raise ValueError("Categorical process codes must be non-negative.")

        encoded: list[Tensor] = []
        for index, (embedding, cardinality) in enumerate(
            zip(self.category_embeddings, self.category_cardinalities, strict=True)
        ):
            codes = rounded[..., index].to(dtype=torch.long)
            if int(codes.max().item()) >= cardinality:
                raise ValueError(
                    "Categorical process code exceeds its configured cardinality: "
                    f"{int(codes.max().item())} >= {cardinality} for categorical column {index}."
                )
            encoded.append(embedding(codes))
        return torch.cat(encoded, dim=-1)

    def forward(self, X: Tensor) -> Tensor:
        """Return the learned joint material/process/category representation."""

        if not torch.is_tensor(X):
            raise TypeError("X must be a Tensor.")
        if X.ndim == 0 or X.shape[-1] != self.input_dim:
            raise ValueError(
                f"X width must equal input_dim: {X.shape[-1] if X.ndim else 0} != {self.input_dim}."
            )
        if not X.is_floating_point() or not torch.isfinite(X).all():
            raise ValueError("X must be a finite floating-point Tensor.")

        continuous = X.index_select(-1, self.continuous_indices.to(device=X.device))
        packed = self.packer(continuous)
        material_numeric = self.crabnet(packed)
        categorical = self._categorical_features(X)
        fused = torch.cat((material_numeric, categorical), dim=-1)
        projected = self.projection(fused)
        if not torch.is_tensor(projected):
            raise TypeError("projection must return a Tensor.")
        expected_shape = (*X.shape[:-1], self.output_dim)
        if projected.shape != expected_shape:
            raise ValueError(
                "projection must preserve leading dimensions and return latent_dim features: "
                f"{tuple(projected.shape)} != {expected_shape}."
            )
        if projected.device != X.device or projected.dtype != X.dtype:
            raise ValueError("projection output must match X's device and dtype.")
        if not torch.isfinite(projected).all():
            raise FloatingPointError("CrabNet mixed DKL projection produced non-finite values.")
        return projected


class CrabNetMixedDKLModel(DeepKernelGaussianGPModel):
    """Exact DKL GP over composition, numeric process, and categorical process inputs.

    Composition coordinates are converted back to atomic fractions and encoded by
    CrabNet. Numeric process values are normalized and concatenated with the
    material representation. Each categorical process column is represented by a
    trainable embedding table. The material, numeric, and categorical features are
    then fused by a trainable neural projection before the exact GP kernel.

    The categorical raw columns remain in ``cat_dims`` at the public API level so
    candidate generation can continue to use ``optimize_acqf_mixed``. During each
    fixed categorical assignment, gradients flow through the composition and
    numeric process dimensions into the learned DKL representation.
    """

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
        encoder: CrabNetEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        latent_dim: int = 32,
        category_cardinalities: Sequence[int] | None = None,
        category_embedding_dims: int | Sequence[int] | None = None,
        projection_hidden_dim: int | None = None,
        projection: nn.Module | None = None,
        strict_checkpoint: bool = True,
        trainable_encoder_layers: int | Literal["all"] = 1,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransformArg = "DEFAULT",
    ) -> None:
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape [n, d].")
        if train_Y.ndim > 1 and train_Y.shape[-1] != 1:
            raise ValueError("CrabNetMixedDKLModel currently supports single-output train_Y only.")
        if train_Yvar is not None:
            raise NotImplementedError("CrabNetMixedDKLModel does not yet support train_Yvar.")
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")

        d = train_X.shape[-1]
        normalized_cat_dims = normalize_indices(indices=list(cat_dims), d=d)
        if not normalized_cat_dims:
            raise ValueError("CrabNetMixedDKLModel requires at least one categorical process dimension.")
        raw_composition_indices = [int(index) for index in composition_indices]
        if not raw_composition_indices:
            raise ValueError("composition_indices must not be empty.")
        if len(raw_composition_indices) != len(set(raw_composition_indices)):
            raise ValueError("composition_indices must not contain duplicates.")
        if min(raw_composition_indices) < 0 or max(raw_composition_indices) >= d:
            raise ValueError("composition_indices must be valid train_X columns.")
        if any(index in normalized_cat_dims for index in raw_composition_indices):
            raise ValueError("composition_indices must refer only to continuous composition coordinates.")

        cardinalities = _resolve_category_cardinalities(
            train_X,
            normalized_cat_dims,
            category_cardinalities,
        )
        embedding_dims = _resolve_category_embedding_dims(
            cardinalities,
            category_embedding_dims,
        )
        resolved_trainable_layers = _validate_trainable_encoder_layers(trainable_encoder_layers)
        material_encoder = _resolve_material_encoder(
            encoder,
            checkpoint,
            strict_checkpoint=strict_checkpoint,
        )
        feature_extractor = _CrabNetMixedDKLFeatureExtractor(
            input_dim=d,
            cat_dims=normalized_cat_dims,
            composition_indices=raw_composition_indices,
            element_ids=element_ids,
            method=method,
            reference_index=reference_index,
            process_bounds=process_bounds,
            component_weights=component_weights,
            normalize_process=normalize_process,
            material_encoder=material_encoder,
            category_cardinalities=cardinalities,
            category_embedding_dims=embedding_dims,
            latent_dim=latent_dim,
            projection_hidden_dim=projection_hidden_dim,
            projection=projection,
        ).to(train_X)

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=None,
            likelihood=likelihood,
            input_transform=None,
            outcome_transform=outcome_transform,
            feature_extractor=feature_extractor,
            latent_dim=latent_dim,
        )
        self.cat_dims = list(normalized_cat_dims)
        self.ord_dims = sorted(set(range(d)) - set(normalized_cat_dims))
        self._ignore_X_dims_scaling_check = list(normalized_cat_dims)
        self.raw_composition_indices = tuple(raw_composition_indices)

        training_mode, trainable_modules = _configure_dkl_encoder(
            self.material_encoder,
            resolved_trainable_layers,
        )
        self._trainable_encoder_layers = resolved_trainable_layers
        self.mixed_dkl_feature_extractor.crabnet._configure_encoder_training(
            training_mode,
            trainable_modules,
        )

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> CrabNetMixedDKLModel:
        """Move all components and refresh the wrapper dtype/device contract."""

        module = super()._apply(fn, recurse=recurse)
        reference = next(
            (value for value in (*self.parameters(), *self.buffers()) if value.is_floating_point()),
            None,
        )
        if reference is not None:
            self._model_dtype = reference.dtype
            self._model_device = reference.device
        return cast(CrabNetMixedDKLModel, module)

    @property
    def mixed_dkl_feature_extractor(self) -> _CrabNetMixedDKLFeatureExtractor:
        """Return the joint material/process/category feature extractor."""

        return cast(_CrabNetMixedDKLFeatureExtractor, self.deepkernel.feature_extractor)

    @property
    def material_encoder(self) -> CrabNetEncoder:
        """Return the CrabNet material encoder."""

        return self.mixed_dkl_feature_extractor.crabnet.material_encoder

    @property
    def category_embeddings(self) -> nn.ModuleList:
        """Return the learned categorical process embedding tables."""

        return self.mixed_dkl_feature_extractor.category_embeddings

    @property
    def category_cardinalities(self) -> tuple[int, ...]:
        """Return the configured category count for each categorical process column."""

        return self.mixed_dkl_feature_extractor.category_cardinalities

    @property
    def category_embedding_dims(self) -> tuple[int, ...]:
        """Return the embedding width for each categorical process column."""

        return self.mixed_dkl_feature_extractor.category_embedding_dims

    @property
    def trainable_encoder_layers(self) -> int | Literal["all"]:
        """Return the partial/full CrabNet fine-tuning policy."""

        return self._trainable_encoder_layers

    @property
    def composition_dim(self) -> int:
        """Return the number of elemental fractions presented to CrabNet."""

        return self.mixed_dkl_feature_extractor.crabnet.composition_dim

    @property
    def process_dim(self) -> int:
        """Return the number of numeric process dimensions fused with CrabNet."""

        return self.mixed_dkl_feature_extractor.crabnet.process_dim

    @property
    def categorical_process_dim(self) -> int:
        """Return the number of learned categorical process embeddings."""

        return len(self.cat_dims)


__all__ = ["CrabNetMixedDKLModel"]
