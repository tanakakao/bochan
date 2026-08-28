"""Shared material-representation infrastructure for Gaussian deep kernels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import torch
from botorch.models.transforms.input import InputTransform
from torch import Tensor, nn

from bochan.composition import (
    MaterialEncoder,
    MaterialProcessFusion,
    TorchSimplexTransform,
    build_material_process_fusion,
)

EncoderTrainingMode = Literal["frozen", "partial", "full"]


class CompositionMaterialInputTransform(InputTransform):
    """Pack composition coordinates and continuous process features.

    The transform converts fraction, CLR, ALR, or ILR composition coordinates
    back to atomic fractions using Torch-only operations. Continuous process
    columns may be normalized independently. The complete operation preserves
    autograd for training and acquisition optimization.

    Args:
        input_dim: Width of the raw model-coordinate tensor.
        composition_indices: Columns containing composition coordinates in the
            fixed-element order.
        n_components: Number of elemental fractions in the packed output.
        method: Composition representation: fractions, CLR, ALR, or ILR.
        reference_index: ALR denominator component, when applicable.
        process_bounds: Optional `[2, process_dim]` process bounds.
        component_weights: Optional positive component weights. Atomic weights
            convert weight-basis fractions to atomic fractions.
        normalize_process: Normalize continuous process columns to `[0, 1]`.
    """

    is_one_to_many = False
    composition_indices: Tensor
    process_indices: Tensor
    process_lower: Tensor
    process_range: Tensor
    component_weights: Tensor

    def __init__(
        self,
        *,
        input_dim: int,
        composition_indices: Sequence[int],
        n_components: int,
        method: str = "ilr",
        reference_index: int | None = None,
        process_bounds: Tensor | None = None,
        component_weights: Tensor | None = None,
        normalize_process: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")

        indices = [int(index) for index in composition_indices]
        if not indices:
            raise ValueError("composition_indices must not be empty.")
        if len(indices) != len(set(indices)):
            raise ValueError("composition_indices must not contain duplicates.")
        if min(indices) < 0 or max(indices) >= input_dim:
            raise ValueError("composition_indices must be valid raw input columns.")

        simplex = TorchSimplexTransform(
            n_components=n_components,
            method=method,
            reference_index=reference_index,
        )
        if len(indices) != simplex.input_dim:
            raise ValueError(
                "composition_indices width does not match the configured "
                f"composition representation: {len(indices)} != {simplex.input_dim}."
            )

        index_set = set(indices)
        process_indices = [index for index in range(input_dim) if index not in index_set]
        process_dim = len(process_indices)
        normalize_process = bool(normalize_process)
        if normalize_process and process_dim and process_bounds is None:
            raise ValueError(
                "process_bounds is required when normalize_process=True and "
                "continuous process columns are present."
            )

        if process_bounds is None:
            process_lower = torch.empty(0, dtype=torch.double)
            process_range = torch.empty(0, dtype=torch.double)
        else:
            if not torch.is_tensor(process_bounds):
                raise TypeError("process_bounds must be a Tensor.")
            if process_bounds.shape != torch.Size([2, process_dim]):
                raise ValueError(
                    "process_bounds must have shape [2, process_dim]: "
                    f"{tuple(process_bounds.shape)} != {(2, process_dim)}."
                )
            if not process_bounds.is_floating_point() or not torch.isfinite(process_bounds).all():
                raise ValueError("process_bounds must be a finite floating-point Tensor.")
            process_lower = process_bounds[0].detach().clone()
            raw_range = process_bounds[1] - process_bounds[0]
            if (raw_range < 0).any():
                raise ValueError("process_bounds upper values must be >= lower values.")
            process_range = torch.where(raw_range > 0, raw_range, torch.ones_like(raw_range))

        if component_weights is None:
            component_weights = torch.ones(n_components, dtype=torch.double)
        elif not torch.is_tensor(component_weights):
            raise TypeError("component_weights must be a Tensor.")
        if component_weights.shape != torch.Size([n_components]):
            raise ValueError(
                "component_weights must contain one value per component: "
                f"{tuple(component_weights.shape)} != {(n_components,)}."
            )
        if (
            not component_weights.is_floating_point()
            or not torch.isfinite(component_weights).all()
            or (component_weights <= 0).any()
        ):
            raise ValueError("component_weights must contain finite positive values.")

        self.input_dim = input_dim
        self.composition_dim = int(n_components)
        self.process_dim = process_dim
        self.output_dim = self.composition_dim + self.process_dim
        self.normalize_process = normalize_process
        self.simplex = simplex
        self.transform_on_train = True
        self.transform_on_eval = True
        self.transform_on_fantasize = True
        self.register_buffer(
            "composition_indices",
            torch.tensor(indices, dtype=torch.long),
        )
        self.register_buffer(
            "process_indices",
            torch.tensor(process_indices, dtype=torch.long),
        )
        self.register_buffer("process_lower", process_lower)
        self.register_buffer("process_range", process_range)
        self.register_buffer(
            "component_weights",
            component_weights.detach().clone(),
        )

    def transform(self, X: Tensor) -> Tensor:
        """Return atomic fractions followed by continuous process values."""

        if not torch.is_tensor(X):
            raise TypeError("X must be a Tensor.")
        if X.ndim == 0 or X.shape[-1] != self.input_dim:
            width = X.shape[-1] if X.ndim else 0
            raise ValueError(f"X width must equal input_dim: {width} != {self.input_dim}.")
        if not X.is_floating_point() or not torch.isfinite(X).all():
            raise ValueError("X must be a finite floating-point Tensor.")

        coordinate_indices = self.composition_indices.to(device=X.device)
        coordinates = X.index_select(-1, coordinate_indices)
        basis_fractions = self.simplex(coordinates)
        weights = self.component_weights.to(dtype=X.dtype, device=X.device)
        atomic_values = basis_fractions / weights
        fractions = atomic_values / atomic_values.sum(dim=-1, keepdim=True)

        if not self.process_dim:
            return fractions
        process_indices = self.process_indices.to(device=X.device)
        process = X.index_select(-1, process_indices)
        if self.normalize_process:
            lower = self.process_lower.to(dtype=X.dtype, device=X.device)
            scale = self.process_range.to(dtype=X.dtype, device=X.device)
            process = (process - lower) / scale
        return torch.cat((fractions, process), dim=-1)

    def extra_repr(self) -> str:
        """Return the raw and packed input contracts for module summaries."""

        return (
            f"input_dim={self.input_dim}, composition_dim={self.composition_dim}, "
            f"process_dim={self.process_dim}, method={self.simplex.method!r}, "
            f"normalize_process={self.normalize_process}"
        )


def _validate_composition_element_ids(element_ids: Tensor) -> Tensor:
    """Validate and clone one fixed elemental vocabulary."""

    if not torch.is_tensor(element_ids):
        raise TypeError("element_ids must be a Tensor.")
    if element_ids.ndim != 1 or element_ids.numel() == 0:
        raise ValueError("element_ids must be a non-empty one-dimensional Tensor.")
    if element_ids.dtype != torch.long:
        raise TypeError("element_ids must have dtype torch.long.")
    if (element_ids <= 0).any():
        raise ValueError("element_ids must contain positive atomic numbers.")
    if element_ids.unique().numel() != element_ids.numel():
        raise ValueError("element_ids must not contain duplicate atomic numbers.")
    return element_ids.detach().clone()


def _validate_composition_model_inputs(
    X: Tensor,
    *,
    composition_dim: int,
    input_dim: int,
) -> None:
    """Validate packed fractions and continuous process columns."""

    if not torch.is_tensor(X):
        raise TypeError("X must be a Tensor.")
    if X.ndim == 0 or X.shape[-1] != input_dim:
        width = X.shape[-1] if X.ndim else 0
        raise ValueError(
            "X width must equal composition_dim + process_dim: "
            f"{width} != {input_dim}."
        )
    if not X.is_floating_point():
        raise TypeError("X must have a floating-point dtype.")
    if not torch.isfinite(X).all():
        raise ValueError("X must contain only finite values.")

    fractions = X[..., :composition_dim]
    if (fractions < 0).any():
        raise ValueError("Composition fractions must be non-negative.")
    if not torch.allclose(
        fractions.sum(dim=-1),
        torch.ones_like(fractions[..., 0]),
        rtol=1e-4,
        atol=1e-6,
    ):
        raise ValueError("Composition fractions must sum to one.")


class _BaseMaterialGPFeatureExtractor(nn.Module, ABC):
    """Fuse one material representation with process features and project it."""

    def __init__(
        self,
        *,
        material_encoder: MaterialEncoder,
        input_dim: int,
        process_dim: int,
        latent_dim: int,
        fusion: Literal["concat"] | MaterialProcessFusion,
        projection: nn.Module | None,
    ) -> None:
        super().__init__()
        if not isinstance(material_encoder, MaterialEncoder):
            raise TypeError("material_encoder must implement MaterialEncoder.")
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if isinstance(process_dim, bool) or not isinstance(process_dim, int) or process_dim < 0:
            raise ValueError("process_dim must be a non-negative integer.")
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")

        self.material_encoder = material_encoder
        self.input_dim = input_dim
        self.process_dim = process_dim
        self.output_dim = latent_dim
        self._encoder_training_mode: EncoderTrainingMode = "frozen"
        self._trainable_encoder_modules: tuple[nn.Module, ...] = ()
        self.fusion = build_material_process_fusion(
            fusion,
            material_dim=material_encoder.output_dim,
            process_dim=process_dim,
        )

        if projection is None:
            projection = nn.Linear(self.fusion.output_dim, latent_dim)
        elif not isinstance(projection, nn.Module):
            raise TypeError("projection must be a torch.nn.Module.")
        declared_output_dim = getattr(projection, "output_dim", None)
        if declared_output_dim is not None and int(declared_output_dim) != latent_dim:
            raise ValueError(
                "projection.output_dim does not match latent_dim: "
                f"{int(declared_output_dim)} != {latent_dim}."
            )
        if isinstance(projection, nn.Linear):
            if projection.in_features != self.fusion.output_dim:
                raise ValueError(
                    "projection.in_features does not match the fused feature width: "
                    f"{projection.in_features} != {self.fusion.output_dim}."
                )
            if projection.out_features != latent_dim:
                raise ValueError(
                    "projection.out_features does not match latent_dim: "
                    f"{projection.out_features} != {latent_dim}."
                )
        self.projection = projection
        for parameter in self.material_encoder.parameters():
            parameter.requires_grad_(False)
        self.material_encoder.eval()

    def train(self, mode: bool = True) -> _BaseMaterialGPFeatureExtractor:
        """Apply the configured frozen, partial, or full encoder mode policy."""

        super().train(mode)
        if self._encoder_training_mode == "frozen":
            self.material_encoder.eval()
        elif self._encoder_training_mode == "partial":
            self.material_encoder.eval()
            for module in self._trainable_encoder_modules:
                module.train(mode)
        return self

    def _on_encoder_training_policy_change(self) -> None:
        """Invalidate encoder-derived state before the training policy changes."""

    def _configure_encoder_training(
        self,
        mode: EncoderTrainingMode,
        trainable_modules: tuple[nn.Module, ...] = (),
    ) -> None:
        """Set the encoder mode policy and apply it immediately."""

        if mode == "partial" and not trainable_modules:
            raise ValueError("Partial encoder training requires at least one trainable module.")
        if mode != "partial" and trainable_modules:
            raise ValueError("Only partial encoder training accepts trainable_modules.")
        self._on_encoder_training_policy_change()
        if mode == "frozen":
            for parameter in self.material_encoder.parameters():
                parameter.requires_grad_(False)
        self._encoder_training_mode = mode
        self._trainable_encoder_modules = trainable_modules
        self.train(self.training)

    @abstractmethod
    def validate_input(self, X: Tensor) -> None:
        """Validate the encoder-specific packed input contract."""

        raise NotImplementedError

    @abstractmethod
    def _material_features(self, X: Tensor) -> Tensor:
        """Encode material inputs while preserving `X` leading dimensions."""

        raise NotImplementedError

    @abstractmethod
    def _process_features(self, X: Tensor) -> Tensor | None:
        """Return process features aligned with the material representation."""

        raise NotImplementedError

    def forward(self, X: Tensor) -> Tensor:
        """Return projected material/process features for a GP kernel."""

        self.validate_input(X)
        material_features = self._material_features(X)
        if not torch.is_tensor(material_features):
            raise TypeError("material_encoder must return a Tensor.")
        expected_material_shape = (*X.shape[:-1], self.material_encoder.output_dim)
        if material_features.shape != expected_material_shape:
            raise ValueError(
                "material_encoder must preserve leading dimensions and return "
                f"output_dim features: {tuple(material_features.shape)} != "
                f"{expected_material_shape}."
            )
        if material_features.device != X.device or material_features.dtype != X.dtype:
            raise ValueError("material_encoder output must match X's device and dtype.")
        if not torch.isfinite(material_features).all():
            raise FloatingPointError("material_encoder produced non-finite features.")

        process_features = self._process_features(X)
        fused_features = self.fusion(material_features, process_features)
        projected_features = self.projection(fused_features)
        if not torch.is_tensor(projected_features):
            raise TypeError("projection must return a Tensor.")
        expected_shape = (*X.shape[:-1], self.output_dim)
        if projected_features.shape != expected_shape:
            raise ValueError(
                "projection must preserve leading dimensions and return latent_dim "
                f"features: {tuple(projected_features.shape)} != {expected_shape}."
            )
        if projected_features.device != X.device or projected_features.dtype != X.dtype:
            raise ValueError("projection output must match X's device and dtype.")
        if not torch.isfinite(projected_features).all():
            raise FloatingPointError("Material/process projection produced non-finite values.")
        return projected_features


class MaterialGPFeatureExtractor(_BaseMaterialGPFeatureExtractor):
    """Encode fixed-vocabulary fractions and process features for a GP.

    This composition contract is shared by material encoders such as CrabNet
    and Roost. Leading batch and q-batch dimensions are preserved. Element
    slots whose fractions are at or below `zero_tolerance` are converted to
    padding without crossing NumPy or detaching active fraction gradients.
    """

    element_ids: Tensor

    def __init__(
        self,
        *,
        material_encoder: MaterialEncoder,
        element_ids: Tensor,
        process_dim: int,
        latent_dim: int,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        zero_tolerance: float = 0.0,
    ) -> None:
        validated_element_ids = _validate_composition_element_ids(element_ids)
        if not isinstance(zero_tolerance, (int, float)) or isinstance(zero_tolerance, bool):
            raise TypeError("zero_tolerance must be a real number.")
        zero_tolerance = float(zero_tolerance)
        if not 0.0 <= zero_tolerance < 1.0:
            raise ValueError("zero_tolerance must be in the interval [0, 1).")

        composition_dim = int(validated_element_ids.numel())
        super().__init__(
            material_encoder=material_encoder,
            input_dim=composition_dim + process_dim,
            process_dim=process_dim,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
        )
        self.register_buffer("element_ids", validated_element_ids)
        self.composition_dim = composition_dim
        self.zero_tolerance = zero_tolerance

    def validate_input(self, X: Tensor) -> None:
        """Validate fractions and continuous process columns."""

        _validate_composition_model_inputs(
            X,
            composition_dim=self.composition_dim,
            input_dim=self.input_dim,
        )

    def _material_features(self, X: Tensor) -> Tensor:
        fractions = X[..., : self.composition_dim]
        active = fractions > self.zero_tolerance
        if not active.any(dim=-1).all():
            raise ValueError("Each composition must contain at least one active element.")
        active_fractions = torch.where(active, fractions, torch.zeros_like(fractions))
        active_fractions = active_fractions / active_fractions.sum(dim=-1, keepdim=True)
        expanded_ids = self.element_ids.view(
            *((1,) * (fractions.ndim - 1)),
            self.composition_dim,
        ).expand_as(fractions)
        active_element_ids = expanded_ids.masked_fill(~active, 0)
        return self.material_encoder(active_element_ids, active_fractions)

    def _process_features(self, X: Tensor) -> Tensor | None:
        if not self.process_dim:
            return None
        return X[..., self.composition_dim :]


__all__ = [
    "CompositionMaterialInputTransform",
    "MaterialGPFeatureExtractor",
]
