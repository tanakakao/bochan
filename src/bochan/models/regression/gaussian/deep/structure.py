"""Shared crystal-structure representation infrastructure for Gaussian deep kernels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch
from botorch.models.transforms.input import Normalize
from torch import Tensor, nn

from bochan.composition import MaterialEncoder, MaterialProcessFusion

from .deepkernel import InputTransformArg
from .material import _BaseMaterialGPFeatureExtractor


def _validate_structure_bank(
    structures: Sequence[Any],
    *,
    argument_name: str = "structure_inputs",
) -> tuple[Any, ...]:
    """Validate and freeze one non-empty bank of structure-specific inputs."""

    if not isinstance(structures, Sequence) or isinstance(structures, (str, bytes)):
        raise TypeError(f"{argument_name} must be a sequence.")
    if not structures:
        raise ValueError(f"{argument_name} must contain at least one structure input.")
    return tuple(structures)


def _validate_structure_model_inputs(
    X: Tensor,
    *,
    num_structures: int,
    input_dim: int,
    encoder_name: str | None = None,
) -> None:
    """Validate a discrete structure selector followed by process columns."""

    if not torch.is_tensor(X):
        raise TypeError("X must be a Tensor.")
    if X.ndim == 0 or X.shape[-1] != input_dim:
        width = X.shape[-1] if X.ndim else 0
        raise ValueError(f"X width must equal input_dim: {width} != {input_dim}.")
    if not X.is_floating_point():
        raise TypeError("X must have a floating-point dtype.")
    if not torch.isfinite(X).all():
        raise ValueError("X must contain only finite values.")

    indices = X[..., 0]
    rounded = indices.round()
    if not torch.allclose(indices, rounded, rtol=0.0, atol=1e-6):
        prefix = f"{encoder_name} " if encoder_name else "structure "
        raise ValueError(
            f"The first {prefix}input column must contain integer-valued structure indices."
        )
    if (rounded < 0).any() or (rounded >= num_structures).any():
        raise ValueError(f"Structure indices must be in [0, {num_structures - 1}].")


def _resolve_structure_input_transform(
    train_X: Tensor,
    *,
    input_transform: InputTransformArg,
) -> InputTransformArg:
    """Resolve ``DEFAULT`` to process-only normalization, preserving structure IDs."""

    if not isinstance(input_transform, str) or input_transform.upper() != "DEFAULT":
        return input_transform
    process_dims = list(range(1, train_X.shape[-1]))
    if not process_dims:
        return None
    return Normalize(d=train_X.shape[-1], indices=process_dims)


class _StructureGPFeatureExtractor(_BaseMaterialGPFeatureExtractor):
    """Map structure-bank indices and process features to a GP latent space.

    The class owns behavior that is independent of the atomistic backend:
    structure-index validation, frozen structure-representation caching,
    process-feature extraction, material/process fusion, and latent projection.
    Concrete backends only need to provide a :class:`MaterialEncoder` that
    accepts the selected structure inputs and returns one fixed-width vector per
    structure.
    """

    def __init__(
        self,
        *,
        material_encoder: MaterialEncoder,
        structure_inputs: Sequence[Any],
        process_dim: int,
        latent_dim: int,
        fusion: Literal["concat"] | MaterialProcessFusion,
        projection: nn.Module | None,
        structure_argument_name: str = "structure_inputs",
        encoder_name: str | None = None,
    ) -> None:
        validated_structures = _validate_structure_bank(
            structure_inputs,
            argument_name=structure_argument_name,
        )
        super().__init__(
            material_encoder=material_encoder,
            input_dim=1 + process_dim,
            process_dim=process_dim,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
        )
        self.structure_inputs = validated_structures
        self.num_structures = len(self.structure_inputs)
        self.structure_argument_name = structure_argument_name
        self.encoder_name = encoder_name
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

    def _encoder_label(self) -> str:
        return f"{self.encoder_name} encoder" if self.encoder_name else "Structure encoder"

    def _structure_feature_label(self) -> str:
        return f"{self.encoder_name} structure features" if self.encoder_name else "Structure features"

    def _encode_structure_inputs(self, structures: Sequence[Any]) -> Tensor:
        """Encode one selected batch of backend-specific structure inputs."""

        return self.material_encoder(list(structures))

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
                cache = self._encode_structure_inputs(self.structure_inputs).detach()
            expected = (self.num_structures, int(self.material_encoder.output_dim))
            if tuple(cache.shape) != expected:
                raise ValueError(
                    f"{self._encoder_label()} must return one cached vector per structure: "
                    f"{tuple(cache.shape)} != {expected}."
                )
            if cache.device != X.device or cache.dtype != X.dtype:
                raise ValueError(
                    f"Cached {self._structure_feature_label()} must match X's device and dtype."
                )
            if not torch.isfinite(cache).all():
                raise FloatingPointError(
                    f"Cached {self._structure_feature_label()} contain non-finite values."
                )
            self._material_feature_cache = cache
            self._material_feature_cache_versions = self._encoder_parameter_versions()
        return cache

    def _on_encoder_training_policy_change(self) -> None:
        """Discard frozen features before an encoder policy change."""

        self.clear_material_feature_cache()

    def validate_input(self, X: Tensor) -> None:
        _validate_structure_model_inputs(
            X,
            num_structures=self.num_structures,
            input_dim=self.input_dim,
            encoder_name=self.encoder_name,
        )

    def _material_features(self, X: Tensor) -> Tensor:
        """Return selected structure features with leading input shape restored."""

        leading_shape = X.shape[:-1]
        flat_X = X.reshape(-1, self.input_dim)
        structure_index_tensor = flat_X[:, 0].round().to(dtype=torch.long)
        cached_bank = self._cached_material_features(X)
        if cached_bank is None:
            structure_indices = structure_index_tensor.detach().cpu().tolist()
            structures = [self.structure_inputs[index] for index in structure_indices]
            material_features = self._encode_structure_inputs(structures)
        else:
            material_features = cached_bank.index_select(
                0,
                structure_index_tensor.to(device=cached_bank.device),
            )
        if material_features.device != X.device or material_features.dtype != X.dtype:
            raise ValueError(
                f"{self._structure_feature_label()} must match X's device and dtype."
            )
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


__all__ = [
    "_StructureGPFeatureExtractor",
    "_resolve_structure_input_transform",
    "_validate_structure_bank",
    "_validate_structure_model_inputs",
]
