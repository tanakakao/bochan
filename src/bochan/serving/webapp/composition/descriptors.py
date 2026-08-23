"""Differentiable composition descriptor augmentation for the Web workbench.

Descriptors are derived from composition decision coordinates inside a BoTorch
InputTransform. They therefore participate in model fitting, posterior
prediction, and acquisition evaluation without becoming independent optimizer
decision variables.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import log
from typing import Any

import torch
from botorch.models.transforms.input import ChainedInputTransform, InputTransform, Normalize
from torch import Tensor

from bochan.composition import ATOMIC_NUMBERS, ATOMIC_WEIGHTS, TorchSimplexTransform

_ALLOWED_STATISTICS = ("mean", "std", "min", "max", "range")
_BUILTIN_PROPERTIES: dict[str, Mapping[str, float]] = {
    "atomic_number": ATOMIC_NUMBERS,
    "atomic_weight": ATOMIC_WEIGHTS,
}


def _unique_strings(values: Sequence[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _resolved_sequence(
    value: Any,
    *,
    default: Sequence[str],
) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [item for item in value.replace(",", " ").split() if item]
    return _unique_strings(value)


def _reference_index(config: Mapping[str, Any], elements: Sequence[str]) -> int | None:
    if str(config["representation"]).lower() != "alr":
        return None
    reference = config.get("reference_element")
    if reference in (None, ""):
        return len(elements) - 1
    if reference not in elements:
        raise ValueError(
            f"reference_element {reference!r} is not included in elements."
        )
    return list(elements).index(str(reference))


def _property_table(
    config: Mapping[str, Any],
    elements: Sequence[str],
) -> tuple[list[str], Tensor]:
    properties = _resolved_sequence(
        config.get("descriptor_properties"),
        default=("atomic_number", "atomic_weight"),
    )
    custom = dict(config.get("element_properties") or {})
    columns: list[list[float]] = []
    for property_name in properties:
        if property_name in _BUILTIN_PROPERTIES:
            values = _BUILTIN_PROPERTIES[property_name]
        elif property_name in custom:
            values = dict(custom[property_name])
        else:
            raise KeyError(f"Unknown elemental property {property_name!r}.")
        missing = [element for element in elements if element not in values]
        if missing:
            raise KeyError(
                f"Property {property_name!r} is missing for elements {missing!r}."
            )
        columns.append([float(values[element]) for element in elements])
    if not columns:
        return properties, torch.empty((len(elements), 0), dtype=torch.double)
    return properties, torch.tensor(columns, dtype=torch.double).transpose(0, 1)


def descriptor_feature_names(
    config: Mapping[str, Any],
) -> list[str]:
    """Return deterministic derived descriptor feature names."""

    prefix = str(config["column"])
    properties = _resolved_sequence(
        config.get("descriptor_properties"),
        default=("atomic_number", "atomic_weight"),
    )
    statistics = _resolved_sequence(
        config.get("descriptor_statistics"),
        default=_ALLOWED_STATISTICS,
    )
    unknown_statistics = set(statistics) - set(_ALLOWED_STATISTICS)
    if unknown_statistics:
        raise ValueError(
            f"Unknown descriptor statistics: {sorted(unknown_statistics)!r}."
        )
    names = [
        f"{prefix}__descriptor__{property_name}__{statistic}"
        for property_name in properties
        for statistic in statistics
    ]
    if bool(config.get("descriptor_include_num_elements", True)):
        names.append(f"{prefix}__descriptor__num_elements")
    if bool(config.get("descriptor_include_mixing_entropy", True)):
        names.append(f"{prefix}__descriptor__mixing_entropy")
    return names


def descriptor_bounds(
    config: Mapping[str, Any],
    *,
    property_values: Tensor,
) -> Tensor:
    """Return physically safe bounds for every derived descriptor."""

    properties = _resolved_sequence(
        config.get("descriptor_properties"),
        default=("atomic_number", "atomic_weight"),
    )
    statistics = _resolved_sequence(
        config.get("descriptor_statistics"),
        default=_ALLOWED_STATISTICS,
    )
    if property_values.shape != (len(config["elements"]), len(properties)):
        raise ValueError("property_values shape does not match elements/properties.")

    lower: list[float] = []
    upper: list[float] = []
    for property_index, _property_name in enumerate(properties):
        values = property_values[:, property_index]
        minimum = float(values.min())
        maximum = float(values.max())
        spread = maximum - minimum
        for statistic in statistics:
            if statistic in {"mean", "min", "max"}:
                lower.append(minimum)
                upper.append(maximum)
            elif statistic == "std":
                lower.append(0.0)
                upper.append(max(spread / 2.0, 1e-12))
            elif statistic == "range":
                lower.append(0.0)
                upper.append(max(spread, 1e-12))
            else:
                raise ValueError(f"Unknown descriptor statistic {statistic!r}.")

    n_elements = len(config["elements"])
    if bool(config.get("descriptor_include_num_elements", True)):
        lower.append(1.0)
        upper.append(float(n_elements))
    if bool(config.get("descriptor_include_mixing_entropy", True)):
        lower.append(0.0)
        upper.append(max(log(n_elements), 1e-12))
    if not lower:
        return torch.empty((2, 0), dtype=property_values.dtype)
    return property_values.new_tensor([lower, upper])


class CompositionDescriptorInputTransform(InputTransform):
    """Append elemental-property descriptors derived from composition coordinates."""

    is_one_to_many = False
    composition_indices: Tensor
    component_weights: Tensor
    property_values: Tensor

    def __init__(
        self,
        *,
        input_dim: int,
        composition_indices: Sequence[int],
        n_components: int,
        method: str,
        reference_index: int | None,
        component_weights: Tensor,
        property_values: Tensor,
        statistics: Sequence[str],
        include_num_elements: bool,
        include_mixing_entropy: bool,
        active_threshold: float = 1e-10,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        indices = [int(index) for index in composition_indices]
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("composition_indices must be non-empty and unique.")
        if min(indices) < 0 or max(indices) >= input_dim:
            raise ValueError("composition_indices contains an out-of-range index.")

        simplex = TorchSimplexTransform(
            n_components=n_components,
            method=method,
            reference_index=reference_index,
        )
        if simplex.input_dim != len(indices):
            raise ValueError(
                "composition coordinate width does not match representation."
            )
        statistics = tuple(str(value) for value in statistics)
        unknown = set(statistics) - set(_ALLOWED_STATISTICS)
        if unknown:
            raise ValueError(f"Unknown descriptor statistics: {sorted(unknown)!r}.")
        if property_values.ndim != 2 or property_values.shape[0] != n_components:
            raise ValueError(
                "property_values must have shape [n_components, n_properties]."
            )
        if component_weights.shape != (n_components,):
            raise ValueError("component_weights must contain one value per component.")
        if (component_weights <= 0).any() or not torch.isfinite(component_weights).all():
            raise ValueError("component_weights must contain finite positive values.")
        if not torch.isfinite(property_values).all():
            raise ValueError("property_values must contain finite values.")
        if active_threshold < 0:
            raise ValueError("active_threshold must be non-negative.")

        self.input_dim = int(input_dim)
        self.simplex = simplex
        self.statistics = statistics
        self.include_num_elements = bool(include_num_elements)
        self.include_mixing_entropy = bool(include_mixing_entropy)
        self.active_threshold = float(active_threshold)
        self.transform_on_train = True
        self.transform_on_eval = True
        self.transform_on_fantasize = True
        self.register_buffer(
            "composition_indices",
            torch.tensor(indices, dtype=torch.long),
        )
        self.register_buffer("component_weights", component_weights.detach().clone())
        self.register_buffer("property_values", property_values.detach().clone())

    @property
    def descriptor_dim(self) -> int:
        return (
            int(self.property_values.shape[-1]) * len(self.statistics)
            + int(self.include_num_elements)
            + int(self.include_mixing_entropy)
        )

    def _atomic_fractions(self, X: Tensor) -> Tensor:
        indices = self.composition_indices.to(device=X.device)
        coordinates = X.index_select(-1, indices)
        basis_fractions = self.simplex(coordinates)
        weights = self.component_weights.to(dtype=X.dtype, device=X.device)
        atomic_values = basis_fractions / weights
        return atomic_values / atomic_values.sum(dim=-1, keepdim=True)

    def _property_descriptors(self, fractions: Tensor) -> list[Tensor]:
        if self.property_values.shape[-1] == 0:
            return []
        values = self.property_values.to(dtype=fractions.dtype, device=fractions.device)
        weights = fractions.unsqueeze(-1)
        selected = values.view(*([1] * (fractions.ndim - 1)), *values.shape)
        mean = (weights * selected).sum(dim=-2)
        variance = (
            weights * (selected - mean.unsqueeze(-2)).square()
        ).sum(dim=-2)
        std = variance.clamp_min(0.0).sqrt()

        active = fractions > self.active_threshold
        expanded_active = active.unsqueeze(-1)
        positive_inf = torch.full_like(selected, float("inf"))
        negative_inf = torch.full_like(selected, -float("inf"))
        minimum = torch.where(expanded_active, selected, positive_inf).amin(dim=-2)
        maximum = torch.where(expanded_active, selected, negative_inf).amax(dim=-2)
        range_values = maximum - minimum

        by_statistic = {
            "mean": mean,
            "std": std,
            "min": minimum,
            "max": maximum,
            "range": range_values,
        }
        descriptors: list[Tensor] = []
        for property_index in range(values.shape[-1]):
            for statistic in self.statistics:
                descriptors.append(
                    by_statistic[statistic][..., property_index : property_index + 1]
                )
        return descriptors

    def transform(self, X: Tensor) -> Tensor:
        if not torch.is_tensor(X) or not X.is_floating_point():
            raise TypeError("X must be a floating-point Tensor.")
        if X.ndim == 0 or X.shape[-1] != self.input_dim:
            raise ValueError(
                f"X width must equal input_dim: {X.shape[-1] if X.ndim else 0} != {self.input_dim}."
            )
        if not torch.isfinite(X).all():
            raise ValueError("X must contain only finite values.")

        fractions = self._atomic_fractions(X)
        descriptors = self._property_descriptors(fractions)
        if self.include_num_elements:
            descriptors.append(
                (fractions > self.active_threshold).sum(dim=-1, keepdim=True).to(X)
            )
        if self.include_mixing_entropy:
            positive = fractions.clamp_min(torch.finfo(fractions.dtype).tiny)
            descriptors.append(
                -(fractions * positive.log()).sum(dim=-1, keepdim=True)
            )
        if not descriptors:
            return X
        return torch.cat((X, *descriptors), dim=-1)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, descriptor_dim={self.descriptor_dim}, "
            f"method={self.simplex.method!r}"
        )


def build_composition_descriptor_input_transform(
    *,
    feature_names: Sequence[str],
    bounds: Tensor,
    categorical_idx: Sequence[int] | None,
    config: Mapping[str, Any],
    normalize: bool,
) -> tuple[InputTransform, list[str], Tensor]:
    """Build a derived-descriptor transform and augmented normalization bounds."""

    names = [str(name) for name in feature_names]
    coordinate_names = [
        str(name)
        for name in (config.get("feature_names") or ())
        if "__descriptor__" not in str(name)
    ]
    missing = [name for name in coordinate_names if name not in names]
    if missing:
        raise ValueError(
            f"Composition coordinates are missing from model features: {missing!r}."
        )
    elements = [str(value) for value in config["elements"]]
    property_names, property_values = _property_table(config, elements)
    statistics = _resolved_sequence(
        config.get("descriptor_statistics"),
        default=_ALLOWED_STATISTICS,
    )
    descriptor_names = descriptor_feature_names(config)
    if not descriptor_names:
        raise ValueError(
            "Composition descriptors are enabled but no descriptor features were selected."
        )

    reference_index = _reference_index(config, elements)
    component_weights = bounds.new_ones(len(elements))
    if str(config["normalization"]).lower() == "weight_fraction":
        component_weights = bounds.new_tensor(
            [ATOMIC_WEIGHTS[element] for element in elements]
        )
    descriptor_transform = CompositionDescriptorInputTransform(
        input_dim=len(names),
        composition_indices=[names.index(name) for name in coordinate_names],
        n_components=len(elements),
        method=str(config["representation"]),
        reference_index=reference_index,
        component_weights=component_weights,
        property_values=property_values.to(bounds),
        statistics=statistics,
        include_num_elements=bool(
            config.get("descriptor_include_num_elements", True)
        ),
        include_mixing_entropy=bool(
            config.get("descriptor_include_mixing_entropy", True)
        ),
    ).to(bounds)

    derived_bounds = descriptor_bounds(
        {**dict(config), "descriptor_properties": property_names},
        property_values=property_values.to(bounds),
    )
    augmented_bounds = torch.cat((bounds, derived_bounds.to(bounds)), dim=-1)

    if not normalize:
        return descriptor_transform, descriptor_names, augmented_bounds

    raw_categorical = {int(index) for index in (categorical_idx or ())}
    continuous_indices = [
        index
        for index in range(augmented_bounds.shape[-1])
        if index not in raw_categorical
    ]
    normalization = Normalize(
        d=int(augmented_bounds.shape[-1]),
        bounds=augmented_bounds,
        indices=continuous_indices,
    ).to(bounds)
    chained = ChainedInputTransform(
        descriptors=descriptor_transform,
        normalize=normalization,
    )
    return chained, descriptor_names, augmented_bounds


__all__ = [
    "CompositionDescriptorInputTransform",
    "build_composition_descriptor_input_transform",
    "descriptor_bounds",
    "descriptor_feature_names",
]
