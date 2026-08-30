"""Differentiable bridge between raw composition decisions and model coordinates.

Element support is a property of raw composition fractions.  CLR, ALR, and ILR
coordinates are model representations and must never be interpreted as element
presence indicators.  This module keeps those two spaces explicit:

- decision space: one non-negative fraction per candidate element;
- model space: the fitted composition representation used by the surrogate.

The bridge is intentionally independent of acquisition optimization.  A later
candidate-search layer can optimize in decision space and evaluate the surrogate
through :meth:`decision_to_model` without changing the fitted model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from bochan.composition import TorchSimplexTransform, ilr_basis

_LOG_RATIO_METHODS = {"clr", "alr", "ilr"}


def _normalized_method(value: str) -> str:
    method = str(value).lower()
    if method == "fractions":
        return "none"
    if method not in {"none", *_LOG_RATIO_METHODS}:
        raise ValueError(
            "Composition representation must be fractions, clr, alr, or ilr."
        )
    return method


def _validate_tensor(values: Tensor, expected_dim: int, *, name: str) -> None:
    if not isinstance(values, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not values.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype.")
    if values.ndim < 1 or int(values.shape[-1]) != int(expected_dim):
        raise ValueError(
            f"{name} must have final dimension {expected_dim}, got {tuple(values.shape)}."
        )
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")


def _fractions_to_coordinates(
    fractions: Tensor,
    *,
    method: str,
    pseudocount: float,
    reference_index: int | None,
) -> Tensor:
    """Map raw fractions to model coordinates while preserving autograd."""

    if not isinstance(fractions, Tensor):
        raise TypeError("fractions must be a torch.Tensor.")
    if not fractions.is_floating_point():
        raise TypeError("fractions must have a floating-point dtype.")
    if fractions.ndim < 1 or int(fractions.shape[-1]) < 2:
        raise ValueError("fractions must contain at least two components.")
    if not torch.isfinite(fractions).all():
        raise ValueError("fractions must contain only finite values.")
    if torch.any(fractions < 0):
        raise ValueError("Raw composition fractions must be non-negative.")

    method = _normalized_method(method)
    if method == "none":
        totals = fractions.sum(dim=-1, keepdim=True)
        if torch.any(totals <= 0):
            raise ValueError("Each composition must contain a positive total.")
        return fractions / totals

    pseudocount = float(pseudocount)
    if pseudocount <= 0:
        raise ValueError("Log-ratio composition transforms require pseudocount > 0.")
    closed = fractions + pseudocount
    closed = closed / closed.sum(dim=-1, keepdim=True)
    log_values = torch.log(closed)

    if method == "clr":
        return log_values - log_values.mean(dim=-1, keepdim=True)

    n_components = int(fractions.shape[-1])
    if method == "alr":
        reference = n_components - 1 if reference_index is None else int(reference_index)
        if not 0 <= reference < n_components:
            raise ValueError(
                f"reference_index must be between 0 and {n_components - 1}."
            )
        indices = [index for index in range(n_components) if index != reference]
        return log_values[..., indices] - log_values[..., [reference]]

    basis = torch.as_tensor(
        ilr_basis(n_components),
        dtype=fractions.dtype,
        device=fractions.device,
    )
    return log_values @ basis


@dataclass(frozen=True)
class CompositionRawDecisionBridge:
    """Map a full tabular candidate between raw-fraction and model spaces.

    The fitted composition coordinate columns must form one contiguous block in
    ``model_feature_names``.  The decision-space feature layout replaces that
    block with one raw fraction per fitted element while preserving every other
    process feature and its ordering.
    """

    model_feature_names: tuple[str, ...]
    coordinate_names: tuple[str, ...]
    fraction_names: tuple[str, ...]
    elements: tuple[str, ...]
    method: str
    pseudocount: float
    reference_index: int | None
    coordinate_start: int

    @classmethod
    def from_transformer(
        cls,
        transformer: Any,
        model_feature_names: Sequence[Any],
    ) -> CompositionRawDecisionBridge:
        """Build a bridge from one fitted ``CompositionTransformer``."""

        elements = tuple(str(value) for value in transformer.fitted_elements)
        coordinate_names = tuple(
            str(value) for value in transformer.representation_feature_names_
        )
        model_names = tuple(str(value) for value in model_feature_names)
        if len(elements) < 2:
            raise ValueError("Composition raw-space bridge requires at least two elements.")
        if not coordinate_names:
            raise ValueError("Fitted composition transformer has no model coordinates.")

        positions: list[int] = []
        for name in coordinate_names:
            try:
                positions.append(model_names.index(name))
            except ValueError as exc:
                raise KeyError(
                    f"Composition model coordinate {name!r} is missing from feature_names."
                ) from exc
        start = positions[0]
        expected = list(range(start, start + len(coordinate_names)))
        if positions != expected:
            raise ValueError(
                "Composition model coordinates must be contiguous in the tabular feature layout."
            )

        simplex = getattr(transformer, "simplex_transform_", None)
        if simplex is None:
            raise RuntimeError("Composition transformer must be fitted before building a raw bridge.")
        method = _normalized_method(getattr(simplex, "method", transformer.representation))
        reference_index = getattr(simplex, "reference_index", None)
        fraction_names = tuple(
            f"{transformer.prefix}__fraction__{element}" for element in elements
        )
        return cls(
            model_feature_names=model_names,
            coordinate_names=coordinate_names,
            fraction_names=fraction_names,
            elements=elements,
            method=method,
            pseudocount=float(getattr(simplex, "pseudocount", transformer.pseudocount)),
            reference_index=None if reference_index is None else int(reference_index),
            coordinate_start=start,
        )

    @property
    def coordinate_width(self) -> int:
        return len(self.coordinate_names)

    @property
    def fraction_width(self) -> int:
        return len(self.fraction_names)

    @property
    def model_dim(self) -> int:
        return len(self.model_feature_names)

    @property
    def decision_dim(self) -> int:
        return self.model_dim - self.coordinate_width + self.fraction_width

    @property
    def coordinate_stop(self) -> int:
        return self.coordinate_start + self.coordinate_width

    @property
    def fraction_slice(self) -> slice:
        return slice(self.coordinate_start, self.coordinate_start + self.fraction_width)

    @property
    def fraction_indices(self) -> tuple[int, ...]:
        return tuple(range(self.fraction_slice.start, self.fraction_slice.stop))

    @property
    def decision_feature_names(self) -> tuple[str, ...]:
        return (
            self.model_feature_names[: self.coordinate_start]
            + self.fraction_names
            + self.model_feature_names[self.coordinate_stop :]
        )

    @property
    def process_index_map(self) -> dict[int, int]:
        """Map non-composition model indices to decision-space indices."""

        shift = self.fraction_width - self.coordinate_width
        mapping: dict[int, int] = {}
        for index in range(self.model_dim):
            if self.coordinate_start <= index < self.coordinate_stop:
                continue
            mapping[index] = index if index < self.coordinate_start else index + shift
        return mapping

    def decision_to_model(self, values: Tensor) -> Tensor:
        """Transform raw decision candidates into fitted surrogate coordinates."""

        _validate_tensor(values, self.decision_dim, name="decision values")
        fractions = values[..., self.fraction_slice]
        coordinates = _fractions_to_coordinates(
            fractions,
            method=self.method,
            pseudocount=self.pseudocount,
            reference_index=self.reference_index,
        )
        return torch.cat(
            (
                values[..., : self.coordinate_start],
                coordinates,
                values[..., self.fraction_slice.stop :],
            ),
            dim=-1,
        )

    def model_to_decision(self, values: Tensor) -> Tensor:
        """Convert fitted surrogate coordinates to closed raw fractions.

        For log-ratio representations this inverse cannot recover structural
        zeros because pseudocount replacement is intentionally lossy.  Exact
        element support therefore remains owned by decision space.
        """

        _validate_tensor(values, self.model_dim, name="model values")
        coordinates = values[..., self.coordinate_start : self.coordinate_stop]
        inverse = TorchSimplexTransform(
            self.fraction_width,
            method=self.method,
            reference_index=self.reference_index,
        ).to(dtype=values.dtype, device=values.device)
        fractions = inverse(coordinates)
        return torch.cat(
            (
                values[..., : self.coordinate_start],
                fractions,
                values[..., self.coordinate_stop :],
            ),
            dim=-1,
        )

    def fraction_values(self, values: Tensor) -> Tensor:
        """Return the raw fraction block from decision-space candidates."""

        _validate_tensor(values, self.decision_dim, name="decision values")
        return values[..., self.fraction_slice]

    def decision_bounds(
        self,
        model_bounds: Tensor,
        *,
        component_bounds: Mapping[str, Sequence[float]] | None = None,
        total: float = 1.0,
    ) -> Tensor:
        """Replace model-coordinate bounds with normalized raw component bounds."""

        if not isinstance(model_bounds, Tensor):
            raise TypeError("model_bounds must be a torch.Tensor.")
        if model_bounds.ndim != 2 or tuple(model_bounds.shape) != (2, self.model_dim):
            raise ValueError(
                f"model_bounds must have shape (2, {self.model_dim}), got {tuple(model_bounds.shape)}."
            )
        if not model_bounds.is_floating_point() or not torch.isfinite(model_bounds).all():
            raise ValueError("model_bounds must be a finite floating-point tensor.")
        total = float(total)
        if total <= 0:
            raise ValueError("Composition total must be positive.")

        configured = dict(component_bounds or {})
        lower: list[float] = []
        upper: list[float] = []
        for element in self.elements:
            pair = tuple(configured.get(element, (0.0, total)))
            if len(pair) != 2:
                raise ValueError(f"Bounds for {element!r} must contain two values.")
            low, high = map(float, pair)
            if low < 0 or high < low or high > total:
                raise ValueError(
                    f"Invalid composition bounds for {element!r}: {(low, high)!r}."
                )
            lower.append(low / total)
            upper.append(high / total)

        raw_lower = model_bounds.new_tensor(lower)
        raw_upper = model_bounds.new_tensor(upper)
        return torch.cat(
            (
                model_bounds[:, : self.coordinate_start],
                torch.stack((raw_lower, raw_upper)),
                model_bounds[:, self.coordinate_stop :],
            ),
            dim=1,
        )


__all__ = ["CompositionRawDecisionBridge"]
