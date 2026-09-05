"""Shared data contract for multi-fidelity feature axes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


def _normalize_feature_indices(
    indices: Sequence[int],
    *,
    d: int,
    label: str,
) -> tuple[int, ...]:
    if d <= 0:
        raise ValueError("Input dimension d must be positive.")

    normalized: list[int] = []
    seen: set[int] = set()
    for raw_index in indices:
        index = int(raw_index)
        if index < 0:
            index += int(d)
        if index < 0 or index >= int(d):
            raise ValueError(f"Invalid {label} dim {raw_index} for input dim {d}.")
        if index in seen:
            raise ValueError(f"Duplicate {label} dim {raw_index} resolves to feature {index}.")
        seen.add(index)
        normalized.append(index)
    return tuple(normalized)


@dataclass(frozen=True)
class FidelitySpec:
    """Describe which input columns represent fidelity.

    ``fidelity_features`` accepts one or more feature indices. Negative indices
    remain unresolved until the public input dimensionality is known.
    ``target_fidelities`` keys use the same indexing convention.
    """

    fidelity_features: tuple[int, ...]
    target_fidelities: Mapping[int, float] | None = None

    def __post_init__(self) -> None:
        features = tuple(int(index) for index in self.fidelity_features)
        if not features:
            raise ValueError("fidelity_features must contain at least one feature index.")
        if len(set(features)) != len(features):
            raise ValueError("fidelity_features must not contain duplicate indices.")
        object.__setattr__(self, "fidelity_features", features)

        if self.target_fidelities is not None:
            targets = {int(index): float(value) for index, value in self.target_fidelities.items()}
            if any(not torch.isfinite(torch.tensor(value)) for value in targets.values()):
                raise ValueError("target_fidelities values must be finite.")
            object.__setattr__(self, "target_fidelities", targets)

    def resolve(
        self,
        d: int,
        *,
        cat_dims: Sequence[int] | None = None,
        bounds: Tensor | None = None,
        single_fidelity_only: bool = False,
    ) -> ResolvedFidelitySpec:
        """Resolve negative indices and validate the feature contract.

        ``single_fidelity_only`` remains available as a compatibility guard for
        callers whose algorithm still supports exactly one fidelity dimension.
        The shared Phase-59 model contract itself accepts multiple dimensions.
        """

        fidelity_features = _normalize_feature_indices(
            self.fidelity_features,
            d=int(d),
            label="fidelity",
        )
        if single_fidelity_only and len(fidelity_features) != 1:
            raise ValueError("This operation supports exactly one continuous fidelity feature.")

        categorical = ()
        if cat_dims is not None:
            categorical = _normalize_feature_indices(cat_dims, d=int(d), label="categorical")
            overlap = sorted(set(fidelity_features).intersection(categorical))
            if overlap:
                raise ValueError(
                    "Categorical and fidelity features must be disjoint; "
                    f"overlap={overlap}."
                )

        normalized_targets: dict[int, float] | None = None
        if self.target_fidelities is not None:
            normalized_targets = {}
            for raw_index, target in self.target_fidelities.items():
                resolved_index = _normalize_feature_indices(
                    (raw_index,), d=int(d), label="target fidelity"
                )[0]
                if resolved_index not in fidelity_features:
                    raise ValueError(
                        f"target_fidelities key {raw_index} resolves to feature "
                        f"{resolved_index}, which is not a fidelity feature."
                    )
                if resolved_index in normalized_targets:
                    raise ValueError(
                        "target_fidelities contains duplicate keys after negative-index resolution."
                    )
                normalized_targets[resolved_index] = float(target)

        normalized_bounds = None
        if bounds is not None:
            normalized_bounds = torch.as_tensor(bounds)
            if normalized_bounds.ndim != 2 or normalized_bounds.shape != (2, int(d)):
                raise ValueError(f"bounds must have shape [2, {d}].")
            if not torch.isfinite(normalized_bounds).all():
                raise ValueError("bounds must contain only finite values.")
            if bool((normalized_bounds[0] > normalized_bounds[1]).any()):
                raise ValueError("Each lower bound must be <= its upper bound.")
            if normalized_targets is not None:
                for index, target in normalized_targets.items():
                    lower = float(normalized_bounds[0, index])
                    upper = float(normalized_bounds[1, index])
                    if target < lower or target > upper:
                        raise ValueError(
                            f"Target fidelity {target} for feature {index} is outside "
                            f"bounds [{lower}, {upper}]."
                        )

        return ResolvedFidelitySpec(
            fidelity_features=fidelity_features,
            target_fidelities=normalized_targets,
            categorical_features=categorical,
        )


@dataclass(frozen=True)
class ResolvedFidelitySpec:
    """Dimension-resolved fidelity metadata used by models and optimizers."""

    fidelity_features: tuple[int, ...]
    target_fidelities: Mapping[int, float] | None = None
    categorical_features: tuple[int, ...] = ()

    @property
    def primary_fidelity_feature(self) -> int:
        """Return the fidelity index for algorithms restricted to one dimension."""

        if len(self.fidelity_features) != 1:
            raise ValueError("Expected exactly one resolved fidelity feature.")
        return self.fidelity_features[0]


__all__ = ["FidelitySpec", "ResolvedFidelitySpec"]
